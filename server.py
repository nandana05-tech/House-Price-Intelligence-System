"""
MCP Server — LLM-Powered House Price Intelligence System
=========================================================
Exposes three MCP tools callable by any LLM via StreamableHTTP:
  • predict_price       — dual-model CatBoost regression
  • classify_segment    — 4-class price segment classification
  • cluster_property    — KMeans + UMAP market clustering

Each tool also publishes a fire-and-forget event to Kafka (when KAFKA_ENABLED=true)
so ML pods can audit, re-score, and log to MLflow asynchronously.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP

load_dotenv()

# ── Boot: load all models before accepting requests ───────────────────────
from services.model_loader import models

models.load()

# ── Kafka producer (optional) ─────────────────────────────────────────────
KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
_kafka_producer = None


async def _get_producer():
    global _kafka_producer
    if _kafka_producer is None and KAFKA_ENABLED:
        from aiokafka import AIOKafkaProducer

        _kafka_producer = AIOKafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        await _kafka_producer.start()
    return _kafka_producer


async def _publish(topic: str, payload: dict) -> None:
    """Fire-and-forget Kafka publish — never blocks the MCP response."""
    if not KAFKA_ENABLED:
        return
    try:
        producer = await _get_producer()
        await producer.send_and_wait(topic, payload)
    except Exception as exc:  # noqa: BLE001
        print(f"[Kafka] publish failed ({topic}): {exc}")


# ── MCP App ───────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="House Price Intelligence",
    instructions=(
        "You are a real-estate AI assistant for the Depok area. "
        "Use predict_price to estimate property values, classify_segment to determine "
        "the price tier, and cluster_property to find the market cluster. "
        "Always answer in Bahasa Indonesia unless the user writes in English."
    ),
    host="0.0.0.0",
    port=int(os.getenv("MCP_PORT", "8000")),
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Predict Price
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def predict_price(
    kamar_tidur: Annotated[int, "Jumlah kamar tidur (1-10)"],
    kamar_mandi: Annotated[int, "Jumlah kamar mandi (1-10)"],
    garasi: Annotated[int, "Kapasitas garasi (0-5 mobil)"],
    luas_tanah: Annotated[float, "Luas tanah dalam meter persegi"],
    luas_bangunan: Annotated[float, "Luas bangunan dalam meter persegi"],
    lokasi: Annotated[str, "Nama kecamatan/area di Depok (contoh: Cinere, Sawangan, Beji)"],
    ctx: Context,
) -> dict:
    """
    Estimasi harga rumah di Depok menggunakan dual-model CatBoost.
    Model dipilih otomatis: model_low (≤1.2M IDR) atau model_high (>1.2M IDR).
    """
    await ctx.info(f"Menghitung estimasi harga untuk properti di {lokasi}...")
    await ctx.report_progress(20, 100)

    from services.predictor import predict_price as _predict

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _predict(kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, lokasi),
    )

    await ctx.report_progress(80, 100)

    # Publish audit event to Kafka asynchronously
    await _publish(
        "property.prediction.regression",
        {
            "tool": "predict_price",
            "input": {
                "kamar_tidur": kamar_tidur,
                "kamar_mandi": kamar_mandi,
                "garasi": garasi,
                "luas_tanah": luas_tanah,
                "luas_bangunan": luas_bangunan,
                "lokasi": lokasi,
            },
            "output": result,
        },
    )

    await ctx.info(f"Selesai — estimasi: {result['harga_estimasi_format']}")
    await ctx.report_progress(100, 100)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Classify Segment
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def classify_segment(
    kamar_tidur: Annotated[int, "Jumlah kamar tidur"],
    kamar_mandi: Annotated[int, "Jumlah kamar mandi"],
    garasi: Annotated[int, "Kapasitas garasi"],
    luas_tanah: Annotated[float, "Luas tanah (m²)"],
    luas_bangunan: Annotated[float, "Luas bangunan (m²)"],
    lokasi: Annotated[str, "Nama kecamatan/area di Depok"],
    harga: Annotated[Optional[float], "Harga properti (IDR). Jika kosong, akan diestimasi otomatis."] = None,
    ctx: Context = None,
) -> dict:
    """
    Klasifikasi segmen harga properti ke 4 kelas:
      0 = Murah (≤745 juta)  |  1 = Menengah (745 juta–1.3 miliar)
      2 = Atas (1.3–2.645 miliar)  |  3 = Mewah (>2.645 miliar)
    Jika harga tidak diisi, akan dihitung otomatis via model regresi.
    """
    if ctx:
        await ctx.info("Mengklasifikasikan segmen properti...")
        await ctx.report_progress(20, 100)

    from services.predictor import classify_segment as _classify

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _classify(kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, lokasi, harga),
    )

    if ctx:
        await ctx.report_progress(80, 100)

    await _publish(
        "property.prediction.classification",
        {
            "tool": "classify_segment",
            "input": {
                "kamar_tidur": kamar_tidur, "kamar_mandi": kamar_mandi,
                "garasi": garasi, "luas_tanah": luas_tanah,
                "luas_bangunan": luas_bangunan, "lokasi": lokasi, "harga": harga,
            },
            "output": result,
        },
    )

    if ctx:
        await ctx.info(f"Segmen: {result['kelas_label']}")
        await ctx.report_progress(100, 100)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — Cluster Property
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def cluster_property(
    luas_tanah: Annotated[float, "Luas tanah (m²)"],
    luas_bangunan: Annotated[float, "Luas bangunan (m²)"],
    kamar_tidur: Annotated[int, "Jumlah kamar tidur"],
    kamar_mandi: Annotated[int, "Jumlah kamar mandi"],
    lokasi: Annotated[str, "Nama kecamatan/area di Depok"],
    harga: Annotated[Optional[float], "Harga properti (IDR). Jika kosong, akan diestimasi otomatis."] = None,
    ctx: Context = None,
) -> dict:
    """
    Tentukan klaster pasar properti menggunakan KMeans + UMAP (6 klaster):
    Budget → Affordable → Mid-Market → Premium → Luxury → Ultra-Luxury.
    Berguna untuk rekomendasi properti serupa dan analisis segmen pasar.
    """
    if ctx:
        await ctx.info("Menentukan klaster pasar properti...")
        await ctx.report_progress(20, 100)

    from services.predictor import cluster_property as _cluster

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _cluster(luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, lokasi, harga),
    )

    if ctx:
        await ctx.report_progress(80, 100)

    await _publish(
        "property.prediction.clustering",
        {
            "tool": "cluster_property",
            "input": {
                "luas_tanah": luas_tanah, "luas_bangunan": luas_bangunan,
                "kamar_tidur": kamar_tidur, "kamar_mandi": kamar_mandi,
                "lokasi": lokasi, "harga": harga,
            },
            "output": result,
        },
    )

    if ctx:
        await ctx.info(f"Klaster: {result['cluster_label']}")
        await ctx.report_progress(100, 100)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")

    if transport == "streamable-http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
