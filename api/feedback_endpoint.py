"""
Feedback endpoint — receives user price corrections and publishes them
to Kafka for the feedback consumer to persist into PostgreSQL.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from aiokafka import AIOKafkaProducer
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

load_dotenv()

from kafka.topics import FEEDBACK_EVENTS

router = APIRouter(tags=["Feedback"])

# Lazy Kafka producer (shared within process)
_producer: AIOKafkaProducer | None = None


async def _get_producer() -> AIOKafkaProducer:
    global _producer
    if _producer is None:
        _producer = AIOKafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        await _producer.start()
    return _producer


# ── Request / Response schemas ─────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    prediction_id: Optional[str] = Field(None, description="ID prediksi MCP yang dikoreksi (opsional)")
    kamar_tidur: int = Field(..., ge=1, le=20)
    kamar_mandi: int = Field(..., ge=1, le=20)
    garasi: int = Field(..., ge=0, le=10)
    luas_tanah: float = Field(..., gt=0)
    luas_bangunan: float = Field(..., gt=0)
    lokasi: str = Field(..., min_length=2)
    harga_prediksi: float = Field(..., gt=0, description="Harga yang diprediksi model (IDR)")
    harga_asli: float = Field(..., gt=0, description="Harga aktual / koreksi user (IDR)")
    sumber: str = Field("user_feedback", description="Sumber feedback")

    @field_validator("harga_asli", "harga_prediksi")
    @classmethod
    def harga_reasonable(cls, v: float) -> float:
        if v < 10_000_000:
            raise ValueError("Harga terlalu kecil (minimum Rp 10 juta)")
        if v > 500_000_000_000:
            raise ValueError("Harga terlalu besar")
        return v


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    selisih_persen: float


# ── Route ──────────────────────────────────────────────────────────────────

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    """
    Submit a price correction for a property prediction.
    The feedback is published to Kafka and asynchronously stored in PostgreSQL.
    Accumulated feedback triggers automatic model retraining when threshold is reached.
    """
    selisih = abs(body.harga_asli - body.harga_prediksi) / body.harga_prediksi * 100

    payload = {
        **body.model_dump(),
        "selisih_persen": round(selisih, 2),
    }

    try:
        producer = await _get_producer()
        await producer.send_and_wait(FEEDBACK_EVENTS, payload)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Kafka tidak tersedia: {exc}") from exc

    return FeedbackResponse(
        success=True,
        message=f"Feedback diterima. Selisih prediksi: {selisih:.1f}%",
        selisih_persen=round(selisih, 2),
    )
