# predictor.py
"""
Core prediction logic — wraps model inference and returns structured dicts.
Each function is called directly by the MCP tool and also by the Kafka ML pods.
"""
from __future__ import annotations

import time

import numpy as np

from services.feature_engineer import (
    engineer_classification_features,
    engineer_clustering_features,
    engineer_regression_features,
)
from services.model_loader import models


# ── Regression ─────────────────────────────────────────────────────────────

def predict_price(
    kamar_tidur: int,
    kamar_mandi: int,
    garasi: int,
    luas_tanah: float,
    luas_bangunan: float,
    lokasi: str,
) -> dict:
    """
    Dual-model CatBoost price regression.

    Strategy:
    1. Perform a first-pass using model_low for an initial price estimate
    2. If the estimate ≤ segment_threshold → use model_low (final)
        If the estimate  > segment_threshold → use model_high (final)

    The segment threshold is read from meta_regression["segment_threshold"] (single source of truth),
    ensuring it always stays consistent with the trained models.
    """
    t0 = time.perf_counter()

    # Read segment threshold from metadata — NOT hardcoded
    batas_segmen = float(models.meta_regresi["batas_segmen"])
    # Blending zone: ±5% around the threshold to avoid cliff-effect
    blend_margin = batas_segmen * 0.05

    X = engineer_regression_features(
        kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, lokasi
    )

    # First-pass using model_low to determine the segment
    log_harga_low  = float(models.model_low.predict(X)[0])
    harga_low      = float(np.expm1(log_harga_low))
    log_harga_high = float(models.model_high.predict(X)[0])
    harga_high     = float(np.expm1(log_harga_high))

    if harga_low < batas_segmen - blend_margin:
        # Clearly in the lower segment.
        harga_final    = harga_low
        model_digunakan = "model_low"
        mape = models.meta_regresi["model_low"]["mape"]
    elif harga_low > batas_segmen + blend_margin:
        # Clearly in the upper segment.
        harga_final    = harga_high
        model_digunakan = "model_high"
        mape = models.meta_regresi["model_high"]["mape"]
    else:
        # Blending zone: weighted average based on distance to the boundary
        ratio_high = (harga_low - (batas_segmen - blend_margin)) / (2 * blend_margin)
        ratio_high = max(0.0, min(1.0, ratio_high))
        harga_final    = harga_low * (1 - ratio_high) + harga_high * ratio_high
        model_digunakan = "blended"
        mape = (models.meta_regresi["model_low"]["mape"] * (1 - ratio_high)
                + models.meta_regresi["model_high"]["mape"] * ratio_high)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "harga_estimasi": round(harga_final),
        "harga_estimasi_format": _format_rupiah(harga_final),
        "model_digunakan": model_digunakan,
        "mape_persen": mape,
        "batas_segmen_idr": batas_segmen,
        "latency_ms": latency_ms,
    }


# ── Classification ─────────────────────────────────────────────────────────

def classify_segment(
    kamar_tidur: int,
    kamar_mandi: int,
    garasi: int,
    luas_tanah: float,
    luas_bangunan: float,
    lokasi: str,
    harga: float | None = None,
) -> dict:
    """
    4-class price segment classifier (CatBoost).
    If `price` is not provided, the price is first estimated using a regression model, 
    then used as a feature for classification.
    """
    t0 = time.perf_counter()

    harga_sumber = "input"
    if harga is None:
        reg = predict_price(kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, lokasi)
        harga = reg["harga_estimasi"]
        harga_sumber = "estimated_by_regression"

    X = engineer_classification_features(
        kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, lokasi, harga
    )

    kelas_id = int(models.model_clf.predict(X.values).flatten()[0])
    proba = models.model_clf.predict_proba(X.values)[0].tolist()
    kelas_label = models.meta_klasifikasi["kelas"][str(kelas_id)]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "kelas_id": kelas_id,
        "kelas_label": kelas_label,
        "probabilitas": {
            models.meta_klasifikasi["kelas"][str(i)]: round(p, 4)
            for i, p in enumerate(proba)
        },
        "harga_digunakan": round(harga),
        "harga_sumber": harga_sumber,
        "akurasi_model": models.meta_klasifikasi["performa"]["accuracy"],
        "latency_ms": latency_ms,
    }


# ── Clustering ─────────────────────────────────────────────────────────────

def cluster_property(
    luas_tanah: float,
    luas_bangunan: float,
    kamar_tidur: int,
    kamar_mandi: int,
    lokasi: str,
    harga: float | None = None,
    garasi: int = 0,
) -> dict:
    """
    KMeans + UMAP clustering (6 clusters).

    If `price` is not provided, the price is first estimated using a regression model,
    then used as a feature for clustering.
    """
    t0 = time.perf_counter()

    harga_sumber = "input"
    if harga is None:
        reg = predict_price(
            kamar_tidur, kamar_mandi, garasi=garasi,
            luas_tanah=luas_tanah, luas_bangunan=luas_bangunan, lokasi=lokasi
        )
        harga = reg["harga_estimasi"]
        harga_sumber = "estimated_by_regression"

    X_raw = engineer_clustering_features(
        harga, luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, lokasi
    )

    # Scale → UMAP → KMeans
    X_scaled = models.scaler.transform(X_raw)
    X_umap   = models.umap.transform(X_scaled)
    cluster_id = int(models.kmeans.predict(X_umap)[0])

    label_map: dict = models.meta_clustering.get("label_map", {})
    cluster_label = label_map.get(str(cluster_id), f"Cluster-{cluster_id}")

    # Find cluster summary from metadata
    summary = next(
        (c for c in models.meta_clustering.get("cluster_summary", [])
         if c["cluster_id"] == cluster_id),
        {},
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "cluster_summary": summary,
        "harga_digunakan": round(harga),
        "harga_sumber": harga_sumber,
        "silhouette_score": models.meta_clustering["evaluasi"]["silhouette_score"],
        "latency_ms": latency_ms,
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _format_rupiah(nilai: float) -> str:
    """Format numbers into a readable Indonesian Rupiah string (e.g., “Rp 1.25 Billion”)"""
    if nilai >= 1_000_000_000:
        return f"Rp {nilai / 1_000_000_000:.2f} Miliar"
    elif nilai >= 1_000_000:
        return f"Rp {nilai / 1_000_000:.1f} Juta"
    return f"Rp {int(nilai):,}"
