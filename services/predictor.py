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

# Boundary between low-price and high-price regression model (IDR)
PRICE_THRESHOLD: float = 1_200_000_000.0


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
    Uses model_low for ≤ 1.2B IDR segment, model_high otherwise.
    Returns a first-pass prediction with model_low to determine which model to use.
    """
    t0 = time.perf_counter()
    X = engineer_regression_features(
        kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, lokasi
    )

    # First-pass with low model to determine segment
    log_harga_low = float(models.model_low.predict(X)[0])
    harga_low = float(np.expm1(log_harga_low))

    if harga_low <= PRICE_THRESHOLD:
        harga_final = harga_low
        model_digunakan = "model_low"
        mape = models.meta_regresi["model_low"]["mape"]
    else:
        log_harga_high = float(models.model_high.predict(X)[0])
        harga_final = float(np.expm1(log_harga_high))
        model_digunakan = "model_high"
        mape = models.meta_regresi["model_high"]["mape"]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "harga_estimasi": round(harga_final),
        "harga_estimasi_format": _format_rupiah(harga_final),
        "model_digunakan": model_digunakan,
        "mape_persen": mape,
        "batas_segmen_idr": PRICE_THRESHOLD,
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
    If `harga` is not provided, it is estimated first via the regression model.
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

    kelas_id = int(models.model_clf.predict(X)[0])
    proba = models.model_clf.predict_proba(X)[0].tolist()
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
    garasi: int = 0
) -> dict:
    """
    KMeans + UMAP clustering (6 clusters).
    If `harga` is not provided, it is estimated via regression first.
    """
    t0 = time.perf_counter()

    harga_sumber = "input"
    if harga is None:
        reg = predict_price(kamar_tidur, kamar_mandi, garasi=garasi, luas_tanah=luas_tanah,
                            luas_bangunan=luas_bangunan, lokasi=lokasi)
        harga = reg["harga_estimasi"]
        harga_sumber = "estimated_by_regression"

    X_raw = engineer_clustering_features(
        harga, luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, lokasi
    )

    # Scale → UMAP → KMeans
    X_scaled = models.scaler.transform(X_raw)
    X_umap = models.umap.transform(X_scaled)
    cluster_id = int(models.kmeans.predict(X_umap)[0])

    label_map: dict = models.meta_clustering["label_map"]
    cluster_label = label_map.get(str(cluster_id), f"Cluster-{cluster_id}")

    # Find cluster summary from metadata
    summary = next(
        (c for c in models.meta_clustering["cluster_summary"] if c["cluster_id"] == cluster_id),
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
    """Format angka ke string Rupiah yang mudah dibaca (e.g. 'Rp 1,25 M')."""
    if nilai >= 1_000_000_000:
        return f"Rp {nilai / 1_000_000_000:.2f} Miliar"
    elif nilai >= 1_000_000:
        return f"Rp {nilai / 1_000_000:.1f} Juta"
    return f"Rp {int(nilai):,}"
