"""
Feature engineering functions — mirrors the exact transformations used during training
for each of the three model types (regression, classification, clustering).
"""
import numpy as np
import pandas as pd

from services.model_loader import models


# ── Lokasi list (from classification metadata one-hot columns) ─────────────
def _known_locations() -> list[str]:
    """Extract known location names from classification feature metadata."""
    return [
        col.replace("Lokasi_", "")
        for col in models.meta_klasifikasi["fitur"]
        if col.startswith("Lokasi_")
    ]


# ── Regression ─────────────────────────────────────────────────────────────

def engineer_regression_features(
    kamar_tidur: int,
    kamar_mandi: int,
    garasi: int,
    luas_tanah: float,
    luas_bangunan: float,
    lokasi: str,
) -> pd.DataFrame:
    """Return a single-row DataFrame ready for the CatBoost regression models."""
    lokasi_target = models.target_encoder["map"].get(lokasi, models.target_encoder["fallback"])

    row = {
        "Kamar Tidur": kamar_tidur,
        "Kamar Mandi": kamar_mandi,
        "Garasi": garasi,
        "Luas Tanah": luas_tanah,
        "Luas Bangunan": luas_bangunan,
        "Lokasi_Target": lokasi_target,
    }
    return pd.DataFrame([row])[models.meta_regresi["fitur"]]


# ── Classification ─────────────────────────────────────────────────────────

def engineer_classification_features(
    kamar_tidur: int,
    kamar_mandi: int,
    garasi: int,
    luas_tanah: float,
    luas_bangunan: float,
    lokasi: str,
    harga: float,
) -> pd.DataFrame:
    """Return a single-row DataFrame ready for the CatBoost classifier."""
    lt = luas_tanah if luas_tanah > 0 else 1
    lb = luas_bangunan if luas_bangunan > 0 else 1
    total_kamar = kamar_tidur + kamar_mandi

    row: dict = {
        "Kamar Tidur": kamar_tidur,
        "Kamar Mandi": kamar_mandi,
        "Garasi": garasi,
        "Luas Tanah": luas_tanah,
        "Luas Bangunan": luas_bangunan,
        "rasio_bangunan_tanah": lb / lt,
        "total_kamar": total_kamar,
        "harga_per_m2_tanah": harga / lt,
        "harga_per_m2_bangunan": harga / lb,
        "luas_total": lt + lb,
        "kamar_per_luas": total_kamar / lb,
        "garasi_flag": 1 if garasi > 0 else 0,
    }

    # One-hot encode lokasi
    for loc in _known_locations():
        row[f"Lokasi_{loc}"] = 1 if lokasi == loc else 0

    return pd.DataFrame([row])[models.meta_klasifikasi["fitur"]]


# ── Clustering ─────────────────────────────────────────────────────────────

def engineer_clustering_features(
    harga: float,
    luas_tanah: float,
    luas_bangunan: float,
    kamar_tidur: int,
    kamar_mandi: int,
    lokasi: str,
) -> pd.DataFrame:
    """Return a single-row DataFrame ready for the UMAP + KMeans pipeline."""
    lt = luas_tanah if luas_tanah > 0 else 1

    lokasi_enc = models.target_encoder["map"].get(lokasi, models.target_encoder["fallback"])

    row = {
        "log_Harga": np.log1p(harga),
        "log_LT": np.log1p(luas_tanah),
        "log_LB": np.log1p(luas_bangunan),
        "log_Harga_m2": np.log1p(harga / lt),
        "rasio_LB_LT": luas_bangunan / lt,
        "Kamar Tidur": kamar_tidur,
        "Kamar Mandi": kamar_mandi,
        "Lokasi_enc": lokasi_enc,
    }
    return pd.DataFrame([row])[models.meta_clustering["features"]]
