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
    lokasi_target = float(
        models.target_encoder.transform(
            pd.DataFrame({"Lokasi": [lokasi]})
        )["Lokasi"].iloc[0]
    )

    row = {
        "Kamar Tidur": kamar_tidur,
        "Kamar Mandi": kamar_mandi,
        "Garasi": garasi,
        "Luas Tanah": luas_tanah,
        "Luas Bangunan": luas_bangunan,
        "Lokasi_Target": lokasi_target,
    }
    df = pd.DataFrame([row])

    # Tambahkan fitur yang sama dengan training
    df['total_kamar']        = df['Kamar Tidur'] + df['Kamar Mandi']
    df['kamar_ratio']        = df['Kamar Tidur'] / (df['Kamar Mandi'] + 1)
    df['garasi_kamar']       = df['Garasi'] * df['total_kamar']
    df['luxury_score']       = df['Kamar Tidur'] * df['Kamar Mandi'] * (df['Garasi'] + 1)
    df['rasio_bang_tanah']   = df['Luas Bangunan'] / (df['Luas Tanah'] + 1)
    df['luas_per_kamar']     = df['Luas Bangunan'] / (df['total_kamar'] + 1)
    df['luas_total']         = df['Luas Tanah'] + df['Luas Bangunan']
    df['log_luas_tanah']     = np.log1p(df['Luas Tanah'])
    df['log_luas_bangunan']  = np.log1p(df['Luas Bangunan'])
    df['log_lokasi']         = np.log1p(df['Lokasi_Target'])
    df['lokasi_x_kamar']     = df['Lokasi_Target'] * df['total_kamar']
    df['lokasi_x_garasi']    = df['Lokasi_Target'] * df['Garasi']
    df['lokasi_x_luxury']    = df['log_lokasi'] * df['luxury_score']
    df['lokasi_x_luas_bang'] = df['log_lokasi'] * df['log_luas_bangunan']
    df['lokasi_per_kamar']   = df['Lokasi_Target'] / (df['total_kamar'] + 1)

    return df[models.meta_regresi["fitur"]]


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
    lt_log = np.log1p(luas_tanah)
    lb_log = np.log1p(luas_bangunan)
    lt_raw = luas_tanah if luas_tanah > 0 else 1   # untuk harga per m2
    lb_raw = luas_bangunan if luas_bangunan > 0 else 1
    total_kamar = kamar_tidur + kamar_mandi

    row: dict = {
        "Kamar Tidur": kamar_tidur,
        "Kamar Mandi": kamar_mandi,
        "Garasi": garasi,
        "Luas Tanah": lt_log,
        "Luas Bangunan": lb_log,
        "rasio_bangunan_tanah": lb_log / lt_log,
        "total_kamar": total_kamar,
        "harga_per_m2_tanah": harga / lt_raw,
        "harga_per_m2_bangunan": harga / lb_raw,
        "luas_total": lt_raw + lb_raw,
        "kamar_per_luas": total_kamar / lb_raw,
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

    lokasi_enc = float(
        models.target_encoder.transform(
            pd.DataFrame({"Lokasi": [lokasi]})
        )["Lokasi"].iloc[0]
    )

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
