"""
Feature engineering functions — mirrors the exact transformations used during training
for each of the three model types (regression, classification, clustering).

Regression    : 21 features — replicates regresio_v1.2.ipynb
                (6 base + 15 engineered, TargetEncoder smoothing=10)
Classification: pd.get_dummies one-hot + ratio features — replicates clasifikasi_v1.0.ipynb
Clustering    : log-transforms + Lokasi_enc — replicates clustering_v1.2.2.ipynb

IMPORTANT NOTE — classification transformation order:
  1. Compute all ratios/interactions from ORIGINAL values (before log)
  2. Then apply log1p to Land Area and Building Area
  (exactly as in the notebook — ratios are computed from raw values, not from log)
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
    """Return a single-row DataFrame ready for the CatBoost regression models.

    Replicating the exact transformations from the regresio_v1.2.ipynb notebook:
      - TargetEncoder(smoothing=10) for the Lokasi column → Lokasi_Target
      - add_features(): 15 engineered features on top of 6 base features (total 21)
    The final columns are selected from meta_regresi["fitur"] (single source of truth).
    """
    # Target-encode the location using the encoder that was already fitted during training
    lokasi_target = float(
        models.target_encoder.transform(
            pd.DataFrame({"Lokasi": [lokasi]})
        )["Lokasi"].iloc[0]
    )

    lt = max(luas_tanah, 1.0)
    lb = max(luas_bangunan, 1.0)
    total_kamar = kamar_tidur + kamar_mandi

    row = {
        # ── Base features ──────────────────────────────────────────────
        "Kamar Tidur":        kamar_tidur,
        "Kamar Mandi":        kamar_mandi,
        "Garasi":             garasi,
        "Luas Tanah":         luas_tanah,
        "Luas Bangunan":      luas_bangunan,
        "Lokasi_Target":      lokasi_target,
        # ── Engineered features (15) — exactly as in the add_features() notebook ──
        "total_kamar":        total_kamar,
        "kamar_ratio":        kamar_tidur / (kamar_mandi + 1),
        "garasi_kamar":       garasi * total_kamar,
        "luxury_score":       kamar_tidur * kamar_mandi * (garasi + 1),
        "rasio_bang_tanah":   lb / lt,
        "luas_per_kamar":     lb / (total_kamar + 1),
        "luas_total":         lt + lb,
        "log_luas_tanah":     np.log1p(lt),
        "log_luas_bangunan":  np.log1p(lb),
        "log_lokasi":         np.log1p(lokasi_target),
        "lokasi_x_kamar":     lokasi_target * total_kamar,
        "lokasi_x_garasi":    lokasi_target * garasi,
        "lokasi_x_luxury":    np.log1p(lokasi_target) * kamar_tidur * kamar_mandi * (garasi + 1),
        "lokasi_x_luas_bang": np.log1p(lokasi_target) * np.log1p(lb),
        "lokasi_per_kamar":   lokasi_target / (total_kamar + 1),
    }

    df = pd.DataFrame([row])

    # Select columns according to the metadata order (authoritative list)
    feature_cols = models.meta_regresi["fitur"]
    return df[feature_cols]


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
    """Return a single-row DataFrame ready for the CatBoost classifier.

    Replicating the exact transformations from the clasifikasi_v1.0.ipynb notebook (IMPORTANT ORDER):
      Step 1 — compute all ratios & interactions from ORIGINAL values:
        rasio_bangunan_tanah  = LB_raw / (LT_raw + 1)
        harga_per_m2_tanah    = Harga  / (LT_raw + 1)
        harga_per_m2_bangunan = Harga  / (LB_raw + 1)
        luas_total            = LT_raw + LB_raw
        kamar_per_luas        = total_kamar / (LB_raw + 1)

      Step 2 — apply log1p to Land Area and Building Area
      Step 3 — one-hot encode Lokasi
    """
    # ORIGINAL (raw) values for ratio calculations
    lt_raw = max(luas_tanah, 0)
    lb_raw = max(luas_bangunan, 0)
    total_kamar = kamar_tidur + kamar_mandi

    row: dict = {
        "Kamar Tidur":           kamar_tidur,
        "Kamar Mandi":           kamar_mandi,
        "Garasi":                garasi,
        # Step 2: log1p on area (exactly as in the notebook)
        "Luas Tanah":            np.log1p(lt_raw),
        "Luas Bangunan":         np.log1p(lb_raw),
        # Step 1: ratios calculated from ORIGINAL values (not log-transformed)
        "rasio_bangunan_tanah":  lb_raw / (lt_raw + 1),
        "total_kamar":           total_kamar,
        "harga_per_m2_tanah":    harga / (lt_raw + 1),
        "harga_per_m2_bangunan": harga / (lb_raw + 1),
        "luas_total":            lt_raw + lb_raw,
        "kamar_per_luas":        total_kamar / (lb_raw + 1),
        "garasi_flag":           1 if garasi > 0 else 0,
    }

    # Step 3: one-hot encode lokasi
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
    """Return a single-row DataFrame ready for the UMAP + KMeans pipeline.

    Replicating the exact transformations from the clustering_v1.2.2.ipynb notebook:
      - log transforms for Harga, LT, LB, Harga/m²
      - rasio_LB_LT
      - Lokasi_enc via median log_Harga per kecamatan (lokasi_median_encoder.pkl)
        NOT via TargetEncoder sklearn like in regression.
    """
    lt = luas_tanah if luas_tanah > 0 else 1

    # Use the lokasi_median_encoder (dict) if available — exactly as in the clustering notebook
    if models.lokasi_median_encoder is not None:
        # dict: {lokasi_name: median_log_Harga}
        global_median = float(np.median(list(models.lokasi_median_encoder.values())))
        lokasi_enc = float(models.lokasi_median_encoder.get(lokasi, global_median))
    else:
        # Fallback: regression TargetEncoder (less accurate but prevents crashes)
        lokasi_enc = float(
            models.target_encoder.transform(
                pd.DataFrame({"Lokasi": [lokasi]})
            )["Lokasi"].iloc[0]
        )

    row = {
        "log_Harga":    np.log1p(harga),
        "log_LT":       np.log1p(luas_tanah),
        "log_LB":       np.log1p(luas_bangunan),
        "log_Harga_m2": np.log1p(harga / lt),
        "rasio_LB_LT":  luas_bangunan / lt,
        "Kamar Tidur":  kamar_tidur,
        "Kamar Mandi":  kamar_mandi,
        "Lokasi_enc":   lokasi_enc,
    }
    return pd.DataFrame([row])[models.meta_clustering["features"]]
