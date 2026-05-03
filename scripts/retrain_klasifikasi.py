"""
Standalone retraining script untuk model KLASIFIKASI.

Mereplikasi persis notebook clasifikasi_v1.0.ipynb:
  - IQR clip (bukan drop) pada semua kolom numerik
  - Feature engineering dengan harga asli (sebelum log)
  - log1p hanya pada Luas Tanah dan Luas Bangunan
  - pd.get_dummies untuk Lokasi
  - CatBoostClassifier (iterations=1500, depth=8, l2=3)
  - Label: 0=Murah (<=745jt), 1=Menengah (745jt-1.3M), 2=Atas (1.3M-2.645M), 3=Mewah (>2.645M)

Jalankan dari direktori notifications/:
    python scripts/retrain_klasifikasi.py
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
DATA_PATH    = BASE_DIR / "data" / "Raw-Final-Rumah.csv"
MODELS_DIR   = BASE_DIR / "models"
METADATA_DIR = BASE_DIR / "metadata"

# ── Label boundaries ───────────────────────────────────────────────────────
BATAS = [745_000_000, 1_300_000_000, 2_645_000_000]


def label_harga(h: float) -> int:
    if h <= BATAS[0]: return 0
    if h <= BATAS[1]: return 1
    if h <= BATAS[2]: return 2
    return 3


# ── Step 1: Load & parse raw data ──────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print(f"[Data] Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    col_map = {
        "kamar_tidur": "Kamar Tidur", "kamar_mandi": "Kamar Mandi",
        "garasi": "Garasi", "luas_tanah": "Luas Tanah",
        "luas_bangunan": "Luas Bangunan", "lokasi": "Lokasi", "harga": "Harga",
        "kamarTidur": "Kamar Tidur", "kamarMandi": "Kamar Mandi",
        "luasTanah": "Luas Tanah", "luasBangunan": "Luas Bangunan",
        "Kecamatan": "Lokasi",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "Page" in df.columns:
        df = df.drop(columns=["Page"])

    required = ["Kamar Tidur", "Kamar Mandi", "Garasi", "Luas Tanah", "Luas Bangunan", "Lokasi", "Harga"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[Data] Missing columns: {missing}. Got: {list(df.columns)}")

    # Type conversions (persis notebook)
    for col in ["Kamar Tidur", "Kamar Mandi", "Garasi"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col == "Garasi":
            df[col] = df[col].fillna(0)
    for col in ["Luas Tanah", "Luas Bangunan", "Harga"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Harga", "Kamar Tidur", "Kamar Mandi", "Garasi", "Luas Tanah", "Luas Bangunan"])

    for col in ["Kamar Tidur", "Kamar Mandi", "Garasi", "Luas Tanah", "Luas Bangunan", "Harga"]:
        df[col] = df[col].astype(float)

    print(f"[Data] {len(df)} rows after cleaning.")
    return df


# ── Step 2: IQR clip (persis notebook — clip, bukan drop) ─────────────────
def iqr_clip(df: pd.DataFrame) -> pd.DataFrame:
    """IQR clip pada semua kolom numerik (termasuk Harga)."""
    num_cols = ["Harga", "Kamar Tidur", "Kamar Mandi", "Garasi", "Luas Tanah", "Luas Bangunan"]
    df = df.copy()
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df


# ── Step 3: Feature engineering ────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Mereplikasi persis clasifikasi_v1.0.ipynb:
      - rasio/interaksi dihitung dari harga & luas ASLI (setelah clip)
      - log1p hanya pada Luas Tanah & Luas Bangunan
      - Lokasi di-dummies
    Returns X (DataFrame), y (Series), feature_names (list)
    """
    df = df.copy()

    # rasio & interaksi — memakai nilai ASLI
    df["rasio_bangunan_tanah"] = df["Luas Bangunan"] / (df["Luas Tanah"] + 1)
    df["total_kamar"]          = df["Kamar Tidur"] + df["Kamar Mandi"]
    df["harga_per_m2_tanah"]   = df["Harga"] / (df["Luas Tanah"] + 1)
    df["harga_per_m2_bangunan"]= df["Harga"] / (df["Luas Bangunan"] + 1)
    df["luas_total"]           = df["Luas Tanah"] + df["Luas Bangunan"]
    df["kamar_per_luas"]       = df["total_kamar"] / (df["Luas Bangunan"] + 1)
    df["garasi_flag"]          = (df["Garasi"] > 0).astype(int)

    # Label
    df["kelas_harga"] = df["Harga"].apply(label_harga)

    # Pisahkan X dan y
    y = df["kelas_harga"]
    X = df.drop(columns=["Harga", "kelas_harga"])

    # log1p pada luas (persis notebook — dilakukan SETELAH split di notebook,
    # tapi karena kita pakai seluruh data untuk feature, lakukan sekarang)
    X["Luas Tanah"]    = np.log1p(X["Luas Tanah"])
    X["Luas Bangunan"] = np.log1p(X["Luas Bangunan"])

    # Lokasi one-hot
    X = pd.get_dummies(X, columns=["Lokasi"])

    return X, y, list(X.columns)


# ── Step 4: Train ──────────────────────────────────────────────────────────
def train_model(X_train, y_train, X_test, y_test) -> tuple[CatBoostClassifier, dict]:
    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        early_stopping_rounds=100,
        verbose=100,
        random_seed=42,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    return model, report


# ── Step 5: Save ──────────────────────────────────────────────────────────
def save_artifacts(model: CatBoostClassifier, report: dict, feature_names: list) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    model.save_model(str(MODELS_DIR / "model_clf.cbm"))
    print("[Save] model_clf.cbm saved")

    per_kelas = {
        k: {
            "precision": round(v["precision"], 4),
            "recall":    round(v["recall"], 4),
            "f1":        round(v["f1-score"], 4),
            "support":   v["support"],
        }
        for k, v in report.items()
        if k not in ["accuracy", "macro avg", "weighted avg"]
    }

    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fitur": feature_names,
        "kelas": {
            "0": "Murah (<= 745 juta)",
            "1": "Menengah (745 juta - 1.3 miliar)",
            "2": "Atas (1.3 - 2.645 miliar)",
            "3": "Mewah (> 2.645 miliar)",
        },
        "performa": {
            "accuracy":  round(report["accuracy"], 4),
            "macro_f1":  round(report["macro avg"]["f1-score"], 4),
            "per_kelas": per_kelas,
        },
        "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
    }

    meta_path = METADATA_DIR / "metadata_klasifikasi.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=4, ensure_ascii=False)
    print(f"[Save] metadata_klasifikasi.json updated ({len(feature_names)} fitur)")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  House Price Intelligence - Retrain Klasifikasi")
    print("  Mereplikasi: clasifikasi_v1.0.ipynb")
    print("=" * 60)

    df = load_data()
    df = iqr_clip(df)
    X, y, feature_names = build_features(df)

    print(f"\n[Split] Total rows: {len(X):,}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[Split] Train: {len(X_train):,} | Test: {len(X_test):,}\n")

    # Align columns (get_dummies bisa beda kolom antar split)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    model, report = train_model(X_train, y_train, X_test, y_test)

    print("\n-- Saving artifacts --")
    save_artifacts(model, report, list(X_train.columns))

    print("\n[OK] Retraining klasifikasi selesai! Restart server agar model baru di-load.")
    print("   -> python -m uvicorn server:app --reload\n")


if __name__ == "__main__":
    main()
