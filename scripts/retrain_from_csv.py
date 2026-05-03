"""
Standalone retraining script — mereplikasi persis notebook regresio_v1.2.ipynb.

Preprocessing (persis notebook):
  1. Parsing harga string -> float
  2. Parsing luas dengan regex
  3. Drop NaN
  4. Filter outlier KHUSUS REGRESI (bukan IQR standar clustering):
     - LB > 10, LT > 10
     - Hapus kavling: ~(LT > 400 & LB < 150)
     - LB <= 600, LT <= 1000
     - KT <= 8, KM <= 8, Garasi <= 6
     - Harga >= 200jt
     - Filter quantile global [1%, 99%] pada Harga
     - Filter per lokasi (5%-95%) untuk lokasi >= 10 data
     - Filter harga per m2 (5%-95%)
  5. Split train/test SEBELUM encoding (hindari leakage)
  6. TargetEncoder (smoothing=10) untuk Lokasi
  7. Feature engineering (add_features)
  8. Split per segmen: batas = median(Harga) dari data bersih
  9. Sample weight 2x untuk data ekstrem
  10. CatBoostRegressor (iterations=5000, lr=0.02, depth=7)

Fitur yang dilatih (21 total):
  Base (6): Kamar Tidur, Kamar Mandi, Garasi, Luas Tanah, Luas Bangunan, Lokasi_Target
  Engineered (15): total_kamar, kamar_ratio, garasi_kamar, luxury_score,
                   rasio_bang_tanah, luas_per_kamar, luas_total,
                   log_luas_tanah, log_luas_bangunan, log_lokasi,
                   lokasi_x_kamar, lokasi_x_garasi, lokasi_x_luxury,
                   lokasi_x_luas_bang, lokasi_per_kamar

Jalankan dari direktori notifications/:
    python scripts/retrain_from_csv.py
"""
from __future__ import annotations

import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from category_encoders import TargetEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
# Regresi menggunakan Raw CSV dan melakukan preprocessing sendiri
# (pipeline cleaning berbeda dari clustering)
DATA_PATH    = BASE_DIR / "data" / "Raw-Final-Rumah.csv"
MODELS_DIR   = BASE_DIR / "models"
METADATA_DIR = BASE_DIR / "metadata"

FEATURE_NAMES = [
    # Base
    "Kamar Tidur", "Kamar Mandi", "Garasi",
    "Luas Tanah", "Luas Bangunan", "Lokasi_Target",
    # Engineered
    "total_kamar", "kamar_ratio", "garasi_kamar", "luxury_score",
    "rasio_bang_tanah", "luas_per_kamar", "luas_total",
    "log_luas_tanah", "log_luas_bangunan", "log_lokasi",
    "lokasi_x_kamar", "lokasi_x_garasi", "lokasi_x_luxury",
    "lokasi_x_luas_bang", "lokasi_per_kamar",
]


# ── Step 1: Load & parse raw data ──────────────────────────────────────────
def parse_harga(h) -> float | None:
    """Persis notebook: parse string harga 'X juta' / 'X miliar' -> float."""
    if pd.isna(h):
        return None
    h = str(h).replace(',', '.').strip()
    match = re.search(r'[\d.]+', h)
    if not match:
        return None
    angka = float(match.group())
    if 'miliar' in h.lower():
        return angka * 1_000_000_000
    elif 'juta' in h.lower():
        return angka * 1_000_000
    return angka


def parse_luas(x) -> float | None:
    """Persis notebook: parse string luas -> float."""
    match = re.search(r'[\d.]+', str(x))
    return float(match.group()) if match else None


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

    # Parse Harga (persis notebook — handle 'juta' / 'miliar' string)
    if df["Harga"].dtype == object:
        df["Harga"] = df["Harga"].apply(parse_harga)
    df["Harga"] = pd.to_numeric(df["Harga"], errors="coerce")

    # Parse numerik
    df["Kamar Tidur"] = pd.to_numeric(df["Kamar Tidur"], errors="coerce")
    df["Kamar Mandi"] = pd.to_numeric(df["Kamar Mandi"], errors="coerce")
    df["Garasi"]      = pd.to_numeric(df["Garasi"], errors="coerce").fillna(0)

    # Parse Luas (persis notebook — bisa berupa string "120 m²")
    if df["Luas Tanah"].dtype == object:
        df["Luas Tanah"] = df["Luas Tanah"].apply(parse_luas)
    if df["Luas Bangunan"].dtype == object:
        df["Luas Bangunan"] = df["Luas Bangunan"].apply(parse_luas)
    df["Luas Tanah"]    = pd.to_numeric(df["Luas Tanah"], errors="coerce")
    df["Luas Bangunan"] = pd.to_numeric(df["Luas Bangunan"], errors="coerce")

    # Drop NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Harga", "Kamar Tidur", "Kamar Mandi", "Garasi",
                            "Luas Tanah", "Luas Bangunan"])

    print(f"[Data] {len(df)} rows after basic parsing.")
    return df


# ── Step 2: Outlier removal KHUSUS REGRESI ─────────────────────────────────
def outlier_filter_regresi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mereplikasi persis outlier removal notebook regresio_v1.2.ipynb.
    BERBEDA dari clustering (clustering pakai IQR standar).
    """
    df = df.copy()
    before = len(df)

    # Hapus luas tidak wajar
    df = df[df["Luas Bangunan"] > 10]
    df = df[df["Luas Tanah"] > 10]

    # Hapus anomali: tanah sangat luas tapi bangunan kecil (kavling)
    df = df[~((df["Luas Tanah"] > 400) & (df["Luas Bangunan"] < 150))]

    # Hapus anomali bangunan ekstrem
    df = df[df["Luas Bangunan"] <= 600]
    df = df[df["Luas Tanah"] <= 1000]

    # Hapus spec ekstrem (kemungkinan kos/apartemen/ruko)
    df = df[df["Kamar Tidur"] <= 8]
    df = df[df["Kamar Mandi"] <= 8]
    df = df[df["Garasi"] <= 6]

    # Hapus harga terlalu murah untuk rumah (kemungkinan salah input)
    df = df[df["Harga"] >= 200_000_000]

    # Filter harga global [1%, 99%]
    Q1_g, Q3_g = df["Harga"].quantile([0.01, 0.99])
    df = df[(df["Harga"] >= Q1_g) & (df["Harga"] <= Q3_g)]

    # Filter outlier per lokasi (5%-95%) untuk lokasi >= 10 data
    def filter_lokasi_outlier(group):
        if len(group) < 10:
            return group
        q1 = group["Harga"].quantile(0.05)
        q3 = group["Harga"].quantile(0.95)
        return group[(group["Harga"] >= q1) & (group["Harga"] <= q3)]

    df = df.groupby("Lokasi", group_keys=False).apply(filter_lokasi_outlier)

    # Filter harga per m2 tidak wajar (5%-95%)
    df["harga_per_m2"] = df["Harga"] / df["Luas Bangunan"]
    p5, p95 = df["harga_per_m2"].quantile([0.05, 0.95])
    df = df[(df["harga_per_m2"] >= p5) & (df["harga_per_m2"] <= p95)]
    df = df.drop(columns=["harga_per_m2"])

    print(f"[Outlier] Setelah filter khusus regresi: {len(df)} rows (removed {before - len(df)})")
    return df.reset_index(drop=True)


# ── Step 3: Feature engineering (add_features dari notebook) ───────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Persis add_features() di notebook regresio_v1.2.ipynb."""
    df = df.copy()
    total_kamar = df["Kamar Tidur"] + df["Kamar Mandi"]

    # Fitur kamar
    df["total_kamar"]      = total_kamar
    df["kamar_ratio"]      = df["Kamar Tidur"] / (df["Kamar Mandi"] + 1)
    df["garasi_kamar"]     = df["Garasi"] * total_kamar
    df["luxury_score"]     = df["Kamar Tidur"] * df["Kamar Mandi"] * (df["Garasi"] + 1)

    # Fitur luas
    df["rasio_bang_tanah"] = df["Luas Bangunan"] / (df["Luas Tanah"] + 1)
    df["luas_per_kamar"]   = df["Luas Bangunan"] / (total_kamar + 1)
    df["luas_total"]       = df["Luas Tanah"] + df["Luas Bangunan"]
    df["log_luas_tanah"]   = np.log1p(df["Luas Tanah"])
    df["log_luas_bangunan"]= np.log1p(df["Luas Bangunan"])

    # Fitur lokasi
    df["log_lokasi"]         = np.log1p(df["Lokasi_Target"])
    df["lokasi_x_kamar"]     = df["Lokasi_Target"] * total_kamar
    df["lokasi_x_garasi"]    = df["Lokasi_Target"] * df["Garasi"]
    df["lokasi_x_luxury"]    = df["log_lokasi"] * df["luxury_score"]
    df["lokasi_x_luas_bang"] = df["log_lokasi"] * df["log_luas_bangunan"]

    # Signal harga/m² dari lokasi
    df["lokasi_per_kamar"]   = df["Lokasi_Target"] / (total_kamar + 1)

    return df


def build_features(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, TargetEncoder]:
    """
    Split dulu, baru fit encoder pada train — hindari data leakage.
    Persis alur notebook: encoder fit pada X_train, transform X_test.
    """
    X_train = df_train[["Kamar Tidur", "Kamar Mandi", "Garasi",
                         "Luas Tanah", "Luas Bangunan", "Lokasi"]].copy()
    X_test  = df_test[["Kamar Tidur", "Kamar Mandi", "Garasi",
                        "Luas Tanah", "Luas Bangunan", "Lokasi"]].copy()
    y_train = df_train["Harga"]
    y_test  = df_test["Harga"]

    # Target encoding (smoothing=10, persis notebook)
    te = TargetEncoder(cols=["Lokasi"], smoothing=10)
    te.fit(X_train["Lokasi"], y_train)
    X_train["Lokasi_Target"] = te.transform(X_train["Lokasi"])["Lokasi"]
    X_test["Lokasi_Target"]  = te.transform(X_test["Lokasi"])["Lokasi"]
    X_train = X_train.drop(columns=["Lokasi"])
    X_test  = X_test.drop(columns=["Lokasi"])

    # add_features
    X_train = add_features(X_train)
    X_test  = add_features(X_test)

    return X_train[FEATURE_NAMES], X_test[FEATURE_NAMES], te


# ── Step 4: Evaluasi ───────────────────────────────────────────────────────
def regression_metrics(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, mape, r2


# ── Step 5: Save artifacts ─────────────────────────────────────────────────
def save_artifacts(
    model_low: CatBoostRegressor, model_high: CatBoostRegressor,
    encoder: TargetEncoder,
    batas: float, mape_low: float, mape_high: float,
    mape_overall: float, r2_overall: float,
    mae_overall: float, rmse_overall: float,
    n_train_low: int, n_test_low: int,
    n_train_high: int, n_test_high: int,
) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    model_low.save_model(str(MODELS_DIR / "model_low.cbm"))
    print(f"[Save] model_low.cbm saved  (MAPE {mape_low:.4f}%)")

    model_high.save_model(str(MODELS_DIR / "model_high.cbm"))
    print(f"[Save] model_high.cbm saved (MAPE {mape_high:.4f}%)")

    with open(MODELS_DIR / "target_encoder.pkl", "wb") as fh:
        pickle.dump(encoder, fh)
    print("[Save] target_encoder.pkl saved")

    # Update metadata_regresi.json
    meta_path = METADATA_DIR / "metadata_regresi.json"
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    meta["created_at"]       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["batas_segmen"]     = float(batas)
    meta["fitur"]            = FEATURE_NAMES
    meta["model_low"]["deskripsi"]   = f"Harga <= {batas:,.0f}"
    meta["model_low"]["jumlah_train"]= n_train_low
    meta["model_low"]["jumlah_test"] = n_test_low
    meta["model_low"]["mape"]        = round(mape_low, 4)
    meta["model_high"]["deskripsi"]  = f"Harga > {batas:,.0f}"
    meta["model_high"]["jumlah_train"]= n_train_high
    meta["model_high"]["jumlah_test"] = n_test_high
    meta["model_high"]["mape"]        = round(mape_high, 4)
    meta["overall"]["r2"]   = round(r2_overall, 4)
    meta["overall"]["mape"] = round(mape_overall, 4)
    meta["overall"]["mae"]  = round(mae_overall, 2)
    meta["overall"]["rmse"] = round(rmse_overall, 2)

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=4, ensure_ascii=False)
    print("[Save] metadata_regresi.json updated")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  House Price Intelligence — Retraining Script")
    print("  Mereplikasi: regresio_v1.2.ipynb")
    print("=" * 60)

    # 1. Load raw data
    df = load_data()

    # 2. Outlier removal KHUSUS REGRESI
    df_clean = outlier_filter_regresi(df)
    print(f"[Data] {len(df_clean)} baris setelah filter. Korelasi Harga:")
    print(df_clean[["Harga","Kamar Tidur","Kamar Mandi","Garasi",
                     "Luas Tanah","Luas Bangunan"]].corr()["Harga"])

    # 3. Split — persis notebook (test_size=0.2, random_state=42)
    df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)
    print(f"\n[Split] Train: {len(df_train):,} | Test: {len(df_test):,}")

    # 4. Feature engineering (encoder fit only on train)
    X_train, X_test, encoder = build_features(df_train, df_test)
    y_train = df_train["Harga"]
    y_test  = df_test["Harga"]

    # 5. Batas segmentasi = median Harga dari data bersih (persis notebook)
    batas = float(df_clean["Harga"].median())
    print(f"\n[Split] Batas segmentasi (median): {batas:,.0f}")

    mask_train_low  = y_train.values <= batas
    mask_train_high = y_train.values >  batas
    mask_test_low   = y_test.values  <= batas
    mask_test_high  = y_test.values  >  batas

    # 6. Sample weights (bobot 2x untuk data ekstrem — persis notebook)
    w_low  = np.where(y_train.values[mask_train_low]  < batas * 0.5, 2.0, 1.0)
    w_high = np.where(y_train.values[mask_train_high] > batas * 2,   2.0, 1.0)

    # 7. Params persis notebook
    params = dict(
        iterations=5000, learning_rate=0.02, depth=7,
        l2_leaf_reg=3, loss_function="RMSE",
        early_stopping_rounds=200, random_seed=42, verbose=500,
    )

    print(f"\n[Train] Model Low (Harga <= {batas:,.0f}), {mask_train_low.sum()} baris")
    m_low = CatBoostRegressor(**params)
    m_low.fit(
        X_train[mask_train_low], np.log1p(y_train.values[mask_train_low]),
        sample_weight=w_low,
        eval_set=(X_test[mask_test_low], np.log1p(y_test.values[mask_test_low])),
        use_best_model=True,
    )

    print(f"\n[Train] Model High (Harga > {batas:,.0f}), {mask_train_high.sum()} baris")
    m_high = CatBoostRegressor(**params)
    m_high.fit(
        X_train[mask_train_high], np.log1p(y_train.values[mask_train_high]),
        sample_weight=w_high,
        eval_set=(X_test[mask_test_high], np.log1p(y_test.values[mask_test_high])),
        use_best_model=True,
    )

    # 8. Evaluasi overall (persis notebook — predict per segmen lalu gabung)
    pred_low  = np.expm1(m_low.predict(X_test[mask_test_low]))
    pred_high = np.expm1(m_high.predict(X_test[mask_test_high]))

    y_pred_final = np.empty(len(y_test))
    y_pred_final[mask_test_low]  = pred_low
    y_pred_final[mask_test_high] = pred_high

    rmse, mae, mape_overall, r2 = regression_metrics(y_test.values, y_pred_final)
    mape_low  = float(np.mean(np.abs((y_test.values[mask_test_low]  - pred_low)  / y_test.values[mask_test_low]))  * 100)
    mape_high = float(np.mean(np.abs((y_test.values[mask_test_high] - pred_high) / y_test.values[mask_test_high])) * 100)

    print(f"\n{'='*55}")
    print("EVALUASI REGRESI")
    print(f"{'='*55}")
    print(f"  Overall  → R²: {r2:.4f} | MAPE: {mape_overall:.2f}%")
    print(f"  MAPE segmen bawah (≤{batas/1e9:.2f}M): {mape_low:.2f}%")
    print(f"  MAPE segmen atas  (>{batas/1e9:.2f}M): {mape_high:.2f}%")
    print(f"{'='*55}\n")

    # 9. Save
    print("-- Saving artifacts --")
    save_artifacts(
        m_low, m_high, encoder,
        batas, mape_low, mape_high,
        mape_overall, r2, mae, rmse,
        int(mask_train_low.sum()), int(mask_test_low.sum()),
        int(mask_train_high.sum()), int(mask_test_high.sum()),
    )

    print("\n[OK] Retraining regresi selesai! Restart server agar model baru di-load.")
    print("   -> python -m uvicorn server:app --reload\n")


if __name__ == "__main__":
    main()
