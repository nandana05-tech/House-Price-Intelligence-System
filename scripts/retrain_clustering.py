"""
Standalone retraining script untuk model CLUSTERING (KMeans + UMAP).

Mereplikasi persis notebook clustering_v1.2.2.ipynb:
  - Filter harga: 50jt - 10M
  - IQR drop (bukan clip) pada Harga, Luas Tanah, Luas Bangunan
  - Lokasi encoding: median log_Harga per kecamatan (dictionary, BUKAN sklearn TargetEncoder)
  - Features: log_Harga, log_LT, log_LB, log_Harga_m2, rasio_LB_LT, Kamar Tidur, Kamar Mandi, Lokasi_enc
  - StandardScaler -> UMAP(10D) -> KMeans(K=6)
  - Cluster di-urutkan berdasarkan median Harga

PENTING: clustering menggunakan encoder yang BERBEDA dengan regresi.
  - Regresi: sklearn TargetEncoder (smoothing=10) -> target_encoder.pkl
  - Clustering: median log_Harga per kecamatan -> lokasi_median_encoder.pkl (FILE BARU)

Jalankan dari direktori notifications/:
    python scripts/retrain_clustering.py
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
DATA_PATH    = BASE_DIR / "data" / "Raw-Final-Rumah.csv"
MODELS_DIR   = BASE_DIR / "models"
METADATA_DIR = BASE_DIR / "metadata"

N_CLUSTERS = 6
LABEL_MAP = {
    0: "Budget",
    1: "Affordable",
    2: "Mid-Market",
    3: "Premium",
    4: "Luxury",
    5: "Ultra-Luxury",
    6: "Tier-7",
    7: "Tier-8",
}
FEATURES = ["log_Harga", "log_LT", "log_LB", "log_Harga_m2", "rasio_LB_LT", "Kamar Tidur", "Kamar Mandi", "Lokasi_enc"]


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

    # Type conversions
    for col in ["Kamar Tidur", "Kamar Mandi", "Garasi"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col == "Garasi":
            df[col] = df[col].fillna(0)
    for col in ["Luas Tanah", "Luas Bangunan", "Harga"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Harga", "Kamar Tidur", "Kamar Mandi", "Garasi", "Luas Tanah", "Luas Bangunan"])

    for col in ["Kamar Tidur", "Kamar Mandi", "Garasi", "Luas Tanah", "Luas Bangunan", "Harga"]:
        df[col] = df[col].astype(float)

    print(f"[Data] {len(df)} rows after basic cleaning.")
    return df


def outlier_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mereplikasi persis preprocessing clustering_v1.2.2.ipynb:
    1. Filter harga 50jt - 10M
    2. IQR DROP (bukan clip) pada Harga, Luas Tanah, Luas Bangunan
    """
    df = df.copy()
    before = len(df)

    # Filter harga absolut
    df = df[(df["Harga"] >= 50_000_000) & (df["Harga"] <= 10_000_000_000)]
    print(f"[Outlier] Setelah filter harga 50jt-10M: {len(df)} (removed {before - len(df)})")

    # IQR drop
    cols_iqr = ["Harga", "Luas Tanah", "Luas Bangunan"]
    for col in cols_iqr:
        before_col = len(df)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        print(f"[Outlier] {col}: removed {before_col - len(df)} baris")

    print(f"[Data] {len(df)} rows setelah outlier removal.")
    return df


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, dict, StandardScaler]:
    """
    Feature engineering persis clustering_v1.2.2.ipynb.
    Lokasi_enc = median log_Harga per kecamatan (bukan TargetEncoder sklearn!).
    Returns: X_scaled (array), lokasi_median_dict, scaler
    """
    df = df.copy()

    df["log_Harga"]   = np.log1p(df["Harga"])
    df["log_LT"]      = np.log1p(df["Luas Tanah"])
    df["log_LB"]      = np.log1p(df["Luas Bangunan"])
    df["Harga_per_m2"]= df["Harga"] / df["Luas Bangunan"].replace(0, np.nan)
    df["log_Harga_m2"]= np.log1p(df["Harga_per_m2"])
    df["rasio_LB_LT"] = df["Luas Bangunan"] / df["Luas Tanah"].replace(0, np.nan)

    # Lokasi encoding: median log_Harga per kecamatan
    lokasi_median = df.groupby("Lokasi")["log_Harga"].median()
    lokasi_median_dict = lokasi_median.to_dict()
    df["Lokasi_enc"] = df["Lokasi"].map(lokasi_median)

    # Select features dan drop NaN
    X_df = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.loc[X_df.dropna().index].copy()
    X_df = X_df.dropna()
    X = X_df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[Features] Shape: {X_scaled.shape}")
    print(f"[Features] Features: {FEATURES}")

    return X_scaled, df, lokasi_median_dict, scaler


def train_clustering(X_scaled: np.ndarray) -> tuple[np.ndarray, KMeans]:
    """UMAP + KMeans persis notebook."""
    try:
        import umap
    except ImportError:
        print("[ERROR] umap-learn tidak terinstall. Jalankan: pip install umap-learn")
        raise

    print(f"\n-- UMAP reduction (n_components=10) --")
    reducer = umap.UMAP(
        n_components=10,
        n_neighbors=30,
        min_dist=0.0,
        metric="euclidean",
        random_state=42,
    )
    X_umap = reducer.fit_transform(X_scaled)
    print(f"[UMAP] Shape: {X_umap.shape}")

    # Simpan reducer juga
    return X_umap, reducer


def evaluate_and_label(X_umap: np.ndarray, df_clean: pd.DataFrame, km: KMeans) -> pd.DataFrame:
    """Evaluasi clustering dan re-label berdasarkan median Harga."""
    labels = km.labels_
    df_clean = df_clean.copy()
    df_clean["Cluster"] = labels

    # Re-order cluster berdasarkan median Harga (ascending)
    cluster_median = df_clean.groupby("Cluster")["Harga"].median().sort_values()
    rank_map = {old: new for new, old in enumerate(cluster_median.index)}
    df_clean["Cluster"] = df_clean["Cluster"].map(rank_map)

    # Evaluasi
    labels_final = df_clean["Cluster"].values
    sil = silhouette_score(X_umap, labels_final)
    db  = davies_bouldin_score(X_umap, labels_final)
    ch  = calinski_harabasz_score(X_umap, labels_final)

    print(f"\n{'='*55}")
    print("EVALUASI CLUSTERING")
    print(f"{'='*55}")
    print(f"  Silhouette Score     : {sil:.4f}")
    print(f"  Davies-Bouldin Index : {db:.4f}")
    print(f"  Calinski-Harabasz    : {ch:.2f}")
    print(f"{'='*55}\n")

    # Summary per cluster
    summary = df_clean.groupby("Cluster").agg(
        jumlah_data=("Harga", "count"),
        harga_min=("Harga", "min"),
        harga_max=("Harga", "max"),
        harga_median=("Harga", "median"),
        luas_tanah_median=("Luas Tanah", "median"),
        luas_bangunan_median=("Luas Bangunan", "median"),
    ).reset_index()
    summary["label"] = summary["Cluster"].map(LABEL_MAP)
    print(summary.to_string(index=False))

    return df_clean, labels_final, sil, db, ch, summary


def save_artifacts(
    km: KMeans, reducer, scaler: StandardScaler, lokasi_median_dict: dict,
    feature_names: list, sil: float, db: float, ch: float, summary: pd.DataFrame,
) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    # KMeans
    with open(MODELS_DIR / "kmeans_model.pkl", "wb") as fh:
        pickle.dump(km, fh)
    print("[Save] kmeans_model.pkl saved")

    # UMAP reducer
    with open(MODELS_DIR / "umap_reducer.pkl", "wb") as fh:
        pickle.dump(reducer, fh)
    print("[Save] umap_reducer.pkl saved")

    # StandardScaler
    with open(MODELS_DIR / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    print("[Save] scaler.pkl saved")

    # Lokasi median encoder (BARU — khusus clustering)
    with open(MODELS_DIR / "lokasi_median_encoder.pkl", "wb") as fh:
        pickle.dump(lokasi_median_dict, fh)
    print("[Save] lokasi_median_encoder.pkl saved (encoder khusus clustering)")

    # Cluster summary untuk metadata
    cluster_summary = []
    for _, row in summary.iterrows():
        cid = int(row["Cluster"])
        cluster_summary.append({
            "cluster_id": cid,
            "label": LABEL_MAP.get(cid, f"Tier-{cid+1}"),
            "jumlah_data": int(row["jumlah_data"]),
            "harga_min": float(row["harga_min"]),
            "harga_max": float(row["harga_max"]),
            "harga_median": float(row["harga_median"]),
            "luas_tanah_median": float(row["luas_tanah_median"]),
            "luas_bangunan_median": float(row["luas_bangunan_median"]),
        })

    # Metadata
    meta_path = METADATA_DIR / "metadata_clustering.json"
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    meta["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["n_clusters"] = N_CLUSTERS
    meta["features"] = feature_names
    meta["evaluasi"] = {
        "silhouette_score": round(sil, 4),
        "davies_bouldin_index": round(db, 4),
        "calinski_harabasz": round(ch, 2),
    }
    meta["cluster_summary"] = cluster_summary

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print("[Save] metadata_clustering.json updated")


def main() -> None:
    print("=" * 60)
    print("  House Price Intelligence - Retrain Clustering")
    print("  Mereplikasi: clustering_v1.2.2.ipynb")
    print("=" * 60)

    # 1. Load raw data
    df = load_data()

    # 2. Outlier filter (persis notebook clustering)
    df = outlier_filter(df)

    # 3. Feature engineering
    X_scaled, df_clean, lokasi_median_dict, scaler = build_features(df)

    # 4. UMAP
    X_umap, reducer = train_clustering(X_scaled)

    # 5. KMeans final
    print(f"\n-- KMeans (K={N_CLUSTERS}, n_init=50, max_iter=500) --")
    km = KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=50, max_iter=500, random_state=42)
    km.fit(X_umap)

    # 6. Evaluate & label
    df_clean, labels_final, sil, db, ch, summary = evaluate_and_label(X_umap, df_clean, km)

    # Update km labels to use re-ordered labels
    km.labels_ = labels_final

    # 7. Save
    print("\n-- Saving artifacts --")
    save_artifacts(km, reducer, scaler, lokasi_median_dict, FEATURES, sil, db, ch, summary)

    # 8. Simpan data_with_clusters.csv (untuk dipakai retrain regresi)
    data_save_path = BASE_DIR / "data" / "data_with_clusters.csv"
    df_clean["Cluster_Label"] = df_clean["Cluster"].map(LABEL_MAP)
    df_clean.to_csv(data_save_path, index=False)
    print(f"[Save] data_with_clusters.csv saved ({len(df_clean)} rows)")

    print("\n[OK] Retraining clustering selesai! Restart server agar model baru di-load.")
    print("   PENTING: model_loader.py perlu diupdate untuk memuat lokasi_median_encoder.pkl")
    print("   -> python -m uvicorn server:app --reload\n")


if __name__ == "__main__":
    main()
