"""
Clean RawProperty → CleanPropertyClustering.
Mirrors EXACT preprocessing from clustering_v1_2_2.ipynb.

Cleaning steps:
  1. Drop NaN
  2. Filter harga: Rp 50 juta – Rp 10 miliar
  3. IQR Drop (NOT clip) on Harga, Luas Tanah, Luas Bangunan
  4. Log transform: log_Harga, log_LT, log_LB, log_Harga_m2
  5. Derived features: rasio_LB_LT, Harga_per_m2
  6. Target encoding: Lokasi → median log_Harga per kecamatan
  7. Save clustering_encoder.pkl

Run:
    docker exec -e PYTHONPATH=/app hpi_api python scripts/cleaning_pipeline_clustering.py
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

MODELS_DIR = Path("/app/models")
conn = psycopg2.connect(os.getenv("DATABASE_URL"))

df = pd.read_sql('SELECT * FROM "RawProperty"', conn)
df = df.rename(columns={
    "kamarTidur": "Kamar Tidur",
    "kamarMandi": "Kamar Mandi",
    "luasTanah": "Luas Tanah",
    "luasBangunan": "Luas Bangunan",
    "harga": "Harga",
    "lokasi": "Lokasi",
    "garasi": "Garasi",
})
print(f"Raw: {len(df)} rows")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(
    subset=["Harga", "Kamar Tidur", "Kamar Mandi", "Garasi", "Luas Tanah", "Luas Bangunan"]
)
df_clean = df.copy()

df_clean = df_clean[(df_clean["Harga"] >= 50_000_000) & (df_clean["Harga"] <= 10_000_000_000)]

for col in ["Harga", "Luas Tanah", "Luas Bangunan"]:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    before = len(df_clean)
    df_clean = df_clean[(df_clean[col] >= q1 - 1.5 * iqr) & (df_clean[col] <= q3 + 1.5 * iqr)]
    print(f"  {col}: removed {before - len(df_clean)} rows")

df_clean = df_clean.copy()
df_clean["log_Harga"] = np.log1p(df_clean["Harga"])
df_clean["log_LT"] = np.log1p(df_clean["Luas Tanah"])
df_clean["log_LB"] = np.log1p(df_clean["Luas Bangunan"])
df_clean["Harga_per_m2"] = df_clean["Harga"] / df_clean["Luas Bangunan"].replace(0, np.nan)
df_clean["log_Harga_m2"] = np.log1p(df_clean["Harga_per_m2"])
df_clean["rasio_LB_LT"] = df_clean["Luas Bangunan"] / df_clean["Luas Tanah"].replace(0, np.nan)

lokasi_median = df_clean.groupby("Lokasi")["log_Harga"].median()
df_clean["Lokasi_enc"] = df_clean["Lokasi"].map(lokasi_median)

features = [
    "log_Harga",
    "log_LT",
    "log_LB",
    "log_Harga_m2",
    "rasio_LB_LT",
    "Kamar Tidur",
    "Kamar Mandi",
    "Lokasi_enc",
]
x = df_clean[features].replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.loc[x.dropna().index].copy()
print(f"Clean (clustering): {len(df_clean)} rows")

lokasi_enc_map = df_clean.groupby("Lokasi")["Lokasi_enc"].mean().to_dict()
fallback = float(df_clean["Lokasi_enc"].mean())
with open(MODELS_DIR / "clustering_encoder.pkl", "wb") as f:
    pickle.dump({"map": lokasi_enc_map, "fallback": fallback}, f)
print(f"Saved clustering_encoder.pkl ({len(lokasi_enc_map)} locations)")

with conn.cursor() as cur:
    cur.execute('TRUNCATE TABLE "CleanPropertyClustering" RESTART IDENTITY;')
    for _, row in df_clean.iterrows():
        cur.execute(
            """
            INSERT INTO "CleanPropertyClustering"
            (harga, "kamarTidur", "kamarMandi", garasi,
             "luasTanah", "luasBangunan", lokasi,
             "logHarga", "logLT", "logLB",
             "hargaPerM2", "logHargaM2", "rasioLBLT", "lokasi_enc")
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                float(row["Harga"]),
                int(row["Kamar Tidur"]),
                int(row["Kamar Mandi"]),
                int(row["Garasi"]),
                float(row["Luas Tanah"]),
                float(row["Luas Bangunan"]),
                str(row["Lokasi"]),
                float(row["log_Harga"]),
                float(row["log_LT"]),
                float(row["log_LB"]),
                float(row["Harga_per_m2"]),
                float(row["log_Harga_m2"]),
                float(row["rasio_LB_LT"]),
                float(row["Lokasi_enc"]),
            ),
        )
    conn.commit()

print(f"Saved {len(df_clean)} rows to CleanPropertyClustering.")
conn.close()
