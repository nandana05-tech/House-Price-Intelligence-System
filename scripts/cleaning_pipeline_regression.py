"""
Clean RawProperty → CleanPropertyRegression.
Mirrors EXACT preprocessing from regresio_v1.2.ipynb.

Cleaning steps:
  1. Drop NaN
  2. Hapus luas tidak wajar (LB>10, LT>10, LB≤600, LT≤1000)
  3. Hapus anomali kavling (LT>400 & LB<150)
  4. Hapus spec ekstrem (KT≤8, KM≤8, Garasi≤6)
  5. Hapus harga < Rp 200 juta
  6. Filter harga global Q0.01–Q0.99
  7. Filter outlier per lokasi Q0.05–Q0.95 (skip if < 10 rows)
  8. Filter harga per m² P5–P95

Run:
    docker exec -e PYTHONPATH=/app hpi_api python scripts/cleaning_pipeline_regression.py
"""
import os

import numpy as np
import pandas as pd
import psycopg2

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

df_clean = df_clean[df_clean["Luas Bangunan"] > 10]
df_clean = df_clean[df_clean["Luas Tanah"] > 10]
df_clean = df_clean[~((df_clean["Luas Tanah"] > 400) & (df_clean["Luas Bangunan"] < 150))]
df_clean = df_clean[df_clean["Luas Bangunan"] <= 600]
df_clean = df_clean[df_clean["Luas Tanah"] <= 1000]
df_clean = df_clean[df_clean["Kamar Tidur"] <= 8]
df_clean = df_clean[df_clean["Kamar Mandi"] <= 8]
df_clean = df_clean[df_clean["Garasi"] <= 6]
df_clean = df_clean[df_clean["Harga"] >= 200_000_000]

q1, q3 = df_clean["Harga"].quantile([0.01, 0.99])
df_clean = df_clean[(df_clean["Harga"] >= q1) & (df_clean["Harga"] <= q3)]


def filter_lokasi_outlier(group: pd.DataFrame) -> pd.DataFrame:
    if len(group) < 10:
        return group
    q_low = group["Harga"].quantile(0.05)
    q_high = group["Harga"].quantile(0.95)
    return group[(group["Harga"] >= q_low) & (group["Harga"] <= q_high)]


df_clean = df_clean.groupby("Lokasi", group_keys=False).apply(filter_lokasi_outlier)

df_clean["harga_per_m2"] = df_clean["Harga"] / df_clean["Luas Bangunan"]
p5, p95 = df_clean["harga_per_m2"].quantile([0.05, 0.95])
df_clean = df_clean[(df_clean["harga_per_m2"] >= p5) & (df_clean["harga_per_m2"] <= p95)]
df_clean = df_clean.drop(columns=["harga_per_m2"])

print(f"Clean (regression): {len(df_clean)} rows")

with conn.cursor() as cur:
    cur.execute('TRUNCATE TABLE "CleanPropertyRegression" RESTART IDENTITY;')
    for _, row in df_clean.iterrows():
        cur.execute(
            """
            INSERT INTO "CleanPropertyRegression"
            (harga, "kamarTidur", "kamarMandi", garasi, "luasTanah", "luasBangunan", lokasi)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                float(row["Harga"]),
                int(row["Kamar Tidur"]),
                int(row["Kamar Mandi"]),
                int(row["Garasi"]),
                float(row["Luas Tanah"]),
                float(row["Luas Bangunan"]),
                str(row["Lokasi"]),
            ),
        )
    conn.commit()

print(f"Saved {len(df_clean)} rows to CleanPropertyRegression.")
conn.close()
