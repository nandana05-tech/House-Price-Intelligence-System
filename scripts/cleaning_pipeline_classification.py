"""
Clean RawProperty → CleanPropertyClassification.
Mirrors EXACT preprocessing from clasifikasi_v1.0.ipynb.

Cleaning steps:
  1. Drop NaN
  2. IQR Clip (NOT drop) on all numeric columns
  3. Label harga: 0=Murah(≤745jt), 1=Menengah(745jt-1.3M),
                  2=Atas(1.3M-2.645M), 3=Mewah(>2.645M)

Run:
    docker exec -e PYTHONPATH=/app hpi_api python scripts/cleaning_pipeline_classification.py
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

num_cols = ["Harga", "Kamar Tidur", "Kamar Mandi", "Garasi", "Luas Tanah", "Luas Bangunan"]
for col in num_cols:
    df[col] = df[col].astype(float)
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def label_harga(harga: float) -> int:
    if harga <= 745_000_000:
        return 0
    if harga <= 1_300_000_000:
        return 1
    if harga <= 2_645_000_000:
        return 2
    return 3


df["kelas_harga"] = df["Harga"].apply(label_harga)
print(f"Clean (classification): {len(df)} rows")

with conn.cursor() as cur:
    cur.execute('TRUNCATE TABLE "CleanPropertyClassification" RESTART IDENTITY;')
    for _, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO "CleanPropertyClassification"
            (harga, "kamarTidur", "kamarMandi", garasi,
             "luasTanah", "luasBangunan", lokasi, "kelasHarga")
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                float(row["Harga"]),
                int(row["Kamar Tidur"]),
                int(row["Kamar Mandi"]),
                int(row["Garasi"]),
                float(row["Luas Tanah"]),
                float(row["Luas Bangunan"]),
                str(row["Lokasi"]),
                int(row["kelas_harga"]),
            ),
        )
    conn.commit()

print(f"Saved {len(df)} rows to CleanPropertyClassification.")
conn.close()
