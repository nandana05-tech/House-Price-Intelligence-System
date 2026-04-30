"""
Load raw CSV into PostgreSQL RawProperty table.
Mirrors parse_harga() and parse_luas() from all three training notebooks.

Run:
    docker exec -e PYTHONPATH=/app hpi_api python scripts/ingest_raw_data.py
"""
import os, re
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path

DATA_PATH = Path("/app/data/Raw-Final-Rumah.csv")

def parse_harga(h):
    if pd.isna(h): return None
    h = str(h).replace(',', '.').strip()
    match = re.search(r'[\d.]+', h)
    if not match: return None
    angka = float(match.group())
    if 'miliar' in h.lower(): return angka * 1_000_000_000
    elif 'juta' in h.lower(): return angka * 1_000_000
    return angka

def parse_luas(x):
    match = re.search(r'[\d.]+', str(x))
    return float(match.group()) if match else None

print("Loading raw CSV...")
df = pd.read_csv(DATA_PATH)
print(f"  Raw rows: {len(df)}")

df['Harga'] = df['Harga'].apply(parse_harga)
df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce')
df['Kamar Tidur'] = pd.to_numeric(df['Kamar Tidur'], errors='coerce')
df['Kamar Mandi'] = pd.to_numeric(df['Kamar Mandi'], errors='coerce')
df['Garasi'] = pd.to_numeric(df['Garasi'], errors='coerce').fillna(0)
df['Luas Tanah'] = df['Luas Tanah'].apply(parse_luas)
df['Luas Bangunan'] = df['Luas Bangunan'].apply(parse_luas)

if 'Kecamatan' in df.columns:
    df = df.rename(columns={'Kecamatan': 'Lokasi'})
df = df.drop(columns=['Page'], errors='ignore')
df = df.replace([np.inf, -np.inf], np.nan)

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
with conn.cursor() as cur:
    cur.execute('TRUNCATE TABLE "RawProperty" RESTART IDENTITY;')
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO "RawProperty"
            (harga, "kamarTidur", "kamarMandi", garasi, "luasTanah", "luasBangunan", lokasi)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            float(row['Harga']) if pd.notna(row['Harga']) else None,
            int(row['Kamar Tidur']) if pd.notna(row['Kamar Tidur']) else None,
            int(row['Kamar Mandi']) if pd.notna(row['Kamar Mandi']) else None,
            int(row['Garasi']) if pd.notna(row['Garasi']) else 0,
            float(row['Luas Tanah']) if pd.notna(row['Luas Tanah']) else None,
            float(row['Luas Bangunan']) if pd.notna(row['Luas Bangunan']) else None,
            str(row['Lokasi']),
        ))
    conn.commit()
print(f"Inserted {len(df)} raw records into RawProperty.")
conn.close()