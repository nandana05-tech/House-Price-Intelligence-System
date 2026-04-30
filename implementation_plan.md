# Data Pipeline Implementation Plan — House Price Intelligence System (HPI)

**Version**: 1.0
**Goal**: Replace file-based CSV dependency with PostgreSQL as single source of truth for all data pipelines (cleaning, retraining, RAG ingest, clustering).

---

## Overview

```
BEFORE:
Raw CSV (file) → Manual cleaning (notebook) → Cleaned CSV (file) → Model training

AFTER:
Raw CSV → RawProperty (PostgreSQL)
              ↓
    ┌─────────────────────────────────┐
    │  Cleaning Pipeline (per model)  │
    └─────────────────────────────────┘
              ↓
    ┌──────────────────────────────────────────────┐
    │  CleanPropertyRegression      (PostgreSQL)   │
    │  CleanPropertyClassification  (PostgreSQL)   │
    │  CleanPropertyClustering      (PostgreSQL)   │
    └──────────────────────────────────────────────┘
              ↓
    ┌─────────────────────────────────────────────────────┐
    │  Retrain Pipeline  → ML Models (.cbm, .pkl)         │
    │  RAG Ingest        → pgvector (property_embeddings) │
    │  setup_encoder.py  → target_encoder.pkl             │
    └─────────────────────────────────────────────────────┘
```

---

## Cleaning Strategy per Model

Each model uses a **different cleaning approach** mirroring the exact notebook preprocessing:

| Aspect | Regression | Classification | Clustering |
|--------|-----------|----------------|------------|
| Outlier method | **Drop** per lokasi Q0.05–Q0.95 + harga/m² filter | **IQR Clip** all numeric columns | **IQR Drop** Harga/LT/LB |
| Luas filter | LB>10, LT>10, LB≤600, LT≤1000, anomaly kavling | None | None |
| Spec filter | KT≤8, KM≤8, Garasi≤6 | None | None |
| Min price | Rp 200 juta | None | Rp 50 juta |
| Price range | Q0.01–Q0.99 global | IQR clip | Rp 50jt–10M |
| Lokasi encoding | `category_encoders.TargetEncoder` | One-hot (get_dummies) | Median log_Harga per kecamatan |
| Expected rows | ~18,000 | ~24,000 | ~19,000 |

---

## Phase 1 — Prisma Schema Update

### 1.1 Add New Tables to `prisma/schema.prisma`

Add these models alongside existing ones:

```prisma
// Raw data — stores parsed but uncleaned data
model RawProperty {
  id           Int      @id @default(autoincrement())
  harga        Float?
  kamarTidur   Int?
  kamarMandi   Int?
  garasi       Int      @default(0)
  luasTanah    Float?
  luasBangunan Float?
  lokasi       String
  createdAt    DateTime @default(now())
}

// Clean data for regression model
model CleanPropertyRegression {
  id           Int      @id @default(autoincrement())
  harga        Float
  kamarTidur   Int
  kamarMandi   Int
  garasi       Int
  luasTanah    Float
  luasBangunan Float
  lokasi       String
  createdAt    DateTime @default(now())
}

// Clean data for classification model
model CleanPropertyClassification {
  id           Int      @id @default(autoincrement())
  harga        Float
  kamarTidur   Int
  kamarMandi   Int
  garasi       Int
  luasTanah    Float
  luasBangunan Float
  lokasi       String
  kelasHarga   Int      // 0=Murah, 1=Menengah, 2=Atas, 3=Mewah
  createdAt    DateTime @default(now())
}

// Clean data for clustering model (includes engineered features)
model CleanPropertyClustering {
  id           Int      @id @default(autoincrement())
  harga        Float
  kamarTidur   Int
  kamarMandi   Int
  garasi       Int
  luasTanah    Float
  luasBangunan Float
  lokasi       String
  logHarga     Float
  logLT        Float
  logLB        Float
  hargaPerM2   Float
  logHargaM2   Float
  rasioLBLT    Float
  lokasi_enc   Float
  clusterId    Int?
  clusterLabel String?
  createdAt    DateTime @default(now())
}
```

### 1.2 Apply Schema

```bash
docker cp prisma/schema.prisma hpi_api:/app/prisma/schema.prisma
docker exec hpi_api sh -c "prisma db push --accept-data-loss"
```

---

## Phase 2 — New Scripts

### 2.1 Folder Structure Addition

```
hpi/
└── scripts/
    ├── ingest_raw_data.py                  ← NEW: Raw CSV → RawProperty
    ├── cleaning_pipeline_regression.py     ← NEW: RawProperty → CleanPropertyRegression
    ├── cleaning_pipeline_classification.py ← NEW: RawProperty → CleanPropertyClassification
    ├── cleaning_pipeline_clustering.py     ← NEW: RawProperty → CleanPropertyClustering
    ├── ingest_knowledge.py                 ← existing
    ├── ingest_properties.py                ← update: read from CleanPropertyClustering
    ├── setup_encoder.py                    ← update: read from CleanPropertyRegression
    └── predict_csv.py                      ← existing
```

### 2.2 `scripts/ingest_raw_data.py`

Parses raw CSV using the same `parse_harga()` and `parse_luas()` functions from all notebooks, then stores into `RawProperty`.

```python
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
```

### 2.3 `scripts/cleaning_pipeline_regression.py`

Mirrors **regresio_v1.2.ipynb** exactly:

```python
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
    "kamarTidur": "Kamar Tidur", "kamarMandi": "Kamar Mandi",
    "luasTanah": "Luas Tanah", "luasBangunan": "Luas Bangunan",
    "harga": "Harga", "lokasi": "Lokasi", "garasi": "Garasi",
})
print(f"Raw: {len(df)} rows")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['Harga','Kamar Tidur','Kamar Mandi',
                        'Garasi','Luas Tanah','Luas Bangunan'])
df_clean = df.copy()

df_clean = df_clean[df_clean['Luas Bangunan'] > 10]
df_clean = df_clean[df_clean['Luas Tanah'] > 10]
df_clean = df_clean[~((df_clean['Luas Tanah'] > 400) & (df_clean['Luas Bangunan'] < 150))]
df_clean = df_clean[df_clean['Luas Bangunan'] <= 600]
df_clean = df_clean[df_clean['Luas Tanah'] <= 1000]
df_clean = df_clean[df_clean['Kamar Tidur'] <= 8]
df_clean = df_clean[df_clean['Kamar Mandi'] <= 8]
df_clean = df_clean[df_clean['Garasi'] <= 6]
df_clean = df_clean[df_clean['Harga'] >= 200_000_000]

Q1, Q3 = df_clean['Harga'].quantile([0.01, 0.99])
df_clean = df_clean[(df_clean['Harga'] >= Q1) & (df_clean['Harga'] <= Q3)]

def filter_lokasi_outlier(group):
    if len(group) < 10: return group
    q1 = group['Harga'].quantile(0.05)
    q3 = group['Harga'].quantile(0.95)
    return group[(group['Harga'] >= q1) & (group['Harga'] <= q3)]

df_clean = df_clean.groupby('Lokasi', group_keys=False).apply(filter_lokasi_outlier)

df_clean['harga_per_m2'] = df_clean['Harga'] / df_clean['Luas Bangunan']
p5, p95 = df_clean['harga_per_m2'].quantile([0.05, 0.95])
df_clean = df_clean[(df_clean['harga_per_m2'] >= p5) & (df_clean['harga_per_m2'] <= p95)]
df_clean = df_clean.drop(columns=['harga_per_m2'])

print(f"Clean (regression): {len(df_clean)} rows")

with conn.cursor() as cur:
    cur.execute('TRUNCATE TABLE "CleanPropertyRegression" RESTART IDENTITY;')
    for _, row in df_clean.iterrows():
        cur.execute("""
            INSERT INTO "CleanPropertyRegression"
            (harga, "kamarTidur", "kamarMandi", garasi, "luasTanah", "luasBangunan", lokasi)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (float(row['Harga']), int(row['Kamar Tidur']), int(row['Kamar Mandi']),
               int(row['Garasi']), float(row['Luas Tanah']), float(row['Luas Bangunan']),
               str(row['Lokasi'])))
    conn.commit()
print(f"Saved {len(df_clean)} rows to CleanPropertyRegression.")
conn.close()
```

### 2.4 `scripts/cleaning_pipeline_classification.py`

Mirrors **clasifikasi_v1.0.ipynb** exactly:

```python
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
    "kamarTidur": "Kamar Tidur", "kamarMandi": "Kamar Mandi",
    "luasTanah": "Luas Tanah", "luasBangunan": "Luas Bangunan",
    "harga": "Harga", "lokasi": "Lokasi", "garasi": "Garasi",
})
print(f"Raw: {len(df)} rows")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['Harga','Kamar Tidur','Kamar Mandi',
                        'Garasi','Luas Tanah','Luas Bangunan'])

# IQR Clip — bukan drop, sesuai notebook klasifikasi
num_cols = ['Harga','Kamar Tidur','Kamar Mandi','Garasi','Luas Tanah','Luas Bangunan']
for col in num_cols:
    df[col] = df[col].astype(float)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

def label_harga(harga):
    if harga <= 745_000_000: return 0
    elif harga <= 1_300_000_000: return 1
    elif harga <= 2_645_000_000: return 2
    else: return 3

df['kelas_harga'] = df['Harga'].apply(label_harga)
print(f"Clean (classification): {len(df)} rows")

with conn.cursor() as cur:
    cur.execute('TRUNCATE TABLE "CleanPropertyClassification" RESTART IDENTITY;')
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO "CleanPropertyClassification"
            (harga, "kamarTidur", "kamarMandi", garasi,
             "luasTanah", "luasBangunan", lokasi, "kelasHarga")
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (float(row['Harga']), int(row['Kamar Tidur']), int(row['Kamar Mandi']),
               int(row['Garasi']), float(row['Luas Tanah']), float(row['Luas Bangunan']),
               str(row['Lokasi']), int(row['kelas_harga'])))
    conn.commit()
print(f"Saved {len(df)} rows to CleanPropertyClassification.")
conn.close()
```

### 2.5 `scripts/cleaning_pipeline_clustering.py`

Mirrors **clustering_v1_2_2.ipynb** exactly:

```python
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
import os, json, pickle
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path

MODELS_DIR = Path("/app/models")
conn = psycopg2.connect(os.getenv("DATABASE_URL"))

df = pd.read_sql('SELECT * FROM "RawProperty"', conn)
df = df.rename(columns={
    "kamarTidur": "Kamar Tidur", "kamarMandi": "Kamar Mandi",
    "luasTanah": "Luas Tanah", "luasBangunan": "Luas Bangunan",
    "harga": "Harga", "lokasi": "Lokasi", "garasi": "Garasi",
})
print(f"Raw: {len(df)} rows")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['Harga','Kamar Tidur','Kamar Mandi',
                        'Garasi','Luas Tanah','Luas Bangunan'])
df_clean = df.copy()

# Filter harga wajar
df_clean = df_clean[
    (df_clean['Harga'] >= 50_000_000) &
    (df_clean['Harga'] <= 10_000_000_000)
]

# IQR Drop — sesuai notebook clustering
for col in ['Harga','Luas Tanah','Luas Bangunan']:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    before = len(df_clean)
    df_clean = df_clean[
        (df_clean[col] >= Q1 - 1.5 * IQR) &
        (df_clean[col] <= Q3 + 1.5 * IQR)
    ]
    print(f"  {col}: removed {before - len(df_clean)} rows")

# Feature engineering
df_clean = df_clean.copy()
df_clean['log_Harga']    = np.log1p(df_clean['Harga'])
df_clean['log_LT']       = np.log1p(df_clean['Luas Tanah'])
df_clean['log_LB']       = np.log1p(df_clean['Luas Bangunan'])
df_clean['Harga_per_m2'] = df_clean['Harga'] / df_clean['Luas Bangunan'].replace(0, np.nan)
df_clean['log_Harga_m2'] = np.log1p(df_clean['Harga_per_m2'])
df_clean['rasio_LB_LT']  = df_clean['Luas Bangunan'] / df_clean['Luas Tanah'].replace(0, np.nan)

# Target encoding — median log_Harga per lokasi
lokasi_median = df_clean.groupby('Lokasi')['log_Harga'].median()
df_clean['Lokasi_enc'] = df_clean['Lokasi'].map(lokasi_median)

# Drop NaN from feature engineering
features = ['log_Harga','log_LT','log_LB','log_Harga_m2',
            'rasio_LB_LT','Kamar Tidur','Kamar Mandi','Lokasi_enc']
X = df_clean[features].replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.loc[X.dropna().index].copy()
print(f"Clean (clustering): {len(df_clean)} rows")

# Save clustering encoder
lokasi_enc_map = df_clean.groupby('Lokasi')['Lokasi_enc'].mean().to_dict()
fallback = float(df_clean['Lokasi_enc'].mean())
with open(MODELS_DIR / "clustering_encoder.pkl", "wb") as f:
    pickle.dump({"map": lokasi_enc_map, "fallback": fallback}, f)
print(f"Saved clustering_encoder.pkl ({len(lokasi_enc_map)} locations)")

with conn.cursor() as cur:
    cur.execute('TRUNCATE TABLE "CleanPropertyClustering" RESTART IDENTITY;')
    for _, row in df_clean.iterrows():
        cur.execute("""
            INSERT INTO "CleanPropertyClustering"
            (harga, "kamarTidur", "kamarMandi", garasi,
             "luasTanah", "luasBangunan", lokasi,
             "logHarga", "logLT", "logLB",
             "hargaPerM2", "logHargaM2", "rasioLBLT", "lokasi_enc")
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            float(row['Harga']), int(row['Kamar Tidur']),
            int(row['Kamar Mandi']), int(row['Garasi']),
            float(row['Luas Tanah']), float(row['Luas Bangunan']),
            str(row['Lokasi']), float(row['log_Harga']),
            float(row['log_LT']), float(row['log_LB']),
            float(row['Harga_per_m2']), float(row['log_Harga_m2']),
            float(row['rasio_LB_LT']), float(row['Lokasi_enc']),
        ))
    conn.commit()
print(f"Saved {len(df_clean)} rows to CleanPropertyClustering.")
conn.close()
```

---

## Phase 3 — Update Existing Scripts

### 3.1 Update `scripts/setup_encoder.py`

Read from `CleanPropertyRegression` instead of CSV:

```python
# SEBELUM
df = pd.read_csv(DATA_PATH)

# SESUDAH
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
df = pd.read_sql('SELECT * FROM "CleanPropertyRegression"', conn)
df = df.rename(columns={"harga": "Harga", "lokasi": "Lokasi"})
conn.close()
```

### 3.2 Update `scripts/ingest_properties.py` (RAG)

Read from `CleanPropertyClustering` instead of CSV:

```python
# SEBELUM
df = pd.read_csv("data/data_with_clusters.csv")

# SESUDAH
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
df = pd.read_sql("""
    SELECT harga, "kamarTidur" AS "Kamar Tidur",
           "kamarMandi" AS "Kamar Mandi", garasi AS "Garasi",
           "luasTanah" AS "Luas Tanah", "luasBangunan" AS "Luas Bangunan",
           lokasi AS "Lokasi", "clusterLabel" AS "Cluster_Label",
           "clusterId" AS "Cluster"
    FROM "CleanPropertyClustering"
    WHERE "clusterId" IS NOT NULL
""", conn)
conn.close()
```

### 3.3 Update `pipelines/retrain_pipeline.py`

Read from database instead of CSV:

```python
# SEBELUM
df_base = pd.read_csv(DATA_PATH)

# SESUDAH
import psycopg2
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
df_base = pd.read_sql('SELECT * FROM "CleanPropertyRegression"', conn)
df_base = df_base.rename(columns={
    "kamarTidur": "Kamar Tidur", "kamarMandi": "Kamar Mandi",
    "luasTanah": "Luas Tanah", "luasBangunan": "Luas Bangunan",
    "harga": "Harga", "lokasi": "Lokasi", "garasi": "Garasi",
})
conn.close()
print(f"[Retrain] Loaded {len(df_base)} rows from CleanPropertyRegression")
```

Also remove the manual outlier filtering from retrain pipeline since `CleanPropertyRegression` is already clean.

---

## Phase 4 — Execution Order

```bash
# Step 1 — Update Prisma schema
docker cp prisma/schema.prisma hpi_api:/app/prisma/schema.prisma
docker exec hpi_api sh -c "prisma db push --accept-data-loss"

# Step 2 — Copy scripts to container
docker cp scripts/ingest_raw_data.py hpi_api:/app/scripts/ingest_raw_data.py
docker cp scripts/cleaning_pipeline_regression.py hpi_api:/app/scripts/cleaning_pipeline_regression.py
docker cp scripts/cleaning_pipeline_classification.py hpi_api:/app/scripts/cleaning_pipeline_classification.py
docker cp scripts/cleaning_pipeline_clustering.py hpi_api:/app/scripts/cleaning_pipeline_clustering.py

# Step 3 — Copy raw data to container
docker cp data/Raw-Final-Rumah.csv hpi_api:/app/data/Raw-Final-Rumah.csv

# Step 4 — Ingest raw data (parse only, no cleaning)
docker exec -e PYTHONPATH=/app hpi_api python scripts/ingest_raw_data.py

# Step 5 — Run cleaning pipelines (all three, independent)
docker exec -e PYTHONPATH=/app hpi_api python scripts/cleaning_pipeline_regression.py
docker exec -e PYTHONPATH=/app hpi_api python scripts/cleaning_pipeline_classification.py
docker exec -e PYTHONPATH=/app hpi_api python scripts/cleaning_pipeline_clustering.py

# Step 6 — Rebuild encoder from CleanPropertyRegression
docker exec -e PYTHONPATH=/app hpi_api python scripts/setup_encoder.py

# Step 7 — Re-ingest RAG properties from CleanPropertyClustering
docker exec -e PYTHONPATH=/app hpi_api python scripts/ingest_properties.py

# Step 8 — Restart API to reload models
docker-compose restart api
```

---

## Phase 5 — Verification

```bash
# Check row counts in each table
docker exec hpi_postgres psql -U hpi -d house_price_intel -c "
SELECT
  (SELECT COUNT(*) FROM \"RawProperty\") AS raw,
  (SELECT COUNT(*) FROM \"CleanPropertyRegression\") AS regression,
  (SELECT COUNT(*) FROM \"CleanPropertyClassification\") AS classification,
  (SELECT COUNT(*) FROM \"CleanPropertyClustering\") AS clustering;
"
```

Expected output:

| raw | regression | classification | clustering |
|-----|-----------|----------------|------------|
| ~40,200 | ~18,000 | ~24,000 | ~19,000 |

---

## Updated Project Structure

```
hpi/
├── data/
│   └── Raw-Final-Rumah.csv          ← Raw CSV (source of truth for initial load)
│
└── scripts/
    ├── ingest_raw_data.py            ← NEW: CSV → RawProperty
    ├── cleaning_pipeline_regression.py   ← NEW: RawProperty → CleanPropertyRegression
    ├── cleaning_pipeline_classification.py ← NEW: RawProperty → CleanPropertyClassification
    ├── cleaning_pipeline_clustering.py   ← NEW: RawProperty → CleanPropertyClustering
    ├── setup_encoder.py              ← UPDATED: reads CleanPropertyRegression
    ├── ingest_properties.py          ← UPDATED: reads CleanPropertyClustering
    ├── ingest_knowledge.py           ← unchanged
    └── predict_csv.py                ← unchanged
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Separate clean tables | One per model | Each model has different cleaning — cannot share |
| Raw table | Keep all 40,200 rows | Audit trail, re-run cleaning anytime |
| CSV file | Keep in `data/` | Needed for initial `ingest_raw_data.py` run |
| Retrain source | `CleanPropertyRegression` | Most conservative cleaning = best model quality |
| RAG source | `CleanPropertyClustering` | Contains cluster labels needed for comparable search |
| Encoder source | `CleanPropertyRegression` | TargetEncoder trained on same data as regression model |

---

## Implementation Timeline

| Phase | Task | Estimated Effort |
|-------|------|-----------------|
| Phase 1 | Update Prisma schema + push | 0.5 day |
| Phase 2 | Write & test 4 new scripts | 1 day |
| Phase 3 | Update 3 existing scripts | 0.5 day |
| Phase 4 | Run full pipeline end-to-end | 0.5 day |
| Phase 5 | Verify row counts + test API | 0.5 day |
| **Total** | | **~3 days** |