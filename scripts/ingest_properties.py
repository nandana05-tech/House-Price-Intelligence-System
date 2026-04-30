"""
Embed property records into pgvector.
Run once: docker exec -it hpi_api python scripts/ingest_properties.py
Estimated time: ~20–40 min. Estimated cost: ~$0.32

Fix notes:
- Migrated from deprecated langchain_community.PGVector
  to langchain_postgres.PGVector
- Source table changed to CleanPropertyClustering WITHOUT the
  clusterId IS NOT NULL filter, so all 19k rows are ingested.
  clusterId / clusterLabel are kept as optional metadata.
- Column names aligned to exact Prisma schema field names.
"""
import os

import pandas as pd
import psycopg2
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

from rag.embedder import get_embeddings

DATABASE_URL = os.getenv("DATABASE_URL")

# ── 1. Load data ────────────────────────────────────────────────────────────
conn = psycopg2.connect(DATABASE_URL)

# Option A: Use CleanPropertyClustering (all rows, clusterId optional)
df = pd.read_sql(
    """
    SELECT
        id,
        harga,
        "kamarTidur",
        "kamarMandi",
        garasi,
        "luasTanah",
        "luasBangunan",
        lokasi,
        "clusterId",
        "clusterLabel"
    FROM "CleanPropertyClustering"
    """,
    conn,
)

# Option B (fallback): Use CleanPropertyRegression if clustering table is empty
if df.empty:
    print("CleanPropertyClustering is empty, falling back to CleanPropertyRegression...")
    df = pd.read_sql(
        """
        SELECT
            id,
            harga,
            "kamarTidur",
            "kamarMandi",
            garasi,
            "luasTanah",
            "luasBangunan",
            lokasi,
            NULL::int    AS "clusterId",
            NULL::text   AS "clusterLabel"
        FROM "CleanPropertyRegression"
        """,
        conn,
    )

conn.close()

print(f"Rows loaded: {len(df)}")
if df.empty:
    print(df)
    raise SystemExit("No data to ingest. Check your database.")

# ── 2. Build LangChain Documents ─────────────────────────────────────────────
docs = []
for _, row in df.iterrows():
    cluster_label = row["clusterLabel"] or "Unknown"
    cluster_id    = int(row["clusterId"]) if pd.notna(row["clusterId"]) else -1

    text = (
        f"Properti di {row['lokasi']}, Depok. "
        f"{int(row['kamarTidur'])} kamar tidur, {int(row['kamarMandi'])} kamar mandi, "
        f"garasi {int(row['garasi'])} mobil. "
        f"Luas tanah {float(row['luasTanah'])}m², luas bangunan {float(row['luasBangunan'])}m². "
        f"Harga: Rp {float(row['harga']):,.0f}. "
        f"Segmen: {cluster_label}."
    )
    docs.append(Document(
        page_content=text,
        metadata={
            "lokasi":         row["lokasi"],
            "kamar_tidur":    int(row["kamarTidur"]),
            "kamar_mandi":    int(row["kamarMandi"]),
            "garasi":         int(row["garasi"]),
            "luas_tanah":     float(row["luasTanah"]),
            "luas_bangunan":  float(row["luasBangunan"]),
            "harga":          int(row["harga"]),
            "segment_label":  cluster_label,
            "cluster_id":     cluster_id,
        },
    ))

print(f"Documents prepared: {len(docs)}")

# ── 3. Ingest into pgvector ──────────────────────────────────────────────────
# langchain_postgres uses JSONB metadata by default (no deprecation warning).
# Filter operators are now prefixed with $ (e.g. {"cluster_id": {"$eq": 2}}).
vectorstore = PGVector(
    embeddings=get_embeddings(),
    collection_name="property_embeddings",
    connection=DATABASE_URL,
    use_jsonb=True,
)

# Batch in chunks of 500 to avoid memory spikes on 19k+ records
BATCH_SIZE = 500
for i in range(0, len(docs), BATCH_SIZE):
    batch = docs[i : i + BATCH_SIZE]
    vectorstore.add_documents(batch)
    print(f"  Ingested {min(i + BATCH_SIZE, len(docs))}/{len(docs)} records...")

print(f"\nDone. Ingested {len(docs)} property records into 'property_embeddings'.")