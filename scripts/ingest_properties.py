"""
Embed all 40,200 property records into pgvector.
Run once: docker exec -it hpi_api python scripts/ingest_properties.py
Estimated time: ~20–40 min. Estimated cost: ~$0.32
"""
import os
import pandas as pd
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from rag.embedder import get_embeddings

df = pd.read_csv("data/data_with_clusters.csv")

docs = []
for _, row in df.iterrows():
    text = (
        f"Property in {row['Lokasi']}, Depok. "
        f"{int(row['Kamar Tidur'])} bedrooms, {int(row['Kamar Mandi'])} bathrooms, "
        f"garage {int(row['Garasi'])} cars. "
        f"Land {float(row['Luas Tanah'])}m², building {float(row['Luas Bangunan'])}m². "
        f"Price: Rp {float(row['Harga']):,.0f}. "
        f"Segment: {row.get('Cluster_Label', '')}."
    )
    docs.append(Document(
        page_content=text,
        metadata={
            "lokasi": row["Lokasi"],
            "kamar_tidur": int(row["Kamar Tidur"]),
            "kamar_mandi": int(row["Kamar Mandi"]),
            "garasi": int(row["Garasi"]),
            "luas_tanah": float(row["Luas Tanah"]),
            "luas_bangunan": float(row["Luas Bangunan"]),
            "harga": int(row["Harga"]),
            "segment_label": str(row.get("Cluster_Label", "")),
            "cluster_id": int(row.get("Cluster", -1)),
        }
    ))

PGVector.from_documents(
    documents=docs,
    embedding=get_embeddings(),
    collection_name="property_embeddings",
    connection_string=os.getenv("DATABASE_URL"),
)
print(f"Ingested {len(docs)} property records.")