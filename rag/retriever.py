"""
Two retrieval strategies:
1. Static: LangChain PGVector for knowledge_base (area profiles, rules, FAQs)
2. Dynamic: Raw SQL on langchain_pg_embedding.cmetadata for live price statistics
"""
from __future__ import annotations
import os
import re
import psycopg2
from pgvector.psycopg2 import register_vector
from langchain_postgres.vectorstores import PGVector   # ← fixed import
from rag.embedder import get_embeddings

CONNECTION_STRING = os.getenv("DATABASE_URL", "")
# langchain-postgres requires the psycopg3 driver prefix.
PGVECTOR_URL = re.sub(r"^postgresql(\+psycopg2)?", "postgresql+psycopg", CONNECTION_STRING)

_property_store: PGVector | None = None
_knowledge_store: PGVector | None = None
_pg_conn = None


# ── LangChain PGVector — comparable properties ────────────────────────────

def _get_property_store() -> PGVector:
    global _property_store
    if _property_store is None:
        _property_store = PGVector(
            embeddings=get_embeddings(),               # ← param renamed
            collection_name="property_embeddings",
            connection=PGVECTOR_URL,                   # ← fixed: connection_string → connection
            use_jsonb=True,
        )
    return _property_store


def _get_knowledge_store() -> PGVector:
    global _knowledge_store
    if _knowledge_store is None:
        _knowledge_store = PGVector(
            embeddings=get_embeddings(),               # ← param renamed
            collection_name="knowledge_base",
            connection=PGVECTOR_URL,                   # ← fixed: connection_string → connection
            use_jsonb=True,
        )
    return _knowledge_store


def get_comparable_properties(
    lokasi: str,
    kamar_tidur: int,
    kamar_mandi: int,
    garasi: int,
    luas_tanah: float,
    luas_bangunan: float,
    harga: float,
    top_k: int = 3,
) -> list[dict]:
    """Retrieve top-K similar properties via vector similarity search."""
    query_text = (
        f"Property in {lokasi}, {kamar_tidur}BR {kamar_mandi}BA, "
        f"LT {luas_tanah}m² LB {luas_bangunan}m², Rp {harga:,.0f}"
    )
    results = _get_property_store().similarity_search_with_score(query_text, k=top_k)
    return [
        {**doc.metadata, "similarity": round(1 - score, 4)}
        for doc, score in results
    ]


def get_knowledge(query: str, top_k: int = 2) -> list[dict]:
    """Retrieve relevant static knowledge documents."""
    results = _get_knowledge_store().similarity_search_with_score(query, k=top_k)
    return [
        {
            "title": doc.metadata.get("title", ""),
            "content": doc.page_content[:300],
            "doc_type": doc.metadata.get("doc_type", ""),
            "similarity": round(1 - score, 4),
        }
        for doc, score in results
    ]


# ── Live SQL — dynamic area statistics ───────────────────────────────────

def _get_pg_conn():
    global _pg_conn
    if _pg_conn is None or _pg_conn.closed:
        _pg_conn = psycopg2.connect(CONNECTION_STRING)
        register_vector(_pg_conn)
    return _pg_conn


def get_area_stats(lokasi: str) -> dict:
    """
    Compute live price statistics from cmetadata JSONB inside langchain_pg_embedding.
    Joins langchain_pg_collection to filter by collection name.
    """
    try:
        conn = _get_pg_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    AVG((e.cmetadata->>'harga')::BIGINT)::BIGINT  AS avg_harga,
                    MIN((e.cmetadata->>'harga')::BIGINT)          AS min_harga,
                    MAX((e.cmetadata->>'harga')::BIGINT)          AS max_harga,
                    COUNT(*)                                       AS jumlah_data,
                    MODE() WITHIN GROUP (
                        ORDER BY e.cmetadata->>'segment_label'
                    )                                              AS dominant_segment
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c
                    ON e.collection_id = c.uuid
                WHERE c.name = 'property_embeddings'
                  AND e.cmetadata->>'lokasi' = %s
            """, (lokasi,))
            row = cur.fetchone()
        if not row or row[3] == 0:
            return {}
        return {
            "avg_harga":        row[0],
            "min_harga":        row[1],
            "max_harga":        row[2],
            "jumlah_data":      row[3],
            "dominant_segment": row[4],
        }
    except Exception:
        return {}