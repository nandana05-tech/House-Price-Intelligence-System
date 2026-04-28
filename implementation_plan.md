# RAG Implementation Plan — House Price Intelligence System (HPI)

**Version**: 3.0 (Updated)
**Goal**: Augment the existing ML prediction pipeline with a Retrieval-Augmented Generation (RAG) layer to enable Explainable Prediction, Comparable Property Analysis, Context-Aware Intelligence, Natural Language Consulting, and Reduced LLM Hallucination.

**Key Updates from v2.0**:
- Knowledge documents contain **static information only** (location, infrastructure, characteristics)
- **Dynamic data** (price stats, dominant segment, appreciation) computed live from `property_embeddings` via `get_area_stats()` — always up-to-date without manual re-ingestion
- LangChain used selectively (embedder + retriever + ingest only)
- Token optimization + Redis caching applied by default

---

## Overview

```
BEFORE:
User → Chat Endpoint → GPT (LLM Router) → MCP Tools → ML Models → Raw Numbers

AFTER:
User → Chat Endpoint → GPT (LLM Router) → MCP Tools → ML Models
                                    ↓
                         [Conditional RAG Trigger]
                                    ↓
                 ┌──────────────────────────────────────┐
                 │  Static: LangChain PGVector           │
                 │  (knowledge_base — area, infra, FAQ)  │
                 │                                       │
                 │  Dynamic: Live SQL query              │
                 │  (property_embeddings — price stats)  │
                 └──────────────────────────────────────┘
                                    ↓
                   context_builder.py (manual)
                                    ↓
                   Enriched Context → GPT Final Answer
                         (cached in Redis 1hr)
```

---

## Static vs Dynamic Knowledge Design

| Data Type | Source | Update Method |
|-----------|--------|--------------|
| Location description | `knowledge/area_profiles/*.md` | Edit `.md` → re-run `ingest_knowledge.py` |
| Infrastructure (malls, schools, toll) | `knowledge/area_profiles/*.md` | Edit `.md` → re-run `ingest_knowledge.py` |
| Area characteristics | `knowledge/area_profiles/*.md` | Edit `.md` → re-run `ingest_knowledge.py` |
| Pricing rules & methodology | `knowledge/market_rules/*.md` | Edit `.md` → re-run `ingest_knowledge.py` |
| FAQs & model explanation | `knowledge/faqs/*.md` | Edit `.md` → re-run `ingest_knowledge.py` |
| **Average price per area** | `property_embeddings` (live SQL) | **Automatic** — always current |
| **Price range per area** | `property_embeddings` (live SQL) | **Automatic** — always current |
| **Dominant segment per area** | `property_embeddings` (live SQL) | **Automatic** — always current |
| **Number of comparable properties** | `property_embeddings` (live SQL) | **Automatic** — always current |

---

## Goals & RAG Use Cases

| Goal | RAG Role |
|------|----------|
| **Explainable Prediction** | Retrieve similar past transactions to justify the predicted price |
| **Comparable Property Analysis** | Find top-K most similar properties from the dataset |
| **Context-Aware Intelligence** | Inject area/market context before LLM generates a response |
| **Natural Language Property Consultant** | Answer questions like "why is Cinere more expensive than Beji?" |
| **Knowledge Update Without Retraining** | Add new market knowledge to vector DB without touching ML models |
| **Trust & Transparency** | Show sources/evidence alongside every prediction |
| **Hybrid Intelligence (ML + Knowledge)** | Combine CatBoost output with retrieved domain knowledge |
| **Decision Support System** | Provide investment context, risk indicators, and market trends |
| **Reduced Hallucination** | Ground LLM answers in retrieved facts, not parametric memory |

---

## LangChain Usage Decision

| Component | Use LangChain? | Reason |
|-----------|---------------|--------|
| `rag/embedder.py` | ✅ `OpenAIEmbeddings` | Removes boilerplate, handles batching |
| `rag/retriever.py` | ✅ `PGVector` | Abstracts raw psycopg2 + pgvector SQL |
| `scripts/ingest_knowledge.py` | ✅ `TextLoader` + `MarkdownTextSplitter` | Smart chunking for markdown docs |
| `rag/context_builder.py` | ❌ Manual | Custom HPI logic, no LangChain equivalent |
| `api/chat_endpoint.py` | ❌ Manual | Native OpenAI function calling already optimal |
| `pipelines/` | ❌ Not relevant | Pure ML retraining logic |
| `kafka/` | ❌ Not relevant | Pure Kafka consumer logic |

---

## Phase 1 — Infrastructure Setup

### 1.1 Vector Database: pgvector (extends existing PostgreSQL)

No new Docker service needed — pgvector is a PostgreSQL extension.

```sql
-- Run once inside hpi_postgres container
CREATE EXTENSION IF NOT EXISTS vector;

-- Comparable properties table (dynamic data source)
CREATE TABLE property_embeddings (
    id            SERIAL PRIMARY KEY,
    property_id   TEXT,
    lokasi        TEXT,
    kamar_tidur   INT,
    kamar_mandi   INT,
    garasi        INT,
    luas_tanah    FLOAT,
    luas_bangunan FLOAT,
    harga         BIGINT,
    cluster_id    INT,
    segment_label TEXT,
    metadata      JSONB,
    embedding     vector(1536)
);

-- Knowledge base table (static data source)
CREATE TABLE knowledge_base (
    id         SERIAL PRIMARY KEY,
    doc_type   TEXT,        -- 'area_profile' | 'market_rule' | 'faq'
    title      TEXT,
    content    TEXT,
    source     TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    embedding  vector(1536)
);

-- HNSW index for fast ANN search
CREATE INDEX ON property_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON knowledge_base USING hnsw (embedding vector_cosine_ops);
```

### 1.2 Update Prisma Schema

```prisma
// prisma/schema.prisma — add RAG tables
model PropertyEmbedding {
  id            Int      @id @default(autoincrement())
  propertyId    String?
  lokasi        String
  kamarTidur    Int
  kamarMandi    Int
  garasi        Int
  luasTanah     Float
  luasBangunan  Float
  harga         BigInt
  clusterId     Int?
  segmentLabel  String?
  metadata      Json?
  createdAt     DateTime @default(now())
  // Note: embedding vector stored natively via pgvector, managed outside Prisma
}

model KnowledgeBase {
  id        Int      @id @default(autoincrement())
  docType   String   // area_profile | market_rule | faq
  title     String
  content   String
  source    String?
  createdAt DateTime @default(now())
}
```

### 1.3 Install New Dependencies

```toml
# pyproject.toml — add these dependencies
[project.dependencies]
# existing deps ...
langchain-openai = ">=0.1.0"
langchain-community = ">=0.2.0"
langchain = ">=0.2.0"          # core only — for MarkdownTextSplitter
pgvector = ">=0.3.0"
tiktoken = ">=0.7.0"
```

---

## Phase 2 — Knowledge Base Construction (Static Only)

### 2.1 Folder Structure Addition

```
hpi/
├── rag/                              ← NEW
│   ├── __init__.py
│   ├── embedder.py                   ← LangChain OpenAIEmbeddings wrapper
│   ├── retriever.py                  ← LangChain PGVector + live SQL stats
│   ├── context_builder.py            ← Manual: assembles static + dynamic context
│   └── cache.py                      ← Redis cache for RAG results
│
├── knowledge/                        ← NEW — STATIC knowledge only
│   ├── area_profiles/                ← Location, infra, characteristics (no prices)
│   │   ├── cinere.md
│   │   ├── beji.md
│   │   ├── sawangan.md
│   │   └── ...                       ← one file per kecamatan in Depok
│   ├── market_rules/
│   │   ├── pricing_factors.md
│   │   ├── segment_definitions.md
│   │   └── investment_guidelines.md
│   └── faqs/
│       ├── how_price_is_calculated.md
│       └── model_methodology.md
│
└── scripts/
    ├── ingest_knowledge.py           ← NEW: load knowledge/ → pgvector
    └── ingest_properties.py          ← NEW: embed dataset → pgvector
```

### 2.2 Knowledge Document Format (Static Only)

```markdown
<!-- knowledge/area_profiles/cinere.md -->
# Cinere

## Location
- Sub-district (kecamatan) in South Depok, West Java
- Borders: Limo (north), Sawangan (east), Tangerang Selatan (west)

## Infrastructure
- Toll access: Cinere–Jagorawi (CiJago) toll road
- Commercial: Cinere Mall, Bellevue Mall
- Education: multiple international schools nearby

## Area Characteristics
- Predominantly residential, popular among Jakarta professionals and expatriates
- Premium residential estates: Jl. Cinere Raya, Perumahan Cinere Indah
- Limited new land supply due to established residential density
```

> **Note**: No price data, segment, or appreciation rates in knowledge documents.
> All dynamic price statistics are computed live from `property_embeddings`.

---

## Phase 3 — RAG Layer Implementation

### 3.1 `rag/embedder.py` — LangChain OpenAIEmbeddings

```python
"""
Embedding generation using LangChain OpenAIEmbeddings.
"""
from __future__ import annotations
import os
from langchain_openai import OpenAIEmbeddings

_embeddings: OpenAIEmbeddings | None = None


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _embeddings


def embed_text(text: str) -> list[float]:
    return get_embeddings().embed_query(text)
```

### 3.2 `rag/retriever.py` — LangChain PGVector + Live SQL Stats

```python
"""
Two retrieval strategies:
1. Static: LangChain PGVector for knowledge_base (area profiles, rules, FAQs)
2. Dynamic: Raw SQL on property_embeddings for live price statistics
"""
from __future__ import annotations
import os
import psycopg2
from pgvector.psycopg2 import register_vector
from langchain_community.vectorstores import PGVector
from rag.embedder import get_embeddings

CONNECTION_STRING = os.getenv("DATABASE_URL", "")

_property_store: PGVector | None = None
_knowledge_store: PGVector | None = None
_pg_conn = None


# ── LangChain PGVector — comparable properties ────────────────────────────

def _get_property_store() -> PGVector:
    global _property_store
    if _property_store is None:
        _property_store = PGVector(
            connection_string=CONNECTION_STRING,
            embedding_function=get_embeddings(),
            collection_name="property_embeddings",
        )
    return _property_store


def _get_knowledge_store() -> PGVector:
    global _knowledge_store
    if _knowledge_store is None:
        _knowledge_store = PGVector(
            connection_string=CONNECTION_STRING,
            embedding_function=get_embeddings(),
            collection_name="knowledge_base",
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
            "content": doc.page_content[:300],   # truncate — token efficient
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
    Compute live price statistics directly from property_embeddings.
    Always up-to-date — no manual knowledge update needed.
    Returns empty dict if lokasi not found.
    """
    try:
        conn = _get_pg_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    AVG(harga)::BIGINT         AS avg_harga,
                    MIN(harga)                 AS min_harga,
                    MAX(harga)                 AS max_harga,
                    COUNT(*)                   AS jumlah_data,
                    MODE() WITHIN GROUP (ORDER BY segment_label) AS dominant_segment
                FROM property_embeddings
                WHERE lokasi = %s
            """, (lokasi,))
            row = cur.fetchone()
        if not row or row[3] == 0:
            return {}
        return {
            "avg_harga": row[0],
            "min_harga": row[1],
            "max_harga": row[2],
            "jumlah_data": row[3],
            "dominant_segment": row[4],
        }
    except Exception:
        return {}
```

### 3.3 `rag/cache.py` — Redis Cache

```python
"""
Cache RAG context in Redis to avoid redundant embedding + SQL calls.
TTL: 1 hour per query hash.
"""
from __future__ import annotations
import hashlib
import json
import os
import redis

_client: redis.Redis | None = None
RAG_CACHE_TTL = 3600  # 1 hour


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    return _client


def _cache_key(query: str, prediction: dict | None) -> str:
    pred_hash = hashlib.md5(
        json.dumps(prediction or {}, sort_keys=True).encode()
    ).hexdigest()
    return f"rag:{hashlib.md5(f'{query}:{pred_hash}'.encode()).hexdigest()}"


def get_cached_context(query: str, prediction: dict | None) -> str | None:
    try:
        cached = _get_client().get(_cache_key(query, prediction))
        return cached.decode() if cached else None
    except Exception:
        return None


def set_cached_context(query: str, prediction: dict | None, context: str) -> None:
    try:
        _get_client().setex(_cache_key(query, prediction), RAG_CACHE_TTL, context)
    except Exception:
        pass  # cache failure is non-critical
```

### 3.4 `rag/context_builder.py` — Static + Dynamic Assembly

```python
"""
Builds enriched prompt context combining:
- ML prediction output
- Static knowledge (area profiles, rules) via pgvector
- Dynamic price statistics (live SQL from property_embeddings)
- Comparable properties (vector similarity search)
"""
from __future__ import annotations
from rag.retriever import get_comparable_properties, get_knowledge, get_area_stats
from rag.cache import get_cached_context, set_cached_context

RAG_KEYWORDS = [
    "harga", "price", "estimasi", "lokasi", "segmen", "cluster",
    "rumah", "properti", "property", "beli", "invest", "mahal", "murah",
]


def should_use_rag(query: str) -> bool:
    """Only trigger RAG for property-related queries — saves tokens."""
    return any(kw in query.lower() for kw in RAG_KEYWORDS)


def build_prediction_context(
    prediction: dict,
    comparables: list[dict],
    knowledge: list[dict],
    area_stats: dict,
) -> str:
    lines = []

    # ── ML Prediction Result ──────────────────────────────────────────
    lines.append("## ML Prediction Result")
    lines.append(f"- Estimated Price: {prediction.get('harga_estimasi_format', 'N/A')}")
    lines.append(f"- Model Used: {prediction.get('model_digunakan', 'N/A')}")
    lines.append(f"- MAPE: {prediction.get('mape_persen', 'N/A')}%")
    lines.append(f"- Segment: {prediction.get('kelas_label', 'N/A')}")
    lines.append(f"- Cluster: {prediction.get('cluster_label', 'N/A')}")

    # ── Dynamic: Live Area Statistics ─────────────────────────────────
    lokasi = prediction.get("lokasi", "")
    if area_stats:
        lines.append(f"\n## Live Market Statistics — {lokasi}")
        lines.append(f"- Average price: Rp {area_stats['avg_harga']:,}")
        lines.append(
            f"- Price range: Rp {area_stats['min_harga']:,} – Rp {area_stats['max_harga']:,}"
        )
        lines.append(f"- Dominant segment: {area_stats['dominant_segment']}")
        lines.append(f"- Data points: {area_stats['jumlah_data']} properties")

    # ── Vector Search: Comparable Properties ──────────────────────────
    if comparables:
        lines.append("\n## Comparable Properties")
        for i, p in enumerate(comparables, 1):
            lines.append(
                f"{i}. {p.get('lokasi')} | {p.get('kamar_tidur')}BR "
                f"| LT {p.get('luas_tanah')}m² | Rp {p.get('harga'):,} "
                f"| {p.get('segment_label')} | sim: {p.get('similarity')}"
            )

    # ── Static: Knowledge Documents ───────────────────────────────────
    if knowledge:
        lines.append("\n## Area & Market Knowledge")
        for doc in knowledge:
            lines.append(f"### {doc['title']}")
            lines.append(doc["content"])

    return "\n".join(lines)


def build_rag_context(query: str, prediction: dict | None = None) -> str:
    """
    Main entry point.
    Returns cached context if available, otherwise retrieves fresh context.
    Returns empty string if query is not property-related.
    """
    if not should_use_rag(query):
        return ""

    # Check Redis cache first
    cached = get_cached_context(query, prediction)
    if cached:
        return cached

    # Static: knowledge documents via pgvector
    knowledge = get_knowledge(query, top_k=2)

    comparables = []
    area_stats = {}

    if prediction:
        lokasi = prediction.get("lokasi", "")

        # Dynamic: live price stats from property_embeddings
        area_stats = get_area_stats(lokasi)

        # Vector search: comparable properties
        comparables = get_comparable_properties(
            lokasi=lokasi,
            kamar_tidur=prediction.get("kamar_tidur", 0),
            kamar_mandi=prediction.get("kamar_mandi", 0),
            garasi=prediction.get("garasi", 0),
            luas_tanah=prediction.get("luas_tanah", 0),
            luas_bangunan=prediction.get("luas_bangunan", 0),
            harga=prediction.get("harga_estimasi", 0),
            top_k=3,
        )

    context = build_prediction_context(prediction or {}, comparables, knowledge, area_stats)

    # Cache result
    set_cached_context(query, prediction, context)
    return context
```

---

## Phase 4 — Integration with Chat Endpoint

Update `api/chat_endpoint.py` — inject RAG context after ML tool calls:

```python
# api/chat_endpoint.py — add these imports
from rag.context_builder import build_rag_context

# Inside chat_with_agent(), replace the second GPT call section:

        # ── RAG INJECTION ─────────────────────────────────────────────
        rag_context = build_rag_context(
            query=body.message,
            prediction=prediction_result,
        )

        if rag_context:
            messages.append({
                "role": "system",
                "content": (
                    "Use the following retrieved context to enrich your answer. "
                    "When relevant, cite comparable properties and area knowledge. "
                    "Be transparent about model confidence and limitations.\n\n"
                    + rag_context
                ),
            })
        # ── END RAG INJECTION ─────────────────────────────────────────

        second_response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        final_reply = second_response.choices[0].message.content
```

---

## Phase 5 — New API Endpoint: Comparable Properties

Add to `api/predict_endpoint.py`:

```python
from rag.retriever import get_comparable_properties

class ComparableRequest(BaseModel):
    kamar_tidur: int = Field(..., ge=1, le=10)
    kamar_mandi: int = Field(..., ge=1, le=10)
    garasi: int = Field(..., ge=0, le=5)
    luas_tanah: float = Field(..., gt=0)
    luas_bangunan: float = Field(..., gt=0)
    lokasi: str = Field(..., min_length=2)
    harga: float = Field(..., gt=0)
    top_k: int = Field(5, ge=1, le=20)

@router.post("/comparable_properties")
async def api_comparable_properties(body: ComparableRequest):
    """Find top-K most similar properties using vector similarity search."""
    try:
        results = get_comparable_properties(**body.model_dump())
        return {"comparables": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Phase 6 — Ingest Scripts

### 6.1 `scripts/ingest_properties.py`

```python
"""
Embed all 40,200 property records into pgvector.
Run once: docker exec -it hpi_api python scripts/ingest_properties.py
Estimated time: ~20–40 min. Estimated cost: ~$0.32
"""
import os
import pandas as pd
from langchain_community.vectorstores import PGVector
from langchain.schema import Document
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
```

### 6.2 `scripts/ingest_knowledge.py`

```python
"""
Embed static markdown knowledge documents into pgvector.
Run once — and re-run only when knowledge/ files are edited:
docker exec -it hpi_api python scripts/ingest_knowledge.py
"""
import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import PGVector
from rag.embedder import get_embeddings

KNOWLEDGE_DIR = Path("knowledge")
splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)

doc_type_map = {
    "area_profiles": "area_profile",
    "market_rules": "market_rule",
    "faqs": "faq",
}

all_docs = []
for folder, doc_type in doc_type_map.items():
    for md_file in (KNOWLEDGE_DIR / folder).glob("*.md"):
        loader = TextLoader(str(md_file), encoding="utf-8")
        raw_docs = loader.load()
        chunks = splitter.split_documents(raw_docs)
        for chunk in chunks:
            chunk.metadata.update({
                "doc_type": doc_type,
                "title": md_file.stem.replace("_", " ").title(),
                "source": str(md_file),
            })
        all_docs.extend(chunks)

PGVector.from_documents(
    documents=all_docs,
    embedding=get_embeddings(),
    collection_name="knowledge_base",
    connection_string=os.getenv("DATABASE_URL"),
)
print(f"Ingested {len(all_docs)} knowledge chunks.")
```

---

## Phase 7 — Docker & Infrastructure Updates

### 7.1 Update `docker-compose.yml`

```yaml
api:
  environment:
    # ... existing vars ...
    EMBEDDING_MODEL: text-embedding-3-small
    RAG_ENABLED: "true"
    RAG_TOP_K_PROPERTIES: "3"
    RAG_TOP_K_KNOWLEDGE: "2"
```

### 7.2 Enable pgvector & Run Ingestion

```bash
# Step 1 — Enable pgvector extension
docker exec -it hpi_postgres psql -U hpi -d house_price_intel \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Step 2 — Apply Prisma schema changes
docker exec -it hpi_api sh -c "prisma db push --accept-data-loss"

# Step 3 — Ingest static knowledge documents (fast, ~30 seconds)
docker exec -it hpi_api python scripts/ingest_knowledge.py

# Step 4 — Ingest property dataset (slow, ~20–40 min)
docker exec -it hpi_api python scripts/ingest_properties.py
```

---

## Updated Project Structure

```
hpi/
├── ...existing files...
│
├── rag/                              ← NEW
│   ├── __init__.py
│   ├── embedder.py                   ← LangChain OpenAIEmbeddings (singleton)
│   ├── retriever.py                  ← LangChain PGVector + live SQL get_area_stats()
│   ├── context_builder.py            ← Manual: static + dynamic context assembly
│   └── cache.py                      ← Redis cache (1hr TTL)
│
├── knowledge/                        ← NEW — STATIC documents only
│   ├── area_profiles/                ← Location, infrastructure, characteristics
│   │   ├── cinere.md
│   │   ├── beji.md
│   │   ├── sawangan.md
│   │   └── ...
│   ├── market_rules/
│   │   ├── pricing_factors.md
│   │   ├── segment_definitions.md
│   │   └── investment_guidelines.md
│   └── faqs/
│       ├── how_price_is_calculated.md
│       └── model_methodology.md
│
└── scripts/
    ├── ingest_properties.py          ← NEW: embed 40K dataset → pgvector
    ├── ingest_knowledge.py           ← NEW: embed static knowledge → pgvector
    └── ...existing scripts...
```

---

## Updated API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict_price` | POST | Price estimation (unchanged) |
| `/api/v1/classify_segment` | POST | Segment classification (unchanged) |
| `/api/v1/cluster_property` | POST | Clustering (unchanged) |
| `/api/v1/comparable_properties` | POST | **NEW** — Top-K similar properties via pgvector |
| `/api/v1/feedback` | POST | Feedback (unchanged) |
| `/api/v1/chat` | POST | Chat — **now RAG-augmented with caching** |
| `/health` | GET | Health check (unchanged) |

---

## Token Usage & Cost

### Per Chat Request (with RAG)

```
System prompt                  ~150 tokens
User message                   ~50 tokens
Tool results (ML)              ~200 tokens
Dynamic: area stats            ~80 tokens    ← live SQL, no embedding cost
RAG: 3 comparable props        ~180 tokens
RAG: 2 static knowledge chunks ~300 tokens
RAG system injection           ~80 tokens
────────────────────────────────────────
Total per request              ~1,040 tokens ← vs ~400 without RAG (2.6x)
```

### Cost Estimate

| Task | Tokens | Estimated Cost |
|------|--------|---------------|
| One-time: ingest 40,200 property records | ~8M tokens | ~$0.32 |
| One-time: ingest static knowledge (~30 files) | ~150K tokens | ~$0.006 |
| Per-query embedding (retrieval) | ~100 tokens | ~$0.000004/query |
| Per-query GPT chat (with RAG) | ~1,040 tokens | ~$0.00015/query |
| `get_area_stats()` per query | 0 tokens | **Free** (SQL only) |
| **Total one-time ingestion** | | **< $0.35** |

> With Redis caching (1hr TTL), repeated similar queries cost **$0** for retrieval.
> `get_area_stats()` always free — no embedding needed, pure SQL.

---

## Implementation Timeline

| Phase | Task | Estimated Effort |
|-------|------|-----------------|
| Phase 1 | pgvector setup, Prisma schema, dependency install | 1 day |
| Phase 2 | Write static knowledge documents | 1–2 days |
| Phase 3 | `rag/embedder.py`, `rag/retriever.py` (+ `get_area_stats`), `context_builder.py`, `cache.py` | 1 day |
| Phase 4 | Update `chat_endpoint.py` with RAG injection | 0.5 day |
| Phase 5 | New `/comparable_properties` endpoint | 0.5 day |
| Phase 6 | Ingest scripts + run ingestion | 1 day |
| Phase 7 | Docker updates, end-to-end testing, prompt tuning | 1–2 days |
| **Total** | | **~6–8 days** |

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Vector DB | pgvector (extends PostgreSQL) | No new Docker service, uses existing infra |
| Embedding model | text-embedding-3-small | Best cost/performance ratio |
| LangChain scope | Embedder + PGVector + TextLoader only | Avoids over-engineering |
| Knowledge content | Static only (location, infra, characteristics) | Price data handled dynamically |
| Dynamic data | Live SQL `get_area_stats()` from `property_embeddings` | Always current, zero embedding cost |
| RAG trigger | Keyword-based conditional | Avoids token waste on non-property queries |
| Caching | Redis 1hr TTL | Eliminates redundant calls |
| Context size | 3 comparables + 2 knowledge chunks + area stats | Balances quality vs token cost |
| Knowledge update | Edit `.md` → re-run `ingest_knowledge.py` | Simple, version-controlled in Git |

---

## Environment Variables to Add

```env
# .env.example additions
EMBEDDING_MODEL=text-embedding-3-small
RAG_ENABLED=true
RAG_TOP_K_PROPERTIES=3
RAG_TOP_K_KNOWLEDGE=2
```