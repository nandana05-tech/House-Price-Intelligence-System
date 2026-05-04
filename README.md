# House Price Intelligence System (HPI)

An LLM-powered house price intelligence platform for **Depok, West Java** — built as a portfolio-grade project with FastAPI backend, OpenAI GPT via MCP, CatBoost and KMeans/UMAP ML models, Kafka event streaming, MLflow tracking, and Prisma ORM for persistence.

> **Dataset**: [Data Harga Rumah di Depok](https://www.kaggle.com/datasets/dimasmaulanaputra/data-harga-rumah-di-depok) — 40,200 property records from Depok, West Java.

---

## Architecture

```
Client (Frontend / API)
    ↓
Nginx (Reverse Proxy)
    ↓
FastAPI Backend (api/main.py)
    ↓
OpenAI GPT / Chat Agent
    ↓
MCP Server (server.py)
    ├─> predict_price
    ├─> classify_segment
    └─> cluster_property
        ↓
      Kafka audit events
        ↓
  ML service pods (kafka/consumer_*.py)
        ↓
   MLflow tracking + PostgreSQL + Redis
        ↓
    Feedback / retraining loop
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Router | OpenAI GPT (via MCP Protocol) |
| Frontend | Next.js 14 |
| API | FastAPI + Uvicorn |
| Load Balancer | Nginx |
| ML Models | CatBoost, KMeans, UMAP (scikit-learn) |
| Event Bus | Apache Kafka 3.7 (KRaft mode) |
| Experiment Tracking | MLflow 2.12 |
| Database | PostgreSQL 16 + Prisma ORM |
| Cache | Redis 7 |
| Containerization | Docker + Docker Compose |

---

## Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| Regression Low (≤ 1.2B IDR) | MAPE | 10.08% |
| Regression High (> 1.2B IDR) | MAPE | 10.67% |
| Segment Classifier | Accuracy | 99.37% |
| Segment Classifier | Macro F1 | 99.37% |
| Clustering | Silhouette Score | 0.591 |
| Clustering | Davies-Bouldin Index | 1.179 |
| Clustering | Calinski-Harabasz | 8681.77 |

### Classification Performance per Class

| Class | Label | Precision | Recall | F1 | Support |
|-------|-------|-----------|--------|----|---------|
| 0 | Budget (≤ 745M IDR) | 99.92% | 99.76% | 99.84% | 1,231 |
| 1 | Mid-range (745M – 1.3B IDR) | 99.07% | 99.30% | 99.19% | 1,290 |
| 2 | Upper (1.3B – 2.645B IDR) | 98.72% | 99.06% | 98.89% | 1,169 |
| 3 | Luxury (> 2.645B IDR) | 99.76% | 99.35% | 99.55% | 1,230 |

### Clustering Configuration (KMeans + UMAP)

| Parameter | Value |
|-----------|-------|
| Number of Clusters | 6 |
| UMAP Components | 10 |
| UMAP n_neighbors | 30 |
| KMeans n_init | 50 |

---

## Project Structure

```
notifications/
├── api/                        ← FastAPI endpoints and API logic
├── catboost_info/              ← CatBoost training diagnostics and logs
├── context/                    ← Runtime context and helper resources
├── data/                       ← Raw and processed dataset files
├── dev/                        ← Debugging and exploration scripts
├── docker-compose.yml          ← Full service orchestration
├── Dockerfile                  ← Backend container build
├── frontend/                   ← Next.js frontend application
├── kafka/                      ← Kafka topic constants and consumer pods
├── knowledge/                  ← Area profiles, FAQs, market rules for RAG
├── mlflow_utils/               ← MLflow logging and model registry utilities
├── metadata/                   ← Model metadata and config files
├── models/                     ← Serialized ML artifacts and binaries
├── pipelines/                  ← Retraining trigger and retraining pipeline
├── prisma/                     ← Prisma ORM schema definitions
├── rag/                        ← RAG retrieval and embedding support
├── scripts/                    ← Utility scripts for batch and ingest workflows
├── services/                   ← Feature engineering and prediction logic
├── server.py                   ← MCP server with callable LLM tools
├── nginx.conf                  ← Reverse proxy configuration
├── .env.example                ← Environment variable template
├── pyproject.toml              ← Python package and dependency config
├── README.md                   ← Project documentation
└── TODO.md                     ← Development notes and next tasks
```

---

## Quick Start

### 1. Setup Environment

```bash
cp .env.example .env
# Edit .env and add OPENAI_API_KEY
```

### 2. Build & Run

```bash
docker compose build
docker compose up -d
```

Wait until all containers are healthy.

```bash
docker compose ps
```

### 3. Check Logs

```bash
docker compose logs -f api
```

### 4. Access Services

| Service | URL |
|---------|-----|
| API | http://localhost:8080 |
| Swagger / OpenAPI | http://localhost:8080/docs |
| Health | http://localhost:8080/health |
| MLflow | http://localhost:5000 |
| PostgreSQL | localhost:5432 |
| Kafka | localhost:9092 |
| Redis | localhost:6379 |

> For local development, use the ports defined in `.env.example` and `docker-compose.yml`.

---

## API Endpoints

Base URL: `http://localhost/api/v1` (via Nginx) or `http://localhost:8080/api/v1` (direct)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict_price` | POST | Estimate house price using dual-model CatBoost |
| `/api/v1/classify_segment` | POST | Classify a property into one of 4 price tiers |
| `/api/v1/cluster_property` | POST | Assign market cluster with KMeans + UMAP |
| `/api/v1/comparable_properties` | POST | Find similar properties via vector search |
| `/api/v1/feedback` | POST | Submit corrected price feedback |
| `/api/v1/chat` | POST | Chat with the LLM agent and perform RAG-enhanced replies |
| `/api/v1/analytics/areas` | GET | Retrieve aggregated area analytics |
| `/health` | GET | Health check |

### Example Requests (Postman / curl)

> **Base URL**: `http://localhost/api/v1`
> **Required header**: `Content-Type: application/json`
> **Windows CMD**: Use `\"` to escape double quotes in JSON.

---

#### 1. `GET /health` — Health Check

```
GET http://localhost/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "hpi-api"
}
```

---

#### 2. `POST /api/v1/predict_price` — House Price Estimation

```
POST http://localhost/api/v1/predict_price
Content-Type: application/json
```

**Request Body:**
```json
{
  "kamar_tidur": 3,
  "kamar_mandi": 2,
  "garasi": 1,
  "luas_tanah": 120.0,
  "luas_bangunan": 90.0,
  "lokasi": "Cinere"
}
```

**Field Validation:**
- `kamar_tidur` _(bedrooms)_: integer, 1–10
- `kamar_mandi` _(bathrooms)_: integer, 1–10
- `garasi` _(garage capacity)_: integer, 0–5
- `luas_tanah` _(land area m²)_ & `luas_bangunan` _(building area m²)_: float, > 0
- `lokasi` _(district/area)_: string, minimum 2 characters

**Response:**
```json
{
  "harga_estimasi": 1850000000,
  "harga_estimasi_format": "Rp 1.85 Miliar",
  "model_digunakan": "model_high",
  "mape_persen": 10.67,
  "batas_segmen_idr": 1200000000.0,
  "latency_ms": 12.5
}
```

---

#### 3. `POST /api/v1/classify_segment` — Price Segment Classification

```
POST http://localhost/api/v1/classify_segment
Content-Type: application/json
```

**Request Body (without price — estimated automatically via regression):**
```json
{
  "kamar_tidur": 3,
  "kamar_mandi": 2,
  "garasi": 1,
  "luas_tanah": 120.0,
  "luas_bangunan": 90.0,
  "lokasi": "Cinere"
}
```

**Request Body (with manual price):**
```json
{
  "kamar_tidur": 3,
  "kamar_mandi": 2,
  "garasi": 1,
  "luas_tanah": 120.0,
  "luas_bangunan": 90.0,
  "lokasi": "Cinere",
  "harga": 1850000000
}
```

**Response:**
```json
{
  "kelas_id": 2,
  "kelas_label": "Atas (1.3 – 2.645 miliar)",
  "probabilitas": {
    "Murah (≤ 745 juta)": 0.0002,
    "Menengah (745 juta – 1.3 miliar)": 0.0134,
    "Atas (1.3 – 2.645 miliar)": 0.9801,
    "Mewah (> 2.645 miliar)": 0.0063
  },
  "harga_digunakan": 1850000000,
  "harga_sumber": "estimated_by_regression",
  "akurasi_model": 0.9937,
  "latency_ms": 45.2
}
```

---

#### 4. `POST /api/v1/cluster_property` — Property Clustering

```
POST http://localhost/api/v1/cluster_property
Content-Type: application/json
```

**Request Body (without price — estimated automatically via regression):**
```json
{
  "kamar_tidur": 3,
  "kamar_mandi": 2,
  "garasi": 1,
  "luas_tanah": 120.0,
  "luas_bangunan": 90.0,
  "lokasi": "Cinere"
}
```

**Request Body (with manual price):**
```json
{
  "kamar_tidur": 3,
  "kamar_mandi": 2,
  "garasi": 1,
  "luas_tanah": 120.0,
  "luas_bangunan": 90.0,
  "lokasi": "Cinere",
  "harga": 1850000000
}
```

**Response:**
```json
{
  "cluster_id": 3,
  "cluster_label": "Cluster-3",
  "cluster_summary": {
    "cluster_id": 3,
    "jumlah": 1842,
    "rata_harga": 1750000000,
    "rata_luas_tanah": 115.4,
    "rata_luas_bangunan": 88.2
  },
  "harga_digunakan": 1850000000,
  "harga_sumber": "input",
  "silhouette_score": 0.591,
  "latency_ms": 38.7
}
```

---

#### 5. `POST /api/v1/feedback` — Submit Price Correction

```
POST http://localhost/api/v1/feedback
Content-Type: application/json
```

**Request Body:**
```json
{
  "prediction_id": "pred_abc123",
  "kamar_tidur": 3,
  "kamar_mandi": 2,
  "garasi": 1,
  "luas_tanah": 120.0,
  "luas_bangunan": 90.0,
  "lokasi": "Cinere",
  "harga_prediksi": 1850000000,
  "harga_asli": 1950000000,
  "sumber": "user_feedback"
}
```

**Notes:**
- `prediction_id`: optional
- `harga_prediksi` & `harga_asli`: minimum IDR 10,000,000 — maximum IDR 500,000,000,000
- `sumber`: defaults to `"user_feedback"`

**Response:**
```json
{
  "success": true,
  "message": "Feedback diterima. Selisih prediksi: 5.4%",
  "selisih_persen": 5.41
}
```

---

#### 6. `POST /api/v1/comparable_properties` — Find Similar Properties

```
POST http://localhost/api/v1/comparable_properties
Content-Type: application/json
```

**Request Body:**
```json
{
  "kamar_tidur": 3,
  "kamar_mandi": 2,
  "garasi": 1,
  "luas_tanah": 120.0,
  "luas_bangunan": 90.0,
  "lokasi": "Cinere",
  "harga": 1850000000,
  "top_k": 5
}
```

**Field Validation:**
- `top_k`: integer, 1–20 (default: 5)
- All other fields same as predict_price

**Response:**
```json
{
  "comparables": [
    {
      "id": "prop_001",
      "kamar_tidur": 3,
      "kamar_mandi": 2,
      "garasi": 1,
      "luas_tanah": 118.0,
      "luas_bangunan": 89.0,
      "lokasi": "Cinere",
      "harga": 1840000000,
      "similarity_score": 0.98
    },
    {
      "id": "prop_002",
      "kamar_tidur": 3,
      "kamar_mandi": 2,
      "garasi": 1,
      "luas_tanah": 125.0,
      "luas_bangunan": 92.0,
      "lokasi": "Cinere",
      "harga": 1870000000,
      "similarity_score": 0.96
    }
  ],
  "count": 2
}
```

---

#### 7. `POST /api/v1/chat` — Chat with LLM Agent

```
POST http://localhost/api/v1/chat
Content-Type: application/json
```

**Request Body:**
```json
{
  "message": "What is the estimated price of a house with 3 bedrooms, 2 bathrooms, 1 garage, 120m2 land, 90m2 building in Cinere?"
}
```

**Optional fields:**
```json
{
  "message": "What is the estimated price of a house with 3 bedrooms, 2 bathrooms, 1 garage, 120m2 land, 90m2 building in Cinere?",
  "history": [],
  "language": "id"
}
```

**Other example messages:**
```json
{ "message": "My house in Beji has 4 bedrooms, 3 bathrooms, 200m2 land, 150m2 building. What segment does it belong to?" }
```
```json
{ "message": "Analyze this property: Sawangan, 3BR, 2BA, 180m2 land, 120m2 building, price 1.5B IDR" }
```

**Response:**
```json
{
  "reply": "Based on the model analysis, the house in Cinere with 3 bedrooms, 2 bathrooms, 1 garage, 120m² land and 90m² building is estimated at Rp 1.85 Billion (MAPE ~10.67%). This property falls in the Upper segment with 98% confidence.",
  "tools_used": ["predict_price", "classify_segment"]
}
```

---

#### 8. `GET /api/v1/analytics/areas` — Get Area Analytics

```
GET http://localhost/api/v1/analytics/areas
```

**Response (returns top 10 areas):**
```json
[
  {
    "nama": "Cinere",
    "avg_per_m2": 15430000.5,
    "total_data": 4850,
    "trend": "±5.2%",
    "segmen_dom": "Atas",
    "catatan": "The data is analyzed dynamically based on the classification model"
  },
  {
    "nama": "Beji",
    "avg_per_m2": 14120000.3,
    "total_data": 4230,
    "trend": "±6.1%",
    "segmen_dom": "Menengah",
    "catatan": "The data is analyzed dynamically based on the classification model"
  },
  {
    "nama": "Sawangan",
    "avg_per_m2": 13890000.8,
    "total_data": 3950,
    "trend": "±4.8%",
    "segmen_dom": "Menengah",
    "catatan": "The data is analyzed dynamically based on the classification model"
  }
]
```

---

## MCP Tools

The MCP Server (`server.py`) exposes three callable tools:

| Tool | Purpose |
|------|---------|
| `predict_price` | Estimate property price with CatBoost |
| `classify_segment` | Predict price segment class (4 classes) |
| `cluster_property` | Determine market cluster with KMeans + UMAP |

Each tool publishes audit events to Kafka for asynchronous logging and retraining.

---

## Kafka Topics

| Topic | Description |
|-------|-------------|
| `property.prediction.regression` | Regression prediction audit events |
| `property.prediction.classification` | Classification prediction audit events |
| `property.prediction.clustering` | Clustering prediction audit events |
| `property.feedback` | User feedback events for retraining |

---

## Feedback & Retrain Loop

```
User submits price correction → POST /api/v1/feedback
    → Kafka (property.feedback)
    → consumer_feedback → PostgreSQL (Feedback table)
    → If count >= RETRAIN_FEEDBACK_THRESHOLD (default: 100)
    → retrain_pipeline.py → new model → MLflow Registry
    → Hot-reload model singleton
```

---

## Batch Prediction

Run batch CSV prediction with:

```bash
python scripts/predict_csv.py data/input.csv
```

Required CSV columns:
- `kamar_tidur`
- `kamar_mandi`
- `garasi`
- `luas_tanah`
- `luas_bangunan`
- `lokasi`

The script outputs JSON predictions and logs the batch to MLflow.

---

## Environment Variables

The repository uses `.env.example` for environment configuration.

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPOSE_PROJECT_NAME` | `hpi` | Docker Compose project name |
| `MCP_TRANSPORT` | `streamable-http` | MCP transport mode |
| `MCP_PORT` | `8000` | MCP server port |
| `OPENAI_API_KEY` | `...` | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `KAFKA_ENABLED` | `true` | Enable Kafka auditing |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka bootstrap servers |
| `KAFKA_GROUP_ID_REGRESSION` | `hpi-regression-pod` | Regression consumer group |
| `KAFKA_GROUP_ID_CLASSIFICATION` | `hpi-classification-pod` | Classification consumer group |
| `KAFKA_GROUP_ID_CLUSTERING` | `hpi-clustering-pod` | Clustering consumer group |
| `KAFKA_GROUP_ID_FEEDBACK` | `hpi-feedback-pod` | Feedback consumer group |
| `DATABASE_URL` | `postgresql://hpi:hpi_secret@127.0.0.1:5435/house_price_intel?schema=prisma` | Prisma/PostgreSQL database URL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis cache URL |
| `PREDICTION_CACHE_TTL` | `3600` | Prediction cache TTL in seconds |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking URI |
| `MLFLOW_EXPERIMENT_REGRESSION` | `house-price-regression` | Regression MLflow experiment |
| `MLFLOW_EXPERIMENT_CLASSIFICATION` | `house-price-classification` | Classification MLflow experiment |
| `MLFLOW_EXPERIMENT_CLUSTERING` | `house-price-clustering` | Clustering MLflow experiment |
| `RETRAIN_FEEDBACK_THRESHOLD` | `100` | Feedback count threshold for retraining |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name for RAG |
| `RAG_ENABLED` | `true` | Enable RAG pipeline |
| `RAG_TOP_K_PROPERTIES` | `3` | Number of comparable properties returned by RAG |
| `RAG_TOP_K_KNOWLEDGE` | `2` | Number of knowledge snippets returned by RAG |