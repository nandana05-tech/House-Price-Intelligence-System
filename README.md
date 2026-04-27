# House Price Intelligence System (HPI)

An LLM-powered house price prediction system for **Depok, West Java** — built as a portfolio project with a production-grade 7-layer architecture using OpenAI GPT as an intelligent LLM router, three ML models (CatBoost + KMeans + UMAP) via MCP Protocol, Kafka as the event bus, MLflow for experiment tracking, and Prisma ORM for data persistence.

> **Dataset**: [Data Harga Rumah di Depok](https://www.kaggle.com/datasets/dimasmaulanaputra/data-harga-rumah-di-depok) — 40,200 property records from Depok, West Java.

---

## Architecture

```
Client (Chat/API)
    ↓
Nginx (Load Balancer :80)
    ↓
FastAPI (api :8080)
    ↓
OpenAI GPT (LLM Router)
    ↓ MCP Protocol
MCP Server (server.py) ──→ Kafka (fire-and-forget audit)
    ↓                           ↓
Direct Prediction        ML Service Pods (consumer_*.py)
                                ↓
                          MLflow Tracking
                                ↓
                      PostgreSQL + Redis (Prisma ORM)
                                ↓
                        Feedback → Retrain Loop
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Router | OpenAI GPT (via MCP Protocol) |
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
hpi/
├── server.py                        ← MCP Server (3 tools)
├── docker-compose.yml               ← Full infrastructure definition
├── Dockerfile                       ← Multi-stage build (python:3.14-rc-slim)
├── nginx.conf                       ← Load balancer config
├── .env.example                     ← Copy to .env and fill in API key
├── prisma/schema.prisma             ← Database schema
│
├── api/
│   ├── main.py                      ← FastAPI app + lifespan
│   ├── predict_endpoint.py          ← POST /api/v1/predict_price, classify, cluster
│   ├── feedback_endpoint.py         ← POST /api/v1/feedback
│   └── chat_endpoint.py             ← POST /api/v1/chat (LLM agent)
│
├── services/
│   ├── model_loader.py              ← Singleton loader for all ML artifacts
│   ├── feature_engineer.py          ← Feature engineering (mirrors training pipeline)
│   └── predictor.py                 ← Core prediction logic
│
├── kafka/
│   ├── topics.py                    ← Topic name constants
│   ├── consumer_regression.py       ← Regression ML pod
│   ├── consumer_classification.py   ← Classification ML pod
│   ├── consumer_clustering.py       ← Clustering ML pod
│   └── consumer_feedback.py         ← Persists feedback to DB
│
├── mlflow_utils/
│   ├── tracker.py                   ← Logs predictions to MLflow
│   └── model_registry.py            ← Register & promote model versions
│
├── pipelines/
│   ├── retrain_trigger.py           ← Checks threshold & triggers retraining
│   └── retrain_pipeline.py          ← Retrains and registers new model
│
├── models/                          ← ML artifacts (pkl, cbm)
│   ├── model_low.cbm
│   ├── model_high.cbm
│   ├── model_clf.cbm
│   ├── kmeans_model.pkl
│   ├── umap_reducer.pkl
│   ├── scaler.pkl
│   └── target_encoder.pkl
│
├── metadata/                        ← Model metadata (JSON)
│   ├── metadata_regresi.json
│   ├── metadata_klasifikasi.json
│   └── metadata_clustering.json
│
└── scripts/
    ├── predict_csv.py               ← Batch prediction from CSV
    ├── setup_encoder.py             ← Target encoder setup
    └── client.py                    ← Test client
```

---

## Quick Start

### 1. Setup Environment

```bash
cp .env.example .env
# Edit .env: fill in OPENAI_API_KEY
```

### 2. Build & Run

```bash
docker-compose build        # Build image (first time only)
docker-compose up -d        # Start all services
```

Wait ~1–2 minutes for all containers to become healthy:

```bash
docker-compose ps
```

### 3. Access Services

| Service | URL |
|---------|-----|
| API + Swagger UI | http://localhost/docs |
| API (direct) | http://localhost:8080/docs |
| MLflow UI | http://localhost:5000 |
| PostgreSQL | localhost:5435 (user: `hpi`, pass: `hpi_secret`) |
| Kafka (external) | localhost:9092 |
| Redis | localhost:6379 |

> **RAM required**: ~2GB for 6 Python containers with ML models loaded in memory.

---

## API Endpoints

Base URL: `http://localhost/api/v1` (via Nginx) or `http://localhost:8080/api/v1` (direct)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict_price` | POST | House price estimation (dual-model CatBoost) |
| `/api/v1/classify_segment` | POST | Price segment classification (4 classes) |
| `/api/v1/cluster_property` | POST | Property clustering (6 clusters, KMeans+UMAP) |
| `/api/v1/feedback` | POST | Submit price correction feedback |
| `/api/v1/chat` | POST | Chat with LLM agent (auto-calls MCP tools) |
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

#### 6. `POST /api/v1/chat` — Chat with LLM Agent

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

## MCP Tools

The MCP Server (`server.py`) exposes 3 tools that are automatically invoked by the LLM router:

| Tool | Input | Output |
|------|-------|--------|
| `predict_price` | kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, lokasi | harga_estimasi, model_digunakan, mape |
| `classify_segment` | + harga (optional) | kelas_id, kelas_label, probabilitas |
| `cluster_property` | + harga (optional) | cluster_id, cluster_label, cluster_summary |

> If `harga` is omitted for `classify_segment` or `cluster_property`, it is automatically estimated via `predict_price`.

---

## Kafka Topics

| Topic | Producer | Consumer |
|-------|----------|----------|
| `property.prediction.regression` | MCP Server | consumer_regression |
| `property.prediction.classification` | MCP Server | consumer_classification |
| `property.prediction.clustering` | MCP Server | consumer_clustering |
| `property.feedback` | Feedback API | consumer_feedback |

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

## Batch Prediction (CSV)

```bash
python scripts/predict_csv.py your_data.csv
```

**Required CSV columns**: `kamar_tidur`, `kamar_mandi`, `garasi`, `luas_tanah`, `luas_bangunan`, `lokasi`

**Output**: `your_data_predictions.json` — predicted_price, segment, and cluster per row + MLflow log.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required** |
| `DATABASE_URL` | `postgresql://hpi:hpi_secret@postgresql:5432/house_price_intel` | Prisma DB URL |
| `REDIS_URL` | `redis://redis:6379/0` | Cache URL |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` | Kafka internal bootstrap |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URI |
| `RETRAIN_FEEDBACK_THRESHOLD` | `100` | Number of feedback entries before retraining |
| `COMPOSE_PROJECT_NAME` | `hpi` | Docker container name prefix |