# Implementation Plan — Full Automation: House Price Intelligence System

**Tujuan**: Membuat seluruh sistem berjalan *end-to-end* secara otomatis melalui Docker Compose — dari data pipeline, model serving, hingga retraining loop — tanpa intervensi manual.

---

## Kondisi Saat Ini (Audit)

### ✅ Sudah Berfungsi
| Komponen | Status |
|---|---|
| `services/feature_engineer.py` | ✅ Fixed — 3 pipeline sinkron |
| `services/predictor.py` | ✅ Fixed — threshold dinamis + blending |
| `services/model_loader.py` | ✅ Fixed — load semua encoder |
| `metadata/*.json` | ✅ Sinkron dengan model (21 / 49 / 8 fitur) |
| `scripts/retrain_from_csv.py` | ✅ Replika persis notebook regresi |
| `scripts/retrain_klasifikasi.py` | ✅ Replika persis notebook klasifikasi |
| `scripts/retrain_clustering.py` | ✅ Replika persis notebook clustering |
| `prisma/schema.prisma` | ✅ Sudah ada semua tabel (Raw, Clean, Feedback, Retrain) |
| `scripts/ingest_*.py` | ✅ Sudah ada (raw data + cleaning per model) |
| `nginx.conf` | ✅ Routing benar (API `/api/v1/`, frontend `/`) |
| `docker-compose.yml` (infra) | ✅ Kafka, PostgreSQL, Redis, MLflow siap |

### ❌ Yang Masih Bermasalah / Belum Terhubung

| # | Masalah | File | Dampak |
|---|---|---|---|
| 1 | **`Dockerfile CMD ["sleep", "infinity"]`** — API tidak pernah start | `Dockerfile` | Container jalan tapi tidak serving |
| 2 | **Port API di-comment** `# ports: - "8080:8080"` | `docker-compose.yml` L104 | Tidak bisa direct access untuk debug |
| 3 | **`cluster_property` MCP missing `garasi`** — NameError saat dipanggil | `server.py` L205 | Clustering crash via MCP |
| 4 | **`retrain_pipeline.py` path model salah** — `BASE_DIR / save_name` bukan `BASE_DIR / "models" / save_name` | `pipelines/retrain_pipeline.py` L167 | Model baru tidak dimuat server |
| 5 | **`retrain_pipeline.py` PRICE_THRESHOLD hardcoded** | `pipelines/retrain_pipeline.py` L25 | Tidak sync dengan metadata |
| 6 | **Retrain hanya regresi** — klasifikasi & clustering tidak ikut update | `pipelines/retrain_pipeline.py` | 2 dari 3 model tidak pernah diperbarui |
| 7 | **Tidak ada init sequence** — database kosong saat pertama deploy | — | Semua pipeline gagal di awal |
| 8 | **`api/main.py` crash** jika model belum ready saat lifespan startup | `api/main.py` | API down saat cold start |
| 9 | **File debug di root** — masuk ke Docker image, membesar ukuran | `debug_*.py`, `catboost_info/` | Image besar, tidak perlu |
| 10 | **Tidak ada `.dockerignore`** | — | Data CSV 50MB masuk ke image |

---

## Arsitektur Target (Fully Automated)

```
docker-compose up
       │
       ├── [Infrastruktur]
       │     ├── postgresql (pgvector:pg16) :5435
       │     ├── kafka (KRaft mode)        :9092
       │     ├── redis                     :6379
       │     └── mlflow                    :5000
       │
       ├── [hpi_init — one-shot, exit setelah selesai]
       │     ├── prisma db push
       │     ├── ingest_raw_data.py        → RawProperty (~24K rows)
       │     ├── cleaning_pipeline_*.py    → CleanProperty* (3 tabel)
       │     ├── setup_encoder.py          → target_encoder.pkl
       │     ├── ingest_properties.py      → pgvector RAG
       │     └── ingest_knowledge.py       → KnowledgeBase
       │
       ├── [hpi_api — FastAPI :8080]
       │     └── /api/v1/predict_price, /classify_segment, /cluster_property
       │
       ├── [hpi_mcp_server — FastMCP :8000]
       │     └── tools: predict_price, classify_segment, cluster_property
       │
       ├── [hpi_frontend — Next.js :3000]
       │
       ├── [nginx :80]
       │     ├── /api/v1/* → hpi_api:8080
       │     └── /         → hpi_frontend:3000
       │
       └── [Kafka Consumers]
             ├── consumer_regression
             ├── consumer_classification
             ├── consumer_clustering
             └── consumer_feedback → trigger retrain jika ≥100 feedback
```

---

## Phase 1 — Perbaikan Kritis (Blocking Issues)

> Masalah yang membuat sistem **tidak bisa jalan sama sekali**.

### [MODIFY] `Dockerfile`

```diff
- CMD ["sleep", "infinity"]
+ CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Alasan: Default CMD harus mengisi sesuatu yang masuk akal. Setiap service di docker-compose sudah override `command:` sendiri (mcp_server pakai `python server.py`, consumer pakai `python -m kafka.*`), jadi ini hanya mempengaruhi fallback — tetapi "sleep infinity" adalah anti-pattern.

---

### [MODIFY] `server.py`

Tambah parameter `garasi` ke tool `cluster_property`:

```diff
  @mcp.tool()
  async def cluster_property(
      luas_tanah: ...,
      luas_bangunan: ...,
      kamar_tidur: ...,
      kamar_mandi: ...,
      lokasi: ...,
      harga: ... = None,
+     garasi: Annotated[int, "Kapasitas garasi (0-5). Default 0."] = 0,
      ctx: Context = None,
  ) -> dict:
      ...
      result = await loop.run_in_executor(
-         None, lambda: _cluster(luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, lokasi, harga, garasi),
+         None, lambda: _cluster(luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, lokasi, harga, garasi),
      )
```

---

### [MODIFY] `pipelines/retrain_pipeline.py`

Fix 3 masalah sekaligus:

```diff
- PRICE_THRESHOLD = 1_200_000_000.0
+ # Baca dari metadata saat runtime — tidak hardcode

  def _train_and_evaluate(df):
      from services.model_loader import models
      models.load()
+     batas = float(models.meta_regresi["batas_segmen"])

-     mask_low_train = harga_asli <= PRICE_THRESHOLD
+     mask_low_train = harga_asli <= batas

      # Fix path penyimpanan model
-     model.save_model(str(BASE_DIR / save_name))
+     model.save_model(str(BASE_DIR / "models" / save_name))
```

Tambah retrain klasifikasi dan clustering:

```python
# Setelah retrain regresi:
import subprocess, sys

def _retrain_all_models():
    for script in ["scripts/retrain_klasifikasi.py", "scripts/retrain_clustering.py"]:
        result = subprocess.run([sys.executable, script], cwd=BASE_DIR,
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[Retrain] WARNING {script}: {result.stderr[:500]}")
        else:
            print(f"[Retrain] {script}: OK")
```

---

## Phase 2 — Tambah `hpi_init` Service (One-Shot Bootstrap)

**File**: `docker-compose.yml`

Tambah service baru setelah `mlflow`:

```yaml
  # ── One-shot DB bootstrap (runs once then exits) ─────────────────────────
  init:
    build: .
    container_name: hpi_init
    environment:
      DATABASE_URL: postgresql://hpi:hpi_secret@postgresql:5432/house_price_intel
      PYTHONPATH: /app
    command: >
      sh -c "
        prisma db push --accept-data-loss &&
        python scripts/ingest_raw_data.py &&
        python scripts/cleaning_pipeline_regression.py &&
        python scripts/cleaning_pipeline_classification.py &&
        python scripts/cleaning_pipeline_clustering.py &&
        python scripts/setup_encoder.py &&
        python scripts/ingest_properties.py &&
        python scripts/ingest_knowledge.py &&
        echo '[hpi_init] Bootstrap selesai.'
      "
    depends_on:
      postgresql:
        condition: service_healthy
      kafka:
        condition: service_healthy
    networks:
      - hpi-network
    restart: "no"
```

Update dependency semua service utama agar tunggu `init`:

```yaml
  api:
    depends_on:
      postgresql:
        condition: service_healthy
      kafka:
        condition: service_healthy
      redis:
        condition: service_healthy
      init:
        condition: service_completed_successfully   # ← TAMBAH INI

  mcp_server:
    depends_on:
      # ... (sama)
      init:
        condition: service_completed_successfully   # ← TAMBAH INI
```

---

## Phase 3 — Fix `docker-compose.yml` — Ports dan Dependencies

```diff
  api:
-   # ports:
-   #   - "8080:8080"
+   ports:
+     - "8080:8080"   # expose untuk debugging langsung (nginx tetap di 80)
```

Tambahkan healthcheck ke `mcp_server`:

```yaml
  mcp_server:
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/ || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
```

---

## Phase 4 — Graceful API Startup (`api/main.py`)

```diff
  @asynccontextmanager
  async def lifespan(app: FastAPI):
      from services.model_loader import models
-     models.load()
+     try:
+         models.load()
+         print("[API] Models loaded OK")
+     except Exception as e:
+         print(f"[API] WARNING: Model loading failed: {e}")
+         # Lanjutkan — /health harus tetap respond meski model belum siap

      app.state.db = Prisma()
      await app.state.db.connect()
      yield
      await app.state.db.disconnect()
```

---

## Phase 5 — Cleanup: `.dockerignore` dan File Debug

### [NEW] `.dockerignore`

```
# Development artifacts
catboost_info/
debug_*.py
*.log
.env

# Python cache
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# Git
.git/
.gitignore

# Verification scripts (tidak perlu di production)
scripts/verify_pipeline.py

# Frontend build cache
frontend/.next/
frontend/node_modules/
```

### File debug yang perlu dipindahkan atau dihapus

| File | Aksi |
|---|---|
| `debug_features.py` | Pindah ke `dev/` atau hapus |
| `debug_importance.py` | Pindah ke `dev/` atau hapus |
| `debug_lokasi.py` | Pindah ke `dev/` atau hapus |
| `debug_model.py` | Pindah ke `dev/` atau hapus |
| `debug_verify.py` | Pindah ke `dev/` atau hapus |
| `catboost_info/` | Tambah ke `.dockerignore` |

---

## Urutan Eksekusi Setelah Semua Fix

```bash
# 1. Build image baru
docker-compose build --no-cache

# 2. Start infrastruktur
docker-compose up -d postgresql kafka redis mlflow

# 3. Tunggu healthy, lalu jalankan init
docker-compose up init
# Tunggu output: "[hpi_init] Bootstrap selesai." → exit 0

# 4. Start semua service utama
docker-compose up -d api mcp_server frontend nginx \
  consumer_regression consumer_classification \
  consumer_clustering consumer_feedback

# 5. Verifikasi
curl http://localhost/health
curl http://localhost/api/v1/predict_price \
  -X POST -H "Content-Type: application/json" \
  -d '{"kamar_tidur":3,"kamar_mandi":2,"garasi":1,
       "luas_tanah":100,"luas_bangunan":80,"lokasi":"Cinere"}'

# Cek row count
docker exec hpi_postgres psql -U hpi -d house_price_intel -c "
  SELECT
    (SELECT COUNT(*) FROM \"RawProperty\") AS raw,
    (SELECT COUNT(*) FROM \"CleanPropertyRegression\") AS regression,
    (SELECT COUNT(*) FROM \"CleanPropertyClassification\") AS classification,
    (SELECT COUNT(*) FROM \"CleanPropertyClustering\") AS clustering;
"
```

---

## Ringkasan Perubahan

| File | Aksi | Prioritas |
|---|---|---|
| `Dockerfile` | Fix CMD | 🔴 CRITICAL |
| `server.py` | Tambah parameter `garasi` ke MCP tool | 🔴 CRITICAL |
| `pipelines/retrain_pipeline.py` | Fix path, fix threshold, tambah retrain semua model | 🔴 CRITICAL |
| `docker-compose.yml` | Tambah `hpi_init`, uncomment port 8080, fix depends_on | 🟠 HIGH |
| `api/main.py` | Graceful model loading | 🟠 HIGH |
| `.dockerignore` | Buat baru | 🟡 MEDIUM |
| `debug_*.py` | Pindah ke `dev/` | 🟡 MEDIUM |
