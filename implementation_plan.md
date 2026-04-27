# Dockerize HPI Application for Production

Memasukkan seluruh aplikasi Python (FastAPI, MCP Server, dan 4 ML Pods) ke dalam Docker agar sepenuhnya siap untuk lingkungan *production*. 

Dengan arsitektur ini, Anda cukup menjalankan `docker-compose up -d`, dan seluruh ekosistem aplikasi (Database, Kafka, API, MCP, dan Consumer) akan otomatis berjalan dan saling terhubung dalam satu jaringan tertutup yang aman.

## User Review Required

> [!IMPORTANT]
> Karena kita akan menjalankan 6 *container* Python (API, MCP, 4 Consumers) di dalam Docker, masing-masing akan memuat model *Machine Learning* ke RAM. 
> Total RAM yang dibutuhkan untuk keseluruhan sistem ini diperkirakan sekitar **1.5 GB - 2 GB**. Pastikan *server* atau laptop Anda memiliki alokasi RAM yang cukup untuk Docker.

> [!WARNING]
> Arsitektur jaringan Kafka akan kita modifikasi. Kafka akan memiliki dua *listener*:
> 1. `INTERNAL` (port 29092) - Digunakan oleh *container* Python di dalam Docker.
> 2. `PLAINTEXT` (port 9092) - Tetap bisa diakses dari *localhost* (luar Docker) jika Anda butuh melakukan proses *debugging* manual.

## QnA
- Q: Apakah Anda setuju dengan pendekatan menggunakan 1 `Dockerfile` (satu *image* utama) yang di-*reuse* oleh ke-6 layanan (*API, MCP, 4 Consumers*) melalui pengaturan `command` di `docker-compose.yml`? Pendekatan ini sangat menghemat ruang penyimpanan *hardisk* Anda.
- A: Saya setuju dengan pendekatan menggunakan 1 `Dockerfile` (satu *image* utama) yang di-*reuse* oleh ke-6 layanan (*API, MCP, 4 Consumers*) melalui pengaturan `command` di `docker-compose.yml`

## Proposed Changes

### 1. `Dockerfile`
Akan ditambahkan di *root* direktori.
- Menggunakan `python:3.10-slim`.
- Meng-instal *dependencies* sistem (termasuk Node.js yang wajib untuk `prisma generate`).
- Menyalin semua kode sumber.
- Menjalankan instalasi paket (`pip install`) dan *generate* Prisma Client Python.

### 2. `docker-compose.yml`
#### [MODIFY] docker-compose.yml
- Mengubah *environment* Kafka agar mensupport koneksi inter-container (`INTERNAL://kafka:29092`).
- Menambahkan *service* berikut:
  - `api`: Menjalankan FastAPI di port `8080`. Perintah awalnya akan mengeksekusi `prisma db push` otomatis untuk memastikan skema database mutakhir sebelum server menyala.
  - `mcp_server`: Menjalankan `python server.py`.
  - `consumer_regression`: Menjalankan `python -m kafka.consumer_regression`.
  - `consumer_classification`: Menjalankan `python -m kafka.consumer_classification`.
  - `consumer_clustering`: Menjalankan `python -m kafka.consumer_clustering`.
  - `consumer_feedback`: Menjalankan `python -m kafka.consumer_feedback`.
- Semua layanan Python ini akan secara otomatis disuntikkan URL internal (*environment variables*), seperti:
  - `DATABASE_URL=postgresql://hpi:hpi_secret@postgresql:5432/house_price_intel`
  - `KAFKA_BOOTSTRAP_SERVERS=kafka:29092`
  - `MLFLOW_TRACKING_URI=http://mlflow:5000`

## Verification Plan

### Automated Tests
1. Menjalankan `docker-compose build` untuk memastikan `Dockerfile` sukses mem-*build image* tanpa masalah *dependency*.
2. Menjalankan `docker-compose up -d`.
3. Memastikan semua status container `healthy` atau `running` menggunakan `docker ps`.
4. Mencoba API via `http://localhost:8080/health` dan Swagger UI untuk memverifikasi bahwa API dapat berjalan di dalam Docker dan terhubung dengan sukses ke Kafka dan PostgreSQL internal.
