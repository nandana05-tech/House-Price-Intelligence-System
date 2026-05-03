"""Verifikasi akhir semua implementasi plan."""
import os

files_to_check = {
    "Dockerfile": [
        ("sleep infinity", False, "CMD tidak lagi sleep infinity"),
        ("uvicorn", True, "CMD sudah pakai uvicorn"),
    ],
    "server.py": [
        ("garasi: Annotated[int", True, "garasi parameter di cluster_property"),
    ],
    "pipelines/retrain_pipeline.py": [
        ("MODELS_DIR / save_name", True, "simpan model ke MODELS_DIR"),
        ("model.save_model(str(BASE_DIR /", False, "tidak lagi save ke BASE_DIR root"),
        ("batas_segmen", True, "threshold dinamis dari metadata"),
        ("_retrain_classification_and_clustering", True, "retrain semua model"),
    ],
    "docker-compose.yml": [
        ("hpi_init", True, "service init ada"),
        ("service_completed_successfully", True, "dependency init benar"),
        ("8080:8080", True, "port 8080 expose"),
        ("8000:8000", True, "port MCP 8000 expose"),
        ("restart: unless-stopped", True, "consumer restart policy"),
    ],
    "api/main.py": [
        ("from prisma import Prisma", True, "import Prisma ada"),
        ("WARNING: ML model loading failed", True, "graceful error message"),
    ],
    ".dockerignore": [
        ("catboost_info/", True, "catboost_info excluded"),
        ("debug_*.py", True, "debug files excluded"),
    ],
}

all_ok = True
for fname, checks in files_to_check.items():
    try:
        with open(fname, encoding="utf-8") as f:
            content = f.read()
        file_ok = True
        for pattern, should_exist, label in checks:
            found = pattern in content
            ok = (found == should_exist)
            if not ok:
                all_ok = False
                file_ok = False
                print(f"  [FAIL] {fname}: {label}")
        if file_ok:
            print(f"  [OK]   {fname}")
    except FileNotFoundError:
        print(f"  [MISS] {fname}")
        all_ok = False

# Cek debug files dipindah
dev_files = os.listdir("dev") if os.path.isdir("dev") else []
debug_count = len([f for f in dev_files if "debug_" in f])
debug_ok = debug_count > 0
if debug_ok:
    print(f"  [OK]   dev/ — {debug_count} debug files dipindah")
else:
    print("  [FAIL] dev/ — debug files tidak ditemukan")
    all_ok = False

print()
print("=" * 50)
if all_ok:
    print("SEMUA CHECKS PASSED!")
else:
    print("Ada item yang gagal (lihat [FAIL])")
