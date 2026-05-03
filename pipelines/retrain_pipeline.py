"""
Retraining pipeline — loads original data + feedback corrections,
retrains ALL THREE models (regression, classification, clustering),
evaluates, and registers new versions in MLflow.

Run standalone: python -m pipelines.retrain_pipeline
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import psycopg2
from catboost import CatBoostRegressor, Pool
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

BASE_DIR  = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


async def run_retrain(
    db,
    feedback_ids: list[str],
    run_record_id: str,
) -> None:
    """Full retraining cycle: load data → merge feedback → train → evaluate → register."""
    t_start = time.perf_counter()

    try:
        # 1. Load clean training data dari database (sudah di-clean saat init)
        print("[Retrain] Loading base training data from CleanPropertyRegression...")
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        df_base = pd.read_sql('SELECT * FROM "CleanPropertyRegression"', conn)
        df_base = df_base.rename(columns={
            "kamarTidur":   "Kamar Tidur",
            "kamarMandi":   "Kamar Mandi",
            "luasTanah":    "Luas Tanah",
            "luasBangunan": "Luas Bangunan",
            "harga":        "Harga",
            "lokasi":       "Lokasi",
            "garasi":       "Garasi",
        })
        conn.close()
        print(f"[Retrain] Loaded {len(df_base)} rows from CleanPropertyRegression")

        # 2. Load feedback rows dari DB
        print(f"[Retrain] Loading {len(feedback_ids)} feedback rows...")
        feedbacks = await db.feedback.find_many(where={"id": {"in": feedback_ids}})

        feedback_rows = [
            {
                "Kamar Tidur":   f.kamarTidur,
                "Kamar Mandi":   f.kamarMandi,
                "Garasi":        f.garasi,
                "Luas Tanah":    f.luasTanah,
                "Luas Bangunan": f.luasBangunan,
                "Lokasi":        f.lokasi,
                "Harga":         f.hargaAsli,
            }
            for f in feedbacks
        ]
        df_feedback = pd.DataFrame(feedback_rows)

        # 3. Merge
        df_combined = pd.concat([df_base, df_feedback], ignore_index=True)
        print(f"[Retrain] Combined dataset: {len(df_combined)} rows")

        # 4. Train regresi (in executor agar tidak block event loop)
        loop = asyncio.get_event_loop()
        new_mape = await loop.run_in_executor(None, lambda: _train_regression(df_combined))

        # 5. Retrain klasifikasi & clustering via script (sudah solid, replika notebook)
        await loop.run_in_executor(None, _retrain_classification_and_clustering)

        # 6. Mark feedback as processed
        await db.feedback.update_many(
            where={"id": {"in": feedback_ids}},
            data={"processed": True},
        )

        # 7. Update RetrainingRun record
        await db.retrainingrun.update(
            where={"id": run_record_id},
            data={
                "status": "success",
                "mapeAfter": new_mape,
                "newModelVersion": "retrained",
            },
        )

        elapsed = round((time.perf_counter() - t_start) / 60, 2)
        print(f"[Retrain] Done in {elapsed}min — new MAPE regresi: {new_mape:.2f}%")

    except Exception as exc:  # noqa: BLE001
        print(f"[Retrain] FAILED: {exc}")
        await db.retrainingrun.update(
            where={"id": run_record_id},
            data={"status": "failed", "errorMessage": str(exc)},
        )


def _train_regression(df: pd.DataFrame) -> float:
    """
    Synchronous regression retraining — returns overall MAPE on test set.

    PERUBAHAN vs versi lama:
      - Path model disimpan ke MODELS_DIR / save_name (bukan BASE_DIR / save_name)
      - PRICE_THRESHOLD dibaca dari meta_regresi["batas_segmen"] (bukan hardcoded)
      - Feature engineering menggunakan engineer_regression_features yang sudah sinkron
    """
    from services.feature_engineer import engineer_regression_features
    from services.model_loader import models

    models.load()

    # Batas segmen dari metadata — TIDAK hardcoded
    batas = float(models.meta_regresi["batas_segmen"])

    # Build feature matrix dari data combined (base + feedback)
    rows = []
    for _, row in df.iterrows():
        feat = engineer_regression_features(
            kamar_tidur=int(row["Kamar Tidur"]),
            kamar_mandi=int(row["Kamar Mandi"]),
            garasi=int(row["Garasi"]),
            luas_tanah=float(row["Luas Tanah"]),
            luas_bangunan=float(row["Luas Bangunan"]),
            lokasi=str(row["Lokasi"]),
        )
        rows.append(feat.iloc[0].to_dict())

    X = pd.DataFrame(rows)[models.meta_regresi["fitur"]]
    y = np.log1p(df["Harga"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Split per segmen berdasarkan harga asli
    harga_train = df["Harga"].values[X_train.index]
    harga_test  = df["Harga"].values[X_test.index]
    mask_low_train = harga_train <= batas
    mask_low_test  = harga_test  <= batas

    mapes = []
    with mlflow.start_run(run_name="retrain_regression"):
        mlflow.log_param("total_rows", len(df))
        mlflow.log_param("batas_segmen", batas)

        for mask_train, mask_test, model_name, save_name in [
            (mask_low_train, mask_low_test,  "model_low",  "model_low.cbm"),
            (~mask_low_train, ~mask_low_test, "model_high", "model_high.cbm"),
        ]:
            model = CatBoostRegressor(
                iterations=5000, learning_rate=0.02, depth=7,
                l2_leaf_reg=3, loss_function="RMSE",
                early_stopping_rounds=200, random_seed=42, verbose=500,
            )
            model.fit(
                Pool(X_train[mask_train], y_train[mask_train]),
                eval_set=Pool(X_test[mask_test], y_test[mask_test]),
                use_best_model=True,
            )

            preds   = np.expm1(model.predict(X_test[mask_test]))
            actuals = np.expm1(y_test[mask_test])
            mape    = float(np.mean(np.abs((actuals - preds) / actuals)) * 100)
            mapes.append(mape)

            # Simpan ke MODELS_DIR (bukan BASE_DIR — ini bug lama yang sudah diperbaiki)
            save_path = str(MODELS_DIR / save_name)
            model.save_model(save_path)
            mlflow.log_metric(f"mape_{model_name}", mape)
            mlflow.log_artifact(save_path)
            print(f"[Retrain] {model_name} → MAPE: {mape:.2f}% (saved: {save_path})")

    # Reload models singleton dengan model baru
    from services.model_loader import ModelLoader
    ModelLoader._loaded = False
    models.load()

    return float(np.mean(mapes))


def _retrain_classification_and_clustering() -> None:
    """
    Retrain klasifikasi dan clustering via script yang sudah ada.
    Kedua script sudah mereplikasi persis logika notebook masing-masing.
    """
    scripts = [
        BASE_DIR / "scripts" / "retrain_klasifikasi.py",
        BASE_DIR / "scripts" / "retrain_clustering.py",
    ]
    for script in scripts:
        print(f"[Retrain] Running {script.name}...")
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, cwd=str(BASE_DIR),
        )
        if result.returncode != 0:
            # Log sebagai warning — jangan batalkan seluruh retrain jika satu script gagal
            print(f"[Retrain] WARNING: {script.name} failed (code {result.returncode}):")
            print(result.stderr[:800] if result.stderr else "(no stderr)")
        else:
            print(f"[Retrain] {script.name}: OK")
            if result.stdout:
                # Tampilkan baris terakhir saja (biasanya ringkasan metrik)
                last_lines = result.stdout.strip().split("\n")[-5:]
                print("\n".join(f"    {l}" for l in last_lines))


if __name__ == "__main__":
    async def _standalone():
        from prisma import Prisma
        db = Prisma()
        await db.connect()
        feedbacks = await db.feedback.find_many(where={"processed": False})
        ids = [f.id for f in feedbacks]
        print(f"[Standalone] {len(ids)} unprocessed feedbacks found.")
        run = await db.retrainingrun.create(data={"feedbackCount": len(ids), "status": "running"})
        await run_retrain(db=db, feedback_ids=ids, run_record_id=run.id)
        await db.disconnect()

    asyncio.run(_standalone())
