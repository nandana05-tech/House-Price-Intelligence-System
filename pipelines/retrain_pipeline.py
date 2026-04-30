"""
Retraining pipeline — loads original data + feedback corrections,
retrains CatBoost models, evaluates, and registers new versions in MLflow.

Run standalone: python -m pipelines.retrain_pipeline
"""
from __future__ import annotations

import asyncio
import os
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

BASE_DIR = Path(__file__).parent.parent
PRICE_THRESHOLD = 1_200_000_000.0

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
        # 1. Load original training data from database
        print("[Retrain] Loading base training data from CleanPropertyRegression...")
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        df_base = pd.read_sql('SELECT * FROM "CleanPropertyRegression"', conn)
        df_base = df_base.rename(columns={
            "kamarTidur": "Kamar Tidur",
            "kamarMandi": "Kamar Mandi",
            "luasTanah": "Luas Tanah",
            "luasBangunan": "Luas Bangunan",
            "harga": "Harga",
            "lokasi": "Lokasi",
            "garasi": "Garasi",
        })
        conn.close()
        print(f"[Retrain] Loaded {len(df_base)} rows from CleanPropertyRegression")

        # 2. Load feedback rows from DB
        print(f"[Retrain] Loading {len(feedback_ids)} feedback rows...")
        feedbacks = await db.feedback.find_many(where={"id": {"in": feedback_ids}})

        feedback_rows = [
            {
                "Kamar Tidur": f.kamarTidur,
                "Kamar Mandi": f.kamarMandi,
                "Garasi": f.garasi,
                "Luas Tanah": f.luasTanah,
                "Luas Bangunan": f.luasBangunan,
                "Lokasi": f.lokasi,
                "Harga": f.hargaAsli,
            }
            for f in feedbacks
        ]
        df_feedback = pd.DataFrame(feedback_rows)

        # 3. Merge and prepare
        df_combined = pd.concat([df_base, df_feedback], ignore_index=True)
        print(f"[Retrain] Combined dataset: {len(df_combined)} rows")

        # 4. Feature engineering (run synchronously in executor)
        loop = asyncio.get_event_loop()
        new_mape = await loop.run_in_executor(None, lambda: _train_and_evaluate(df_combined))

        # 5. Mark feedback as processed
        await db.feedback.update_many(
            where={"id": {"in": feedback_ids}},
            data={"processed": True},
        )

        # 6. Update RetrainingRun record
        await db.retrainingrun.update(
            where={"id": run_record_id},
            data={
                "status": "success",
                "mapeAfter": new_mape,
                "newModelVersion": "retrained",
            },
        )

        elapsed = round((time.perf_counter() - t_start) / 60, 2)
        print(f"[Retrain] Done in {elapsed}min — new MAPE: {new_mape:.2f}%")

    except Exception as exc:  # noqa: BLE001
        print(f"[Retrain] FAILED: {exc}")
        await db.retrainingrun.update(
            where={"id": run_record_id},
            data={"status": "failed", "errorMessage": str(exc)},
        )


def _train_and_evaluate(df: pd.DataFrame) -> float:
    """Synchronous training routine — returns overall MAPE on test set."""
    from services.feature_engineer import engineer_regression_features
    from services.model_loader import models

    models.load()

    # Build regression features with the same logic used in production inference
    X = pd.DataFrame(
        [
            engineer_regression_features(
                kamar_tidur=int(row["Kamar Tidur"]),
                kamar_mandi=int(row["Kamar Mandi"]),
                garasi=int(row["Garasi"]),
                luas_tanah=float(row["Luas Tanah"]),
                luas_bangunan=float(row["Luas Bangunan"]),
                lokasi=str(row["Lokasi"]),
            ).iloc[0].to_dict()
            for _, row in df.iterrows()
        ]
    )[models.meta_regresi["fitur"]]
    y = np.log1p(df["Harga"])  # Train on log price

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split by threshold using actual prices
    harga_asli = df["Harga"].iloc[y_train.index]
    mask_low_train = harga_asli <= PRICE_THRESHOLD

    harga_asli_test = df["Harga"].iloc[y_test.index]
    mask_low_test = harga_asli_test <= PRICE_THRESHOLD

    mapes = []
    with mlflow.start_run(run_name="retrain_regression"):
        mlflow.log_param("total_rows", len(df))
        mlflow.log_param("feedback_rows", max(len(df) - len(X_train) - len(X_test), 0))

        for mask_train, mask_test, model_name, save_name in [
            (mask_low_train, mask_low_test, "model_low", "model_low.cbm"),
            (~mask_low_train, ~mask_low_test, "model_high", "model_high.cbm"),
        ]:
            model = CatBoostRegressor(
                iterations=1000, learning_rate=0.05, depth=7,
                loss_function="RMSE", eval_metric="MAPE",
                verbose=100, early_stopping_rounds=50,
            )
            pool_train = Pool(X_train[mask_train], y_train[mask_train])
            pool_val = Pool(X_test[mask_test], y_test[mask_test])

            model.fit(pool_train, eval_set=pool_val)
            preds = model.predict(X_test[mask_test])
            
            actuals = np.expm1(y_test[mask_test].values)
            preds_exp = np.expm1(preds)
            mape = float(np.mean(np.abs((actuals - preds_exp) / actuals)) * 100)
            
            mapes.append(mape)

            model.save_model(str(BASE_DIR / save_name))
            mlflow.log_metric(f"mape_{model_name}", mape)
            mlflow.log_artifact(str(BASE_DIR / save_name))
            print(f"[Retrain] {model_name} saved — MAPE: {mape:.2f}%")

    # Reload updated models into singleton
    from services.model_loader import ModelLoader
    ModelLoader._loaded = False
    models.load()

    return float(np.mean(mapes))


if __name__ == "__main__":
    async def _standalone():
        from prisma import Prisma
        db = Prisma()
        await db.connect()
        # Get all unprocessed feedback IDs for a manual trigger
        feedbacks = await db.feedback.find_many(where={"processed": False})
        ids = [f.id for f in feedbacks]
        run = await db.retrainingrun.create(data={"feedbackCount": len(ids), "status": "running"})
        await run_retrain(db=db, feedback_ids=ids, run_record_id=run.id)
        await db.disconnect()

    asyncio.run(_standalone())
