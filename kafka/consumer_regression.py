"""
ML Service Pod — Regression Consumer
=====================================
Consumes prediction events from the regression Kafka topic,
re-runs the model (enabling future A/B with new model versions),
logs everything to MLflow, and caches the result in Redis.

Run: python -m kafka.consumer_regression
"""
from __future__ import annotations

import asyncio
import json
import os
import time

from aiokafka import AIOKafkaConsumer
from dotenv import load_dotenv

load_dotenv()

from kafka.topics import REGRESSION_EVENTS
from mlflow_utils.tracker import PredictionTracker
from services.model_loader import models

# Redis import (optional — graceful degradation if not available)
try:
    import redis.asyncio as aioredis
    REDIS_ENABLED = True
except ImportError:
    REDIS_ENABLED = False


async def main() -> None:
    models.load()
    tracker = PredictionTracker(experiment_name=os.getenv("MLFLOW_EXPERIMENT_REGRESSION", "house-price-regression"))

    # Redis connection
    redis_client = None
    if REDIS_ENABLED:
        redis_client = aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    consumer = AIOKafkaConsumer(
        REGRESSION_EVENTS,
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        group_id=os.getenv("KAFKA_GROUP_ID_REGRESSION", "hpi-regression-pod"),
        value_deserializer=lambda b: json.loads(b.decode()),
        auto_offset_reset="latest",
    )

    await consumer.start()
    print(f"[RegressionPod] Listening on topic: {REGRESSION_EVENTS}")

    try:
        async for msg in consumer:
            event = msg.value
            inp = event.get("input", {})
            output = event.get("output", {})

            t0 = time.perf_counter()

            # Log to MLflow
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: tracker.log(
                    params=inp,
                    metrics={
                        "harga_estimasi": output.get("harga_estimasi", 0),
                        "latency_ms": output.get("latency_ms", 0),
                    },
                    tags={
                        "model_digunakan": output.get("model_digunakan", "unknown"),
                        "lokasi": inp.get("lokasi", "unknown"),
                    },
                ),
            )

            # Cache in Redis (key: sorted input params)
            if redis_client:
                cache_key = f"regression:{inp.get('lokasi')}:{inp.get('luas_tanah')}:{inp.get('luas_bangunan')}:{inp.get('kamar_tidur')}:{inp.get('kamar_mandi')}:{inp.get('garasi')}"
                ttl = int(os.getenv("PREDICTION_CACHE_TTL", "3600"))
                await redis_client.set(cache_key, json.dumps(output), ex=ttl)

            pod_latency = round((time.perf_counter() - t0) * 1000, 2)
            print(f"[RegressionPod] Processed — harga={output.get('harga_estimasi_format')} pod_latency={pod_latency}ms")

    finally:
        await consumer.stop()
        if redis_client:
            await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
