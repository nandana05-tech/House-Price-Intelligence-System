"""
ML Service Pod — Classification Consumer
==========================================
Consumes classification events, logs to MLflow, caches in Redis.

Run: python -m kafka.consumer_classification
"""
from __future__ import annotations

import asyncio
import json
import os
import time

from aiokafka import AIOKafkaConsumer
from dotenv import load_dotenv

load_dotenv()

from kafka.topics import CLASSIFICATION_EVENTS
from mlflow_utils.tracker import PredictionTracker
from services.model_loader import models

try:
    import redis.asyncio as aioredis
    REDIS_ENABLED = True
except ImportError:
    REDIS_ENABLED = False


async def main() -> None:
    models.load()
    tracker = PredictionTracker(
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_CLASSIFICATION", "house-price-classification")
    )

    redis_client = None
    if REDIS_ENABLED:
        redis_client = aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    consumer = AIOKafkaConsumer(
        CLASSIFICATION_EVENTS,
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        group_id=os.getenv("KAFKA_GROUP_ID_CLASSIFICATION", "hpi-classification-pod"),
        value_deserializer=lambda b: json.loads(b.decode()),
        auto_offset_reset="latest",
    )

    await consumer.start()
    print(f"[ClassificationPod] Listening on topic: {CLASSIFICATION_EVENTS}")

    try:
        async for msg in consumer:
            event = msg.value
            inp = event.get("input", {})
            output = event.get("output", {})

            t0 = time.perf_counter()

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: tracker.log(
                    params=inp,
                    metrics={"latency_ms": output.get("latency_ms", 0)},
                    tags={
                        "kelas_label": output.get("kelas_label", "unknown"),
                        "kelas_id": str(output.get("kelas_id", -1)),
                        "lokasi": inp.get("lokasi", "unknown"),
                    },
                ),
            )

            if redis_client:
                cache_key = f"classification:{inp.get('lokasi')}:{inp.get('luas_tanah')}:{inp.get('luas_bangunan')}:{inp.get('kamar_tidur')}:{inp.get('harga')}"
                ttl = int(os.getenv("PREDICTION_CACHE_TTL", "3600"))
                await redis_client.set(cache_key, json.dumps(output), ex=ttl)

            pod_latency = round((time.perf_counter() - t0) * 1000, 2)
            print(f"[ClassificationPod] Processed — kelas={output.get('kelas_label')} pod_latency={pod_latency}ms")

    finally:
        await consumer.stop()
        if redis_client:
            await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
