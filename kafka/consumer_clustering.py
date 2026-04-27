"""
ML Service Pod — Clustering Consumer
=======================================
Consumes clustering events, logs to MLflow, caches in Redis.

Run: python -m kafka.consumer_clustering
"""
from __future__ import annotations

import asyncio
import json
import os
import time

from aiokafka import AIOKafkaConsumer
from dotenv import load_dotenv

load_dotenv()

from kafka.topics import CLUSTERING_EVENTS
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
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_CLUSTERING", "house-price-clustering")
    )

    redis_client = None
    if REDIS_ENABLED:
        redis_client = aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    consumer = AIOKafkaConsumer(
        CLUSTERING_EVENTS,
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        group_id=os.getenv("KAFKA_GROUP_ID_CLUSTERING", "hpi-clustering-pod"),
        value_deserializer=lambda b: json.loads(b.decode()),
        auto_offset_reset="latest",
    )

    await consumer.start()
    print(f"[ClusteringPod] Listening on topic: {CLUSTERING_EVENTS}")

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
                    metrics={
                        "cluster_id": float(output.get("cluster_id", -1)),
                        "latency_ms": output.get("latency_ms", 0),
                    },
                    tags={
                        "cluster_label": output.get("cluster_label", "unknown"),
                        "lokasi": inp.get("lokasi", "unknown"),
                    },
                ),
            )

            if redis_client:
                cache_key = f"clustering:{inp.get('lokasi')}:{inp.get('luas_tanah')}:{inp.get('luas_bangunan')}:{inp.get('kamar_tidur')}:{inp.get('harga')}"
                ttl = int(os.getenv("PREDICTION_CACHE_TTL", "3600"))
                await redis_client.set(cache_key, json.dumps(output), ex=ttl)

            pod_latency = round((time.perf_counter() - t0) * 1000, 2)
            print(f"[ClusteringPod] Processed — cluster={output.get('cluster_label')} pod_latency={pod_latency}ms")

    finally:
        await consumer.stop()
        if redis_client:
            await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
