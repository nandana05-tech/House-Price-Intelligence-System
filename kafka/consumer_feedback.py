"""
Feedback Consumer — persists Kafka feedback events to PostgreSQL via Prisma.
Also checks the retraining threshold after each batch.

Run: python -m kafka.consumer_feedback
"""
from __future__ import annotations

import asyncio
import json
import os

from aiokafka import AIOKafkaConsumer
from dotenv import load_dotenv
from prisma import Prisma

load_dotenv()

from kafka.topics import FEEDBACK_EVENTS


async def main() -> None:
    db = Prisma()
    await db.connect()

    consumer = AIOKafkaConsumer(
        FEEDBACK_EVENTS,
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        group_id=os.getenv("KAFKA_GROUP_ID_FEEDBACK", "hpi-feedback-pod"),
        value_deserializer=lambda b: json.loads(b.decode()),
        auto_offset_reset="earliest",
    )

    await consumer.start()
    print(f"[FeedbackConsumer] Listening on topic: {FEEDBACK_EVENTS}")
    threshold = int(os.getenv("RETRAIN_FEEDBACK_THRESHOLD", "100"))

    try:
        async for msg in consumer:
            event = msg.value

            # Persist to PostgreSQL
            await db.feedback.create(
                data={
                    "predictionId": event.get("prediction_id"),
                    "kamarTidur": int(event["kamar_tidur"]),
                    "kamarMandi": int(event["kamar_mandi"]),
                    "garasi": int(event["garasi"]),
                    "luasTanah": float(event["luas_tanah"]),
                    "luasBangunan": float(event["luas_bangunan"]),
                    "lokasi": event["lokasi"],
                    "hargaPrediksi": float(event["harga_prediksi"]),
                    "hargaAsli": float(event["harga_asli"]),
                    "selisihPersen": float(event.get("selisih_persen", 0)),
                    "sumber": event.get("sumber", "user_feedback"),
                }
            )
            print(f"[FeedbackConsumer] Saved feedback — lokasi={event['lokasi']} selisih={event.get('selisih_persen', '?')}%")

            # Check retrain threshold
            unprocessed_count = await db.feedback.count(where={"processed": False})
            if unprocessed_count >= threshold:
                print(f"[FeedbackConsumer] Threshold reached ({unprocessed_count} feedbacks). Triggering retrain...")
                from pipelines.retrain_trigger import trigger_retrain
                asyncio.create_task(trigger_retrain(db))

    finally:
        await consumer.stop()
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
