"""
Retraining trigger — checks accumulated unprocessed feedback
and kicks off the retraining pipeline when the threshold is met.
Can also be run as a standalone scheduled job (e.g. cron).

Run standalone: python -m pipelines.retrain_trigger
"""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

THRESHOLD = int(os.getenv("RETRAIN_FEEDBACK_THRESHOLD", "100"))


async def trigger_retrain(db=None) -> None:
    """
    Check if enough unprocessed feedback exists, then call the retrain pipeline.
    `db` is an optional already-connected Prisma instance (avoids double-connect).
    """
    from prisma import Prisma

    should_disconnect = False
    if db is None:
        db = Prisma()
        await db.connect()
        should_disconnect = True

    try:
        count = await db.feedback.count(where={"processed": False})
        print(f"[RetainTrigger] Unprocessed feedback: {count} / threshold: {THRESHOLD}")

        if count < THRESHOLD:
            print("[RetainTrigger] Threshold not reached — skipping retrain.")
            return

        # Lock rows by marking them in-progress
        feedbacks = await db.feedback.find_many(
            where={"processed": False},
            take=count,
        )
        ids = [f.id for f in feedbacks]

        # Create a RetrainingRun record
        run_record = await db.retrainingrun.create(
            data={"feedbackCount": len(ids), "status": "running"}
        )

        print(f"[RetainTrigger] Spawning retrain pipeline (run_id={run_record.id}, feedback_count={len(ids)})")

        from pipelines.retrain_pipeline import run_retrain
        asyncio.create_task(run_retrain(db=db, feedback_ids=ids, run_record_id=run_record.id))

    finally:
        if should_disconnect:
            await db.disconnect()


if __name__ == "__main__":
    asyncio.run(trigger_retrain())
