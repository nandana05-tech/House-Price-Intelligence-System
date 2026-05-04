"""
FastAPI application — mounts the feedback router and serves health endpoints.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.feedback_endpoint import router as feedback_router
from api.predict_endpoint import router as predict_router
from api.chat_endpoint import router as chat_router
from api.analytics_endpoint import router as analytics_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────
    from prisma import Prisma
    from services.model_loader import models
    try:
        models.load()
        print("[API] ML models loaded successfully")
    except Exception as e:
        # Do not crash — the `/health` endpoint must continue responding so the Docker healthcheck does not fail.
        # The model will be reloaded on the first incoming request via the model_loader.
        print(f"[API] WARNING: ML model loading failed at startup: {e}")

    app.state.db = Prisma()
    await app.state.db.connect()
    print("[API] Database connected")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    await app.state.db.disconnect()
    print("[API] Database disconnected")



app = FastAPI(
    title="House Price Intelligence API",
    description="Feedback collection and retraining trigger endpoint for the HPI system.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(feedback_router, prefix="/api/v1")
app.include_router(predict_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "hpi-api"}
