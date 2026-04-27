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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    from prisma import Prisma
    from services.model_loader import models
    
    models.load()  # Load ML models into memory
    
    app.state.db = Prisma()
    await app.state.db.connect()
    yield
    # Shutdown
    await app.state.db.disconnect()


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


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "hpi-api"}
