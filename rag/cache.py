"""
Cache RAG context in Redis to avoid redundant embedding + SQL calls.
TTL: 1 hour per query hash.
"""
from __future__ import annotations
import hashlib
import json
import os
import redis

_client: redis.Redis | None = None
RAG_CACHE_TTL = 3600  # 1 hour


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    return _client


def _cache_key(query: str, prediction: dict | None) -> str:
    pred_hash = hashlib.md5(
        json.dumps(prediction or {}, sort_keys=True).encode()
    ).hexdigest()
    return f"rag:{hashlib.md5(f'{query}:{pred_hash}'.encode()).hexdigest()}"


def get_cached_context(query: str, prediction: dict | None) -> str | None:
    try:
        cached = _get_client().get(_cache_key(query, prediction))
        return cached.decode() if cached else None
    except Exception:
        return None


def set_cached_context(query: str, prediction: dict | None, context: str) -> None:
    try:
        _get_client().setex(_cache_key(query, prediction), RAG_CACHE_TTL, context)
    except Exception:
        pass  # cache failure is non-critical