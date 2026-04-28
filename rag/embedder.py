"""
Embedding generation using LangChain OpenAIEmbeddings.
"""
from __future__ import annotations
import os
from langchain_openai import OpenAIEmbeddings

_embeddings: OpenAIEmbeddings | None = None


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _embeddings


def embed_text(text: str) -> list[float]:
    return get_embeddings().embed_query(text)