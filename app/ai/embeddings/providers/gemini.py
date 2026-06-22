"""
app/ai/embeddings/gemini.py

Gemini implementation of EmbeddingProvider, using google-genai's
embed_content (async via client.aio.models.embed_content).

model: gemini-embedding-001
  - Native output is 3072 dimensions.
  - Supports output_dimensionality truncation (Matryoshka Representation
    Learning) for cheaper storage/comparison when full precision isn't
    needed. Defaulted to 768 here — a reasonable middle ground for the
    semantic router's similarity checks, which have no persistence
    constraint and no fixed schema width to match.
  - Supports task_type, which meaningfully improves retrieval quality by
    using asymmetric query/document embeddings rather than one embedding
    space for both. embed_query uses RETRIEVAL_QUERY; embed_chunks uses
    RETRIEVAL_DOCUMENT.

This provider is intended for the semantic router today (replacing
MiniLM). It is NOT wired into app/rag/embedder.py yet — RAG embeddings
are persisted in a fixed-width pgvector column, so switching that call
site requires a re-index strategy, not just a provider swap.
"""

from __future__ import annotations

from google import genai
from google.genai import types as gtypes

from app.ai.embeddings.base import EmbeddingProvider
from app.config import settings

MODEL = "gemini-embedding-001"
DIMENSIONS = 768


class GeminiEmbeddingProvider(EmbeddingProvider):

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_query(self, text: str) -> list[float]:
        """
        Single text -> single vector, using RETRIEVAL_QUERY task type.
        Used by the semantic router and (in future) RAG query-time retrieval.
        """
        result = await self._client.aio.models.embed_content(
            model=MODEL,
            contents=text,
            config=gtypes.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self._dimensions,
            ),
        )
        return list(result.embeddings[0].values)

    async def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-embed using RETRIEVAL_DOCUMENT task type, preserving input order.
        Not used by the router today, but part of the contract for future
        RAG-side usage.
        """
        if not texts:
            return []

        result = await self._client.aio.models.embed_content(
            model=MODEL,
            contents=texts,
            config=gtypes.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self._dimensions,
            ),
        )
        return [list(e.values) for e in result.embeddings]