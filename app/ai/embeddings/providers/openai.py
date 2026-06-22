"""
app/ai/embeddings/openai.py

OpenAI implementation of EmbeddingProvider.

This is app/rag/embedder.py's existing logic moved behind the new
interface — no behavioral change. embedder.py becomes a thin caller of
this class so app/rag/indexer.py and app/rag/retriever.py don't need to
change their call sites at all.

Preserves exactly what the original spec required:
  - model: text-embedding-3-small, 1536 dimensions
  - batch_size: 100
  - retry via tenacity: RateLimitError / APITimeoutError,
    exponential backoff 2s -> 4s -> 8s -> 16s, max 4 attempts
  - defensive sort by item.index even though OpenAI guarantees order
"""

from __future__ import annotations

from openai import APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.ai.embeddings.base import EmbeddingProvider
from app.config import settings

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
BATCH_SIZE = 100


class OpenAIEmbeddingProvider(EmbeddingProvider):

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    async def embed_query(self, text: str) -> list[float]:
        """Single text -> single vector. Used at RAG retrieval time."""
        vectors = await self._embed_batch([text])
        return vectors[0]

    async def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-embed in groups of BATCH_SIZE, preserving input order across
        batches. Used at document index time.
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            all_vectors.extend(await self._embed_batch(batch))

        return all_vectors

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        stop=stop_after_attempt(4),
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        One OpenAI embeddings.create() call. Retries on rate limit or
        timeout with exponential backoff: 2s -> 4s -> 8s -> 16s, 4 attempts.
        """
        response = await self._client.embeddings.create(
            model=MODEL,
            input=texts,
        )

        # Defensive sort by item.index — OpenAI guarantees order, but the
        # cost of asserting it is one sort call, and silent misordering
        # here would corrupt every chunk's embedding/content pairing.
        sorted_data = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in sorted_data]