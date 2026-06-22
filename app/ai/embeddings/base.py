"""
app/ai/embeddings/base.py

Abstract base class every embedding provider must implement.

This is a SEPARATE abstraction from app.ai.base.AIProvider, deliberately.
Chat providers are selected per-agent, per-turn, driven by chat config
(agents.llm_provider / llm_model). Embedding providers are selected by a
completely different axis depending on the caller:

  - Semantic router (app/routing/embedder.py): no persistence constraint.
    Swapping models just invalidates a Redis cache via CACHE_VERSION.
    Safe to change freely.

  - RAG indexing/retrieval (app/rag/embedder.py): embeddings are PERSISTED
    in document_chunks.embedding (vector(1536) today). The provider/model
    used at index time MUST match the provider/model used at query time —
    cosine distance between two different embedding spaces is meaningless,
    not just lower-quality. Changing this for an agent requires a re-index,
    not just a config flip. That migration story is out of scope here;
    this interface only makes the provider swappable, it doesn't yet make
    the swap safe to do live.

Not every provider necessarily implements both embed_query and embed_chunks
meaningfully fast/cheap — but both are part of the contract since every
embedding model can technically do both (it's the same API call shape,
just batched vs single).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single piece of text — a user query or router input.

        Used by:
          - app/rag/retriever.py at RAG query time
          - app/routing/embedder.py for semantic routing similarity checks
        """

    @abstractmethod
    async def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-embed multiple texts in one call, preserving input order.

        Used by:
          - app/rag/indexer.py at document index time

        Empty input should return [] without making a network call.
        """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """
        Output vector width for this provider/model combination.

        Needed wherever a caller must know vector width ahead of time —
        e.g. validating against a pgvector column width, or namespacing a
        cache key so embeddings from different models never collide.
        Not enforced inside the provider itself; callers are responsible
        for checking compatibility before persisting or comparing vectors.
        """