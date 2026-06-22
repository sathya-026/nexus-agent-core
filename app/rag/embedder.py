"""
app/rag/embedder.py

OpenAI embedding calls for the RAG pipeline — indexing and query-time retrieval.

This module is now a thin caller of app.ai.embeddings.openai.OpenAIEmbeddingProvider.
All retry/batching logic lives there. This file keeps the original public
function names (embed_chunks, embed_query) unchanged so app/rag/indexer.py
and app/rag/retriever.py don't need any changes at their call sites.

RAG embeddings are NOT yet provider-configurable — they're persisted in
document_chunks.embedding (vector(1536)), and switching providers requires
a re-index strategy, not just a config flip. Hardcoded to OpenAI until
that migration is built. See app/ai/embeddings/base.py for the full
reasoning on why this is split from the semantic router's embedder.
"""

from __future__ import annotations

from app.ai.embeddings.factory import get_embedding_provider

_provider = get_embedding_provider("openai")


async def embed_chunks(texts: list[str]) -> list[list[float]]:
    """Used at index time — takes List[str] chunk content, returns List[List[float]]."""
    return await _provider.embed_chunks(texts)


async def embed_query(text: str) -> list[float]:
    """Used at retrieval time — takes a query string, returns List[float]."""
    return await _provider.embed_query(text)