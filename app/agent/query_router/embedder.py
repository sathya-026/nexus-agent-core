"""
app/routing/embedder.py
-----------------------
Embedding infrastructure for the semantic router.

REPLACED: local MiniLM (all-MiniLM-L6-v2) -> Gemini embeddings
(app.ai.embeddings.gemini.GeminiEmbeddingProvider), called via
get_embedding_provider("gemini").

Why this swap was made:
  MiniLM was causing issues in production (see CACHE_VERSION note below
  for the operational side of this). Gemini's embedding endpoint removes
  the local-model footprint entirely — no ~120 MB process-resident model,
  no singleton-loading ceremony, no warm-up penalty. The cost is an async
  HTTP round-trip per call instead of an in-process forward pass, which is
  an acceptable trade for the router (low QPS relative to chat, no hard
  latency budget comparable to token streaming).

  This swap is SAFE specifically because the router has no persistence
  constraint — see app/ai/embeddings/base.py for why the same swap is not
  safe to make for RAG indexing/retrieval without a re-index strategy.

CACHE_VERSION
-------------
Ties the Redis route cache to the embedding model. Any change to
EMBEDDING_MODEL_NAME (swapping the model) automatically invalidates all
cached embeddings on the next read — no manual Redis flush needed.

Bump the "v1" prefix for breaking changes that aren't captured by the
model name (e.g. switching normalisation strategy or dimension count).
Changing EMBEDDING_DIMENSIONS below is exactly such a change — bump the
prefix if you adjust it after this has run in production once, since
cached vectors at the old width are incomparable to new ones.

What did NOT change
--------------------
cosine_similarities() and blend_embeddings() are pure vector math with no
relationship to which model produced the vectors — they work identically
on MiniLM, OpenAI, or Gemini embeddings. Left untouched.
"""

from __future__ import annotations

import logging

import numpy as np

from app.ai.embeddings.factory import get_embedding_provider

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME   = "gemini-embedding-001"
EMBEDDING_DIMENSIONS   = 768
CACHE_VERSION          = f"v1:{EMBEDDING_MODEL_NAME}:{EMBEDDING_DIMENSIONS}"

_provider = get_embedding_provider("gemini")


# ── Public functions ──────────────────────────────────────────────────────────

async def embed_async(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using Gemini embeddings.
    Returns one vector per input string, dimensions per EMBEDDING_DIMENSIONS.

    Empty input returns [] without making a network call.

    This replaces the old synchronous embed() + asyncio.to_thread() pattern
    MiniLM required — Gemini's call is natively async, so there's no thread
    offload needed.
    """
    if not texts:
        return []
    return await _provider.embed_chunks(texts)


async def embed_query_async(text: str) -> list[float]:
    """
    Embed a single query string. Uses Gemini's RETRIEVAL_QUERY task type
    under the hood via GeminiEmbeddingProvider.embed_query().
    """
    return await _provider.embed_query(text)


def cosine_similarities(
    matrix: list[list[float]],
    query: list[float],
) -> np.ndarray:
    mat = np.array(matrix)
    vec = np.array(query)
    
    # ── DEFENSIVE CHECK 1: Ensure the matrix has exactly 2 dimensions ──
    if mat.ndim == 1:
        # If it accidentally flattened into a 1D array, restore its row dimension
        mat = np.expand_dims(mat, axis=0)
    elif mat.ndim == 0 or mat.size == 0:
        # If the matrix is empty, return an empty array with 0 similarity
        return np.array([0.0])

    # ── DEFENSIVE CHECK 2: Ensure the query vector is exactly 1D ──
    if vec.ndim == 0:
        # If vec is a 0D scalar/object, it means query isn't a clean list of floats
        raise ValueError(f"Query vector is 0-D. Check what embed_query returned. Type: {type(query)}")

    # Matrix multiplication execution
    norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(vec)
    norms = np.where(norms == 0, 1e-10, norms)
    
    return (mat @ vec) / norms


def blend_embeddings(
    embeddings: list[list[float]],
    weights: list[float] | None = None,
) -> list[float]:
    """
    Compute weighted average of multiple embeddings.

    Returns a single embedding as the weighted combination.
    Result is automatically normalised to unit length.

    Unchanged from the MiniLM version — this is pure vector math,
    independent of which model produced the input embeddings.

    Args:
        embeddings: List of embedding vectors (each is list[float])
        weights:    Optional weights for each embedding. If None, uses uniform weight.
                   Must sum to 1.0 or will be normalised.

    Returns:
        Normalised combined embedding vector.

    Example:
        current_embedding = [0.1, 0.2, ...]
        history_embedding = [0.3, 0.1, ...]
        blended = blend_embeddings(
            [current_embedding, history_embedding],
            [0.7, 0.3]
        )
    """
    if not embeddings:
        raise ValueError("embeddings list cannot be empty")

    arr = np.array(embeddings)  # shape: (n_embeddings, embedding_dim)

    if weights is None:
        weights = [1.0 / len(embeddings)] * len(embeddings)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # normalise

    combined = (arr.T @ weights)  # matrix mult: (dim, n) @ (n,) = (dim,)

    norm = np.linalg.norm(combined)
    if norm < 1e-10:
        return combined.tolist()

    return (combined / norm).tolist()