"""
app/routing/embedder.py
-----------------------
Local embedding infrastructure for the semantic router.

The MiniLM model (all-MiniLM-L6-v2) is loaded once at process startup
and reused for the lifetime of the process:
  - ~120 MB RAM
  - ~100x faster than an OpenAI embedding API round-trip
  - Zero per-call cost

CACHE_VERSION
-------------
Ties the Redis route cache to the embedding model. Any change to
MINILM_MODEL_NAME (swapping the model) automatically invalidates all
cached embeddings on the next read — no manual Redis flush needed.

Bump the "v1" prefix for breaking changes that aren't captured by the
model name (e.g. switching normalisation strategy or dimension count).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MINILM_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_VERSION     = f"v1:{MINILM_MODEL_NAME}"

_minilm_model: Optional[object] = None


# ── Singleton loader ──────────────────────────────────────────────────────────

def _get_model():
    """
    Lazy singleton for the MiniLM model.

    Thread-safe via the GIL. Safe for asyncio — no threads involved.
    Import is deferred so the module loads cleanly on machines that
    don't have sentence-transformers installed (e.g. CI environments
    that only run the NestJS service).
    """
    global _minilm_model
    if _minilm_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading MiniLM model '%s' into memory...", MINILM_MODEL_NAME)
        _minilm_model = SentenceTransformer(MINILM_MODEL_NAME)
        logger.info("MiniLM model loaded.")
    return _minilm_model


# ── Public functions ──────────────────────────────────────────────────────────

def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using the local MiniLM model.
    Returns one unit-normalised float vector per input string.

    Empty input returns [] without touching the model.

    Note: encode() is synchronous and CPU-bound. For high-concurrency
    paths, wrap the call with asyncio.to_thread(embed, texts).
    """
    if not texts:
        return []

    model = _get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # unit vectors → dot product == cosine sim
        show_progress_bar=False,
    )
    return embeddings.tolist()


def cosine_similarities(
    matrix: list[list[float]],
    query: list[float],
) -> np.ndarray:
    """
    Return cosine similarity between each row in matrix and query.

    Because embed() normalises to unit vectors, this reduces to a
    dot product. The explicit norm division is kept for safety — it
    handles any non-unit vectors passed in from tests or future callers.
    """
    mat   = np.array(matrix)
    vec   = np.array(query)
    norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(vec)
    norms = np.where(norms == 0, 1e-10, norms)
    return (mat @ vec) / norms


def warmup_model() -> None:
    """
    Force MiniLM to load and JIT-compile at app startup.

    Calling this from the FastAPI lifespan means the first real request
    doesn't pay the ~1-2 s model-load penalty.

    Example:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            warmup_model()
            yield
    """
    embed(["warmup"])
    logger.info("MiniLM model warm-up complete.")

async def embed_async(texts: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(embed, texts)


def blend_embeddings(
    embeddings: list[list[float]],
    weights: list[float] | None = None,
) -> list[float]:
    """
    Compute weighted average of multiple embeddings.
    
    Returns a single embedding as the weighted combination.
    Result is automatically normalised to unit length.
    
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