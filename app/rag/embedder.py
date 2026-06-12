"""
RAG Step 2 — Embedder

Converts a list of text chunks into embedding vectors by calling
the OpenAI embeddings API.

Key decisions:
  - Batching: we send chunks in groups of 100 (well under the 2048 limit)
    so a single API failure only affects one batch, not the whole document.
  - Retry: transient API errors (rate limits, timeouts) are retried with
    exponential backoff via tenacity before we give up and fail the document.
  - Model: text-embedding-3-small — 1536 dims, cheap ($0.02/million tokens),
    fast. Switch to text-embedding-3-large (3072 dims) if retrieval quality
    matters more than cost.
"""

from app.ai.factory import get_provider
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import RateLimitError, APITimeoutError

from app.common.constants import AIProviderType
from app.config import settings
from app.rag.chunker import Chunk

BATCH_SIZE = 100  # Chunks per API call — safely under the 2048 limit


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
)
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Call the OpenAI embedding API for a batch of texts.

    The @retry decorator handles transient failures:
      - Waits 2s, then 4s, then 8s, then 16s (capped at 30s) between attempts.
      - Only retries on RateLimitError and APITimeoutError — not on bad input.
      - Raises after 4 failed attempts so the indexer can mark the doc as failed.
    """
    provider = get_provider(AIProviderType.OPENAI, settings.chat_model)
    response = await provider._client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        # encoding_format="float" is the default — explicit for clarity.
        # "base64" is faster for large batches but requires extra decoding.
        encoding_format="float",
    )
    # The API returns embeddings in the same order as the input texts.
    # Sort by index just to be safe (the API spec guarantees order, but
    # defensive code is cheaper than a hard-to-reproduce bug).
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


async def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """
    Embed all chunks for a document, returning one vector per chunk.

    Processes chunks in batches of BATCH_SIZE. Within each batch,
    runs concurrently. Batches themselves are sequential to avoid
    overwhelming the rate limit.

    Args:
        chunks: Output of chunker.chunk_text() — ordered list of Chunk objects.

    Returns:
        List of embedding vectors, same length and order as input chunks.
    """
    if not chunks:
        return []

    all_embeddings: list[list[float]] = []

    # Split into batches
    batches = [chunks[i : i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    for batch in batches:
        texts = [chunk.content for chunk in batch]
        batch_embeddings = await _embed_batch(texts)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


async def embed_query(query: str) -> list[float]:
    """
    Embed a single user query for vector search.

    Uses the same model as embed_chunks() — this is critical.
    If the query and chunks were embedded with different models,
    their coordinate spaces don't align and similarity scores are meaningless.

    Args:
        query: The user's raw message text.

    Returns:
        Single embedding vector (list of 1536 floats).
    """
    result = await _embed_batch([query])
    return result[0]
