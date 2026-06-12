# app/retrieval/retriever.py
"""
Vector retriever — finds the most relevant knowledge base chunks for a query.

Called by the planner before the ReAct loop. The retrieved chunks are injected
into the system prompt as Knowledge Base Context.

Design notes:
    - agent_id filter applied inside the subquery, before vector search.
      This scopes the HNSW scan to one tenant's chunks — not a post-filter.
    - ORDER BY <=> ASC + LIMIT (inner) lets pgvector use the HNSW index.
      Threshold filtering (outer) runs on the small candidate set after.
    - similarity = 1 - cosine_distance. Range: 0.0 (orthogonal) → 1.0 (identical).
      0.3 default threshold discards noise; tune per agent once you have data.
    - Returns [] on RAG miss. The planner logs the miss and the LLM falls back
      to general knowledge. This function never raises on an empty result.
"""

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embedder import embed_query

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TOP_K     = 5
DEFAULT_THRESHOLD = 0.2


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    content:     str
    document_id: str
    chunk_index: int
    similarity:  float   # 0.0 → 1.0, higher is more relevant


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

async def retrieve(
    db:        AsyncSession,
    agent_id:  str,
    query:     str,
    top_k:     int   = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[RetrievedChunk]:
    """
    Embed `query` and return the most similar chunks for this agent.

    Args:
        db:        Async SQLAlchemy session.
        agent_id:  Scopes the search to one agent's document_chunks only.
        query:     Raw user message — embedded with the same model used at
                   index time (text-embedding-3-small). Model consistency is
                   critical: mismatched models produce meaningless similarities.
        top_k:     Candidate pool size passed to HNSW. The actual number of
                   returned chunks may be lower if some fall below threshold.
        threshold: Minimum similarity to include a chunk. Chunks below this
                   are noise — the query is too dissimilar to be useful context.

    Returns:
        List of RetrievedChunk sorted best → worst similarity.
        Empty list = RAG miss. Caller is responsible for logging and fallback.
    """
    query_vector = await embed_query(query)

    # pgvector literal format: "[0.12, -0.34, ...]"
    # Consistent with the ::vector cast used in the indexer bulk insert.
    embedding_str = "[" + ",".join(str(float(v)) for v in query_vector) + "]"
    
    # Debug: Check if any chunks exist for this agent
    count_result = await db.execute(
        text("SELECT COUNT(*) as cnt FROM document_chunks WHERE agent_id = :agent_id"),
        {"agent_id": agent_id}
    )
    total_chunks = count_result.scalar() or 0
    logger.debug(
        "Total chunks indexed for agent %s: %d",
        agent_id, total_chunks
    )
    
    rows = []
    try:
        result = await db.execute(
            text(f"""
                SELECT content, document_id, chunk_index, similarity
                FROM (
                    SELECT
                        content,
                        document_id,
                        chunk_index,
                        1 - (embedding <=> '{embedding_str}'::vector) AS similarity
                    FROM  document_chunks
                    WHERE agent_id = :agent_id
                    ORDER BY embedding <=> '{embedding_str}'::vector ASC
                    LIMIT :top_k
                ) candidates
                WHERE similarity >= :threshold
            """),
            {
                "agent_id":  agent_id,
                "top_k":     top_k,
                "threshold": threshold,
            },
        )
        rows = result.fetchall()
        await db.commit()
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        rows = []
        await db.rollback()

    chunks = [
        RetrievedChunk(
            content=row.content,
            document_id=str(row.document_id),
            chunk_index=row.chunk_index,
            similarity=float(row.similarity),
        )
        for row in rows
    ]

    logger.debug(
        "Retrieved %d/%d chunks for agent %s (threshold=%.2f, top_k=%d)",
        len(chunks), top_k, agent_id, threshold, top_k,
    )
    
    # Debug: Log top 5 candidates with their scores (before threshold filter)
    if not chunks:
        top_candidates = await db.execute(
            text(f"""
                SELECT content, 1 - (embedding <=> '{embedding_str}'::vector) AS similarity
                FROM document_chunks
                WHERE agent_id = :agent_id
                ORDER BY embedding <=> '{embedding_str}'::vector ASC
                LIMIT 5
            """),
            {"agent_id": agent_id},
        )
        top_rows = top_candidates.fetchall()
        for i, row in enumerate(top_rows):
            logger.debug(
                "  [Candidate %d] similarity=%.4f | content_preview=%s...",
                i+1, float(row.similarity), str(row.content)[:80]
            )

    return chunks


def format_context_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """
    Serialise retrieved chunks into the string injected into the system prompt.

    Output shape:
        [Context 1]
        chunk text...

        [Context 2]
        chunk text...

    Numbered labels give the LLM a stable way to cite sources ("as stated in
    Context 2") and make it easier to trace which chunk drove a given answer
    when debugging retrieval quality.
    """
    return "\n\n".join(
        f"[Context {i + 1}]\n{chunk.content}"
        for i, chunk in enumerate(chunks)
    )