import logging
from typing import Any

from app.rag.chunker import Chunk

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def fetch_indexed_documents(db: AsyncSession, agent_id: str) -> list[Any]:
    try:
        query = """SELECT name, description FROM documents WHERE agent_id = :agent_id AND status = :status"""
        result = await db.execute(
            text(query), {"agent_id": agent_id, "status": "indexed"}
        )
        return result.fetchall()
    except Exception as e:
        logger.exception(f"Error during indexed documents fetch: {str(e)}")

async def insert_chunks(
    db: AsyncSession,
    document_id: str,
    agent_id: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    """
    Insert all chunks + their embeddings into document_chunks in one
    bulk operation.

    Why bulk? Inserting 200 chunks one-by-one would mean 200 round trips
    to Postgres. A single INSERT ... VALUES (...), (...), ... is one round
    trip regardless of row count — 10-50x faster for typical documents.

    We also delete any existing chunks for this document first. This makes
    re-indexing safe — you can call index_document() again after updating
    a file without ending up with duplicate chunks.

    The embedding is cast to the pgvector `vector` type via ::vector.
    pgvector expects the array literal format: '[0.1, 0.2, ...]'
    """
    # Clean up any existing chunks (safe re-indexing)
    await db.execute(
        text("DELETE FROM document_chunks WHERE document_id = :doc_id"),
        {"doc_id": document_id},
    )

    if not chunks:
        return

    # Build the bulk insert.
    # SQLAlchemy's text() with named params handles escaping — safe from injection.
    rows = []
    params: dict[str, Any] = {}

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Convert Python list[float] → pgvector literal string: "[0.1,0.2,...]"
        # vector_literal = "[" + ",".join(str(x) for x in embedding) + "]"
        rows.append(
            f"(:doc_id_{i}, :agent_id_{i}, :content_{i}, CAST(:vector_{i} AS vector), :chunk_index_{i})"
        )
        params[f"doc_id_{i}"] = document_id
        params[f"agent_id_{i}"] = agent_id
        params[f"content_{i}"] = chunk.content
        params[f"vector_{i}"] = [float(x) for x in embedding]
        params[f"chunk_index_{i}"] = chunk.chunk_index

    sql = f"""
        INSERT INTO document_chunks
            (document_id, agent_id, content, embedding, chunk_index)
        VALUES
            {", ".join(rows)}
    """
    try:
        await db.execute(text(sql), params)
        await db.commit()
    except Exception as e:
        logger.exception(f"Error during saving doc chunks {str(e)}")
        await db.rollback()


async def set_status(
    db: AsyncSession, document_id: str, status: str, chunk_count: int = 0
):
    try:
        await db.execute(
            text("""
                UPDATE documents
                SET status = :status,
                    chunk_count = CASE WHEN :chunk_count > 0 THEN :chunk_count ELSE chunk_count END
                WHERE id = :document_id
            """),
            {"status": status, "document_id": document_id, "chunk_count": chunk_count},
        )
        await db.commit()
    except Exception as e:
        logger.exception(f"Error during updating doc indexing status {str(e)}")
        await db.rollback()
