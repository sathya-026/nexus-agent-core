"""
RAG Step 3 — Indexer

Orchestrates the full indexing pipeline for a single document:
  S3 download → text extraction → chunking → embedding → DB insert

Called as a FastAPI BackgroundTask so the HTTP response returns immediately
and indexing happens asynchronously. Status is written to Postgres at each
stage so the dashboard can show live progress.

Failure handling:
  - Any exception at any step marks the document as "failed"
  - The error is logged to analytics_events for the dashboard
  - No partial state is left in document_chunks (we delete before re-indexing)
"""

import asyncio
import json
import logging
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.rag.chunker import extract_text, chunk_text, Chunk
from app.rag.embedder import embed_chunks

from app.agent.semantic_router import invalidate_route_cache
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


# ── S3 client ─────────────────────────────────────────────────────────────────
#
# boto3 is synchronous. We run it in a thread pool executor so it doesn't
# block FastAPI's async event loop while waiting for S3 to respond.

_s3 = boto3.client(
    "s3",
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
)


async def _download_from_s3(s3_key: str) -> bytes:
    """
    Download a file from S3 and return its raw bytes.

    boto3 is blocking I/O, so we run it in a thread pool via asyncio's
    run_in_executor. This yields the event loop while S3 responds instead
    of blocking it for potentially several seconds on large files.
    """
    loop = asyncio.get_event_loop()

    def _sync_download():
        response = _s3.get_object(Bucket=settings.s3_bucket_name, Key=s3_key)
        return response["Body"].read()

    return await loop.run_in_executor(None, _sync_download)


# ── Status helpers ────────────────────────────────────────────────────────────

async def _set_status(db: AsyncSession, document_id: str, status: str, chunk_count: int = 0):
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


async def _log_analytics(
    db: AsyncSession,
    org_id: str,
    agent_id: str,
    document_id: str,
    event_type: str,
    payload: dict[str, Any],
):
    """Fire-and-forget analytics event. Failures here are logged, not raised."""
    try:
        await db.execute(
            text("""
                INSERT INTO analytics_events
                    (org_id, agent_id, conversation_id, event_type, payload)
                VALUES
                    (:org_id, :agent_id, NULL, :event_type, :payload)
            """),
            {
                "org_id": org_id,
                "agent_id": agent_id,
                "event_type": event_type,
                "payload": json.dumps(payload),
            },
        )
        await db.commit()
    except Exception as e:
        logger.warning("Failed to log analytics event: %s", e)
        await db.rollback()


# ── Bulk insert ───────────────────────────────────────────────────────────────

async def _insert_chunks(
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

# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_indexing_pipeline(
    db: AsyncSession,
    redis: Redis,
    document_id: str,
    agent_id: str,
    org_id: str,
    s3_key: str,
    file_type: str,
) -> None:
    """
    Full indexing pipeline for one document. Meant to run as a background task.

    Args:
        document_id: UUID of the document row in Postgres.
        agent_id:    UUID of the agent this document belongs to.
        org_id:      UUID of the org (for analytics events).
        s3_key:      S3 object key to download the file from.
        file_type:   MIME type (e.g. "application/pdf").
    """
    
    try:
        # ── Stage 1: Mark as indexing ────────────────────────────────────
        logger.info("Starting indexing: document_id=%s", document_id)
        await _set_status(db, document_id, "indexing")

        # ── Stage 2: Download from S3 ────────────────────────────────────
        logger.info("Downloading from S3: key=%s", s3_key)
        try:
            file_bytes = await _download_from_s3(s3_key)
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"S3 download failed: {e}") from e

        # ── Stage 3: Extract text ────────────────────────────────────────
        logger.info("Extracting text from %s", file_type)
        try:
            raw_text = extract_text(file_bytes, file_type)
        except ValueError as e:
            raise RuntimeError(f"Text extraction failed: {e}") from e

        if not raw_text.strip():
            raise RuntimeError("Document produced no extractable text")

        logger.info("Extracted %d characters", len(raw_text))

        # ── Stage 4: Chunk ───────────────────────────────────────────────
        chunks = chunk_text(raw_text)
        logger.info("Produced %d chunks", len(chunks))

        if not chunks:
            raise RuntimeError("Chunker produced no chunks from document text")

        # ── Stage 5: Embed ───────────────────────────────────────────────
        logger.info("Embedding %d chunks...", len(chunks))
        embeddings = await embed_chunks(chunks)

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch: {len(embeddings)} vectors for {len(chunks)} chunks"
            )

        # ── Stage 6: Insert into Postgres ────────────────────────────────
        logger.info("Inserting chunks into document_chunks...")
        await _insert_chunks(db, document_id, agent_id, chunks, embeddings)

        # ── Stage 7: Mark as indexed ─────────────────────────────────────
        await _set_status(db, document_id, "indexed", chunk_count=len(chunks))
        logger.info("Indexing complete: %d chunks stored", len(chunks))

        # ── Stage 8: Invalidate semantic router cache ───────────────────
        # So the new document is considered in routing decisions immediately.
        invalidate_route_cache(redis, agent_id)

        await _log_analytics(db, org_id, agent_id, document_id, "document indexed", {
            "chunk_count": len(chunks),
            "file_type": file_type,
        })

    except Exception as e:
        # Mark document as failed so the dashboard shows an error state
        logger.error("Indexing failed for document %s: %s", document_id, e, exc_info=True)
        try:
            await _set_status(db, document_id, "failed")
            await _log_analytics(db, org_id, agent_id, document_id, "indexing failed", {
                "error": str(e),
                "file_type": file_type,
            })
        except Exception as inner:
            logger.error("Failed to update error status: %s", inner)