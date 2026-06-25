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

import logging

from app.common.constants import AnalyticEvent
from app.db.analytics import log_event
from app.db.documents import insert_chunks, set_status
from botocore.exceptions import BotoCoreError, ClientError

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.common.file_helper import extract_text
from app.rag.chunker import chunk_text
from app.rag.embedder import embed_chunks
from app.storage.s3 import download_file


from app.agent.query_router import invalidate_route_cache
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

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
        await set_status(db, document_id, "indexing")

        # ── Stage 2: Download from S3 ────────────────────────────────────
        logger.info("Downloading from S3: key=%s", s3_key)
        try:
            file_bytes = await download_file(s3_key)
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
        await insert_chunks(db, document_id, agent_id, chunks, embeddings)

        # ── Stage 7: Mark as indexed ─────────────────────────────────────
        await set_status(db, document_id, "indexed", chunk_count=len(chunks))
        logger.info("Indexing complete: %d chunks stored", len(chunks))

        # ── Stage 8: Invalidate semantic router cache ───────────────────
        # So the new document is considered in routing decisions immediately.
        await invalidate_route_cache(redis, agent_id)

        await log_event(db, org_id, agent_id, AnalyticEvent.DOCUMENT_INDEXED, {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "file_type": file_type,
        })

    except Exception as e:
        # Mark document as failed so the dashboard shows an error state
        logger.error("Indexing failed for document %s: %s", document_id, e, exc_info=True)
        try:
            await set_status(db, document_id, "failed")
            await log_event(db, org_id, agent_id, AnalyticEvent.DOCUMENT_INDEXING_FAILED, {
                "document_id": document_id,
                "error": str(e),
                "file_type": file_type,
            })
        except Exception as inner:
            logger.error("Failed to update error status: %s", inner)