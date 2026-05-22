"""
Indexing router — called by NestJS after a document upload is confirmed.

Flow:
  1. User uploads file to S3 via NestJS (pre-signed URL)
  2. NestJS calls POST /indexing/index with { document_id, agent_id, s3_key }
  3. This router kicks off a background task: download → chunk → embed → store
  4. NestJS polls document status (pending → indexing → indexed / failed)
"""

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.rag.indexer import run_indexing_pipeline

router = APIRouter()


class IndexRequest(BaseModel):
    document_id: str
    agent_id: str
    org_id: str
    s3_key: str
    file_type: str


def _verify_internal(secret: str):
    """Reject calls that don't come from NestJS."""
    if secret != settings.nestjs_internal_secret:
        raise HTTPException(status_code=401, detail="Unauthorized internal call")

@router.post("/index")
async def index_document(
    req: IndexRequest,
    background_tasks: BackgroundTasks,
    x_internal_secret: str = Header(...),
):
    _verify_internal(x_internal_secret)

    # Returns 200 immediately — indexing runs in the background.
    # NestJS polls document status to know when it's done.
    BackgroundTasks.add_task(
        run_indexing_pipeline,
        document_id=req.document_id,
        agent_id=req.agent_id,
        org_id=req.org_id,
        s3_key=req.s3_key,
        file_type=req.file_type,
    )   

    return {"status": "queued", "document_id": req.document_id}
