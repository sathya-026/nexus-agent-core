# app/routers/chat.py
"""
Internal chat endpoint — called only by NestJS, never by the widget directly.

Auth: X-Internal-Secret header (same pattern as the indexing router).
      All widget-facing auth (API key, domain check, conversation management)
      is handled upstream by NestJS before this endpoint is reached.

Request body carries pre-resolved context:
    agent_id        — validated and org-scoped by NestJS
    org_id          — resolved from the widget's X-Api-Key by NestJS
    conversation_id — created or fetched by NestJS
    message         — raw user text, validated (non-empty, max length) by NestJS

Response: text/event-stream (SSE).
Each event is a JSON object so the widget has a single parsing path.

Event shapes:
    {"type": "token",  "content": "Hello"}   — one per streamed token
    {"type": "done"}                          — stream complete, widget stops reading
    {"type": "error",  "message": "..."}      — planner failed, widget shows fallback
"""

import json
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.redis import get_redis
from app.core.security.widget_session import SessionDep
from app.database import get_db
from app.agent import planner

from app.db.conversations import get_or_create_conversation


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    agent_id: str = Field(..., description="Agent UUID, validated by NestJS")
    session_id: str = Field(
        ..., description="Session UUID"
    )
    message: str = Field(..., min_length=1, max_length=4000)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


async def _sse_stream(generator):
    """
    Wrap the planner's token generator in SSE format.

    SSE wire format:
        data: <json>\n\n

    Each event is a JSON object — consistent parsing for the widget
    regardless of event type.

    Error handling:
        If the generator raises mid-stream, we emit an error event and
        close cleanly. The widget has already received partial content
        at this point — the error event signals it to stop expecting more.
    """
    try:
        async for token in generator:
            payload = json.dumps({"type": "token", "content": token})
            yield f"data: {payload}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as exc:
        logger.exception("Planner stream failed mid-response: %s", exc)
        error_payload = json.dumps(
            {
                "type": "error",
                "message": "The agent encountered an error. Please try again.",
            }
        )
        yield f"data: {error_payload}\n\n"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("", status_code=status.HTTP_200_OK)
async def chat(
    session: SessionDep,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
) -> StreamingResponse:
    """
    Run the ReAct planner and stream the response to Widget
    """
    if body.agent_id != session.agent_id:
        raise HTTPException(status_code=403, detail="Agent mismatch")
    if body.session_id != session.session_id:
        raise HTTPException(status_code=403, detail="Session mismatch")    

    conversation_id = await get_or_create_conversation(db=db, agent_id=body.agent_id, session_id=body.session_id)

    generator = planner.stream(
        db=db,
        redis=redis,
        agent_id=body.agent_id,
        org_id=session.org_id,
        conversation_id=conversation_id,
        user_message=body.message,
    )

    return StreamingResponse(
        _sse_stream(generator),
        media_type="text/event-stream",
        headers={
            # Prevent any intermediate proxy from buffering the stream.
            # nginx needs `proxy_buffering off` on the /chat location block.
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
