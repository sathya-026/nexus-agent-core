from __future__ import annotations
from typing import Annotated

import logging
import jwt
from fastapi import Depends, HTTPException, Query, WebSocketException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.config import settings

_http_bearer = HTTPBearer(auto_error=True)

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"

class WidgetSession(BaseModel):
    sub: str        # JWT ID — uuid v7, useful for idempotency keys
    agent_id: str   # → agents.id
    org_id: str     # → organizations.id
    session_id: str # → conversations.session_id


def _decode(token: str) -> WidgetSession:
    try:
        payload = jwt.decode(
            token,
            settings.widget_jwt_secret,
            algorithms=[ALGORITHM],
            options={"require": ["exp", "sub", "iat"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid session token")
    except Exception as e:
        logger.error(f"Unexpected error decoding JWT: {e}")
        raise HTTPException(status_code=401, detail="Error processing session token")

    if payload.get("type") != "widget_session":
        raise HTTPException(status_code=401, detail="Wrong token type")

    return WidgetSession(
        sub=payload["sub"],
        agent_id=payload["agentId"],
        org_id=payload["orgId"],
        session_id=payload["sessionId"],
    )


def get_session(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_http_bearer)],
) -> WidgetSession:
    """For regular HTTP routes (e.g. REST endpoints)."""
    return _decode(credentials.credentials)

def get_sse_session(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_http_bearer)],
) -> WidgetSession:
    """For SSE chat routes.
    
    SSE is initiated by a regular HTTP POST (fetch API), so Authorization:
    Bearer works identically to REST — no special handling needed unlike WS.
    """
    return _decode(credentials.credentials)

SessionDep = Annotated[WidgetSession, Depends(get_sse_session)]