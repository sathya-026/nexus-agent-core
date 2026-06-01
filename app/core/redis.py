# app/core/redis.py

import logging
import redis.asyncio as aioredis
from redis.asyncio import Redis

from fastapi import Request

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Connection factory
# ──────────────────────────────────────────────────────────────────────────────

def create_redis_client(redis_url: str) -> Redis:
    """
    Create a Redis client with a connection pool.
    decode_responses=True so all values come back as str, not bytes —
    consistent with json.loads() expectations in the router.
    """
    return aioredis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dependency
# ──────────────────────────────────────────────────────────────────────────────

async def get_redis(request: Request) -> Redis:
    """FastAPI dependency — injects the shared Redis client."""
    return request.app.state.redis