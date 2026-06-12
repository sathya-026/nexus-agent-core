"""
app/routing/cache.py
--------------------
Typed Redis stores for agent routing data.

Defines two CacheClient instances built on app.core.cache.CacheClient:

    route_cache   — per-agent store: tool + document names, descriptions,
                    and pre-computed MiniLM embeddings.
    intent_cache  — global store: embeddings for META_INTENTS utterances.
                    Shared across all agents; rebuilt only when META_INTENTS
                    or CACHE_VERSION changes.

TypedDicts
----------
RouteLayerData and IntentCacheData describe the exact JSON shape stored
in each cache. Callers receive typed dicts — no bare .get("key", []) calls
with unknown return types. Zero runtime cost; types are erased at runtime.

Cache invalidation
------------------
    invalidate_route_cache(redis, agent_id)
        → call when a tool is created / updated / deleted
        → call when a document reaches status='indexed' or is deleted

Schema prerequisite
-------------------
    documents.description column must exist:
        ALTER TABLE documents ADD COLUMN description text;
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import CacheClient
from app.agent.query_router.embedder import CACHE_VERSION, embed_async
from app.agent.query_router.classifier import META_INTENTS
from app.db.documents import fetch_indexed_documents
from app.db.tools import load_tools

logger = logging.getLogger(__name__)


# ── Payload shapes ────────────────────────────────────────────────────────────

class RouteLayerData(TypedDict):
    tool_names:        list[str]
    tool_descriptions: list[str]
    tool_embeddings:   list[list[float]]
    doc_names:         list[str]
    doc_descriptions:  list[str]
    doc_embeddings:    list[list[float]]


class IntentCacheData(TypedDict):
    meta_embeddings: list[list[float]]


class ConversationIntentData(TypedDict):
    """
    Per-conversation rolling context embedding for improved routing.
    
    conversation_embedding: Weighted average of recent turns' intents.
        Built from the last N assistant responses (sliding window).
        Allows router to blend current query with conversation history.
    
    turn_count: Number of turns in the window used to build the embedding.
        Helps with debugging: lower counts are noisier.
    """
    conversation_embedding: list[float]
    turn_count: int


# ── Cache store instances ─────────────────────────────────────────────────────
# Module-level constants — created at import time, no Redis connection needed.
# Redis is injected at call time via FastAPI dependency injection.

_ROUTE_TTL  = 86_400   # 24 hours
_INTENT_TTL = 86_400   # 24 hours
_INTENT_KEY = "global" # single shared entry, no per-agent key needed
_CONVERSATION_INTENT_TTL = 86_400  # 24 hours

route_cache: CacheClient[RouteLayerData] = CacheClient(
    namespace="routing:route_layer",
    version=CACHE_VERSION,
    default_ttl=_ROUTE_TTL,
)

intent_cache: CacheClient[IntentCacheData] = CacheClient(
    namespace="routing:intents",
    version=CACHE_VERSION,
    default_ttl=_INTENT_TTL,
)

conversation_intent_cache: CacheClient[ConversationIntentData] = CacheClient(
    namespace="routing:conversation_intent",
    version=CACHE_VERSION,
    default_ttl=_CONVERSATION_INTENT_TTL,
)



# ── Build ─────────────────────────────────────────────────────────────────────

async def build_route_cache(
    db: AsyncSession,
    redis,
    agent_id: str,
) -> RouteLayerData:
    """
    Fetch tools and documents from Postgres, embed their descriptions
    with MiniLM, and write the result to Redis.

    All tool and document descriptions are embedded in a single model.encode()
    call to avoid multiple round-trips through the model.

    Returns the built data so the caller can use it immediately
    without a second Redis round-trip.
    """
    tools     = await load_tools(db, agent_id)
    documents = await fetch_indexed_documents(db, agent_id)

    tool_names        = [row.name        for row in tools]
    tool_descriptions = [row.description for row in tools]
    doc_names         = [row.name        for row in documents]

    # Prefix doc descriptions with their name to give the router
    # richer signal when scoring against short user messages.
    doc_descriptions  = [
        f"{row.name}: {row.description}" for row in documents
    ]

    # Single embed call for all descriptions
    all_descriptions = tool_descriptions + doc_descriptions
    all_embeddings   = await embed_async(all_descriptions)

    n_tools: int = len(tool_descriptions)

    data: RouteLayerData = {
        "tool_names"       : tool_names,
        "tool_descriptions": tool_descriptions,
        "tool_embeddings"  : all_embeddings[:n_tools],
        "doc_names"        : doc_names,
        "doc_descriptions" : doc_descriptions,
        "doc_embeddings"   : all_embeddings[n_tools:],
    }

    await route_cache.set(redis, agent_id, data)
    logger.debug(
        "Route cache built for agent %s — %d tool(s), %d doc(s).",
        agent_id, len(tool_names), len(doc_names),
    )
    return data


async def build_intent_cache(redis) -> IntentCacheData:
    """
    Embed META_INTENTS and write to Redis under the shared global key.

    This entry is shared across all agents. It only needs rebuilding
    when META_INTENTS is edited or CACHE_VERSION changes — the version
    mismatch on next read triggers an automatic rebuild.
    """
    data: IntentCacheData = {
        "meta_embeddings": await embed_async(META_INTENTS),
    }
    await intent_cache.set(redis, _INTENT_KEY, data)
    logger.info(
        "Intent cache built — %d meta utterance(s).", len(META_INTENTS)
    )
    return data


# ── Read-or-build helpers ─────────────────────────────────────────────────────

async def get_or_build_route_cache(
    db: AsyncSession,
    redis,
    agent_id: str,
) -> RouteLayerData:
    """Return cached route data for agent_id, building it on a miss."""
    cached = await route_cache.get(redis, agent_id)
    if cached is not None:
        return cached
    return await build_route_cache(db, redis, agent_id)


async def get_or_build_intent_cache(redis) -> IntentCacheData:
    """Return the global intent cache, building it on a miss."""
    cached = await intent_cache.get(redis, _INTENT_KEY)
    if cached is not None:
        return cached
    return await build_intent_cache(redis)


async def get_conversation_intent(
    redis,
    conversation_id: str,
) -> ConversationIntentData | None:
    """
    Retrieve stored conversation intent embedding for a conversation.
    
    Returns None if no conversation intent has been built yet (first turn).
    """
    return await conversation_intent_cache.get(redis, conversation_id)


async def set_conversation_intent(
    redis,
    conversation_id: str,
    data: ConversationIntentData,
) -> None:
    """
    Store or update conversation intent embedding.
    
    Called after each planner turn with the weighted embedding of recent
    assistant responses. Automatically expires after _CONVERSATION_INTENT_TTL.
    """
    await conversation_intent_cache.set(redis, conversation_id, data)


# ── Invalidate ────────────────────────────────────────────────────────────────

async def invalidate_route_cache(redis, agent_id: str) -> None:
    """
    Drop the route cache entry for agent_id.

    Call this from the NestJS → FastAPI event whenever:
        - a tool is created, updated, or deleted
        - a document reaches status='indexed' or is deleted

    The next request for this agent will trigger build_route_cache()
    automatically via get_or_build_route_cache().
    """
    await route_cache.delete(redis, agent_id)
    logger.info("Route cache invalidated for agent %s.", agent_id)