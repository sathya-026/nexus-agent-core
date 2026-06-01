"""
semantic_router.py
------------------
Decides whether a user message needs RAG, tools, both, or neither —
before the ReAct loop runs. Uses embedding similarity against agent-specific
tool and document descriptions. Results are cached in Redis per agent.

Route decision:
    rag   → retrieve from knowledge base only
    tool  → call tools only (returns matched tool names)
    both  → retrieve + call tools
    none  → skip both, go straight to LLM

Embedding backend:
    Primary  → sentence-transformers (all-MiniLM-L6-v2), local, zero API cost
    Model is loaded once at process startup (lazy singleton) and reused.

Cache key : route_layer:{agent_id}
Cache TTL : 24 hours (safety net — primary invalidation is event-driven)

Invalidate when:
    - A tool is created, updated, or deleted
    - A document reaches status = 'indexed' or is deleted
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Optional, Any

import numpy as np
import re
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)

META_INTENTS = [
    "what can you do",
    "what are your capabilities",
    "how can you help me",
    "what are you able to do",
    "show me what you can do",
    "list your features",
    "what are your functions",
    "what kind of tasks can you handle",
    "give me an overview of your abilities",
    "what are you good at",
]

INTENT_CACHE_KEY = "route_layer:intents"
INTENT_SIMILARITY_THRESHOLD = 0.75  # strict — meta intents must be unambiguous

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

MINILM_MODEL_NAME = "all-MiniLM-L6-v2"

# Separate thresholds — tool descriptions are precise/functional,
# doc descriptions are broader/topical. Tune per your domain.
TOOL_SIMILARITY_THRESHOLD = 0.55
DOC_SIMILARITY_THRESHOLD = 0.50
DOC_INFORMATION_REQUEST_THRESHOLD = 0.35

CACHE_TTL_SECONDS = 86_400  # 24 hours

ACKNOWLEDGEMENT_MESSAGES = {
    "ok",
    "okay",
    "k",
    "kk",
    "alright",
    "all right",
    "got it",
    "thanks",
    "thank you",
    "thx",
    "cool",
    "fine",
    "sure",
    "yes",
    "yep",
    "no",
    "nope",
}

SMALLTALK_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how are you doing",
    "whats up",
    "what's up",
}

QUESTION_RE = re.compile(
    r"\b(who|what|when|where|why|how|which|whose|whom|"
    r"can|could|would|should|do|does|did|is|are|was|were|"
    r"will|has|have|had)\b",
    re.IGNORECASE,
)
INFO_REQUEST_RE = re.compile(
    r"\b(tell me|explain|describe|summari[sz]e|list|show|give me|"
    r"find|search|look up|retrieve|compare)\b",
    re.IGNORECASE,
)
CONTEXT_REFERENCE_RE = re.compile(
    r"\b(he|him|his|she|her|hers|they|them|their|theirs|it|its|"
    r"this|that|these|those|there|same|above|previous|earlier|"
    r"former|latter)\b",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────────────────────
# MiniLM singleton
# ──────────────────────────────────────────────────────────────────────────────

# Loaded once on first use, reused for the lifetime of the process.
# ~120 MB RAM, ~100x faster than an API round-trip, zero per-call cost.

_minilm_model = None  # type: Optional[any]


def _get_model():
    """
    Lazy singleton loader for the MiniLM model.
    Thread-safe in Python due to the GIL; safe for asyncio (no threads here).
    Import is deferred so the module doesn't fail on machines without the package.
    """
    global _minilm_model
    if _minilm_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from e

        logger.exception("Loading MiniLM model '%s' into memory...", MINILM_MODEL_NAME)
        _minilm_model = SentenceTransformer(MINILM_MODEL_NAME)
        logger.exception("MiniLM model loaded.")
    return _minilm_model


# ──────────────────────────────────────────────────────────────────────────────
# Route enum
# ──────────────────────────────────────────────────────────────────────────────


class Route(str, Enum):
    RAG  = "rag"
    TOOL = "tool"
    BOTH = "both"
    META = "meta"
    NONE = "none"

@dataclass
class RouteResult:
    route         : Route
    matched_tools : list[str]        = field(default_factory=list)
    meta_context  : dict | None      = None
    query_text    : str | None       = None
    # meta_context shape when Route.META:
    # {
    #   "tool_names"        : [...],
    #   "tool_descriptions" : [...],
    #   "doc_names"         : [...],
    #   "doc_descriptions"  : [...],
    # }


# ──────────────────────────────────────────────────────────────────────────────
# Embedding helper
# ──────────────────────────────────────────────────────────────────────────────


def _embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using the local MiniLM model.
    Returns a list of float vectors (one per input text).

    Note: SentenceTransformer.encode() is synchronous and CPU-bound.
    For large batches this is fine. If you later need true async,
    wrap with asyncio.to_thread().
    """
    if not texts:
        return []

    model = _get_model()
    # normalize_embeddings=True gives unit vectors, making dot product == cosine sim.
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Cosine similarity
# ──────────────────────────────────────────────────────────────────────────────


def _cosine_similarities(
    matrix: list[list[float]],
    query: list[float],
) -> np.ndarray:
    """
    Return cosine similarity between each row in matrix and query.
    Since embeddings are already L2-normalised (normalize_embeddings=True),
    this is just a dot product — kept explicit for clarity.
    """
    mat = np.array(matrix)
    vec = np.array(query)
    norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(vec)
    norms = np.where(norms == 0, 1e-10, norms)
    return (mat @ vec) / norms

def _normalise_short_message(text_value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text_value.lower())).strip()


def _is_acknowledgement_or_smalltalk(message: str) -> bool:
    normalised = _normalise_short_message(message)
    if not normalised:
        return True

    return normalised in ACKNOWLEDGEMENT_MESSAGES or normalised in SMALLTALK_MESSAGES

def _is_information_request(message: str) -> bool:
    text_value = message.strip()
    if not text_value or _is_acknowledgement_or_smalltalk(text_value):
        return False

    return bool(
        "?" in text_value
        or QUESTION_RE.search(text_value)
        or INFO_REQUEST_RE.search(text_value)
    )


def _is_contextual_follow_up(message: str) -> bool:
    return  bool(CONTEXT_REFERENCE_RE.search(message)) or _is_information_request(message)


def _routing_query(user_message: str, conversation_context: str | None) -> str:
    message = user_message.strip()
    if conversation_context and _is_contextual_follow_up(message):
        return (
            "Recent conversation:\n"
            f"{conversation_context}\n\n"
            f"Current user request: {message}"
        )
    return message


# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────


async def _fetch_active_tools(db: AsyncSession, agent_id: str) -> list[Any]:
    try:
        query = """SELECT name, description FROM tools WHERE agent_id = :agent_id AND is_active = :is_active"""
        result = await db.execute(
            text(query), {"agent_id": agent_id, "is_active": True}
        )
        return result.fetchall()
    except Exception as e:
        logger.exception(f"Error during active tools fetch: {str(e)}")


async def _fetch_indexed_documents(db: AsyncSession, agent_id: str) -> list[Any]:
    try:
        query = """SELECT name, description FROM documents WHERE agent_id = :agent_id AND status = :status"""
        result = await db.execute(
            text(query), {"agent_id": agent_id, "status": "indexed"}
        )
        return result.fetchall()
    except Exception as e:
        logger.exception(f"Error during active tools fetch: {str(e)}")


# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────────────

# Bump this whenever you change the embedding model or dimension.
# Old caches with a different version will be silently ignored and rebuilt.
CACHE_VERSION = f"v1:{MINILM_MODEL_NAME}"


def _cache_key(agent_id: str) -> str:
    return f"route_layer:{agent_id}"

async def _read_intent_cache(redis) -> dict | None:
    raw = await redis.get(INTENT_CACHE_KEY)
    if not raw:
        return None
    data = json.loads(raw)
    if data.get("version") != CACHE_VERSION:
        logger.info("Intent cache version mismatch. Rebuilding.")
        return None
    return data


async def _read_cache(redis, agent_id: str) -> dict | None:
    raw = await redis.get(_cache_key(agent_id))
    if not raw:
        return None
    data = json.loads(raw)
    # Invalidate stale cache from a different model/version
    if data.get("version") != CACHE_VERSION:
        logger.exception(
            "Cache version mismatch for agent %s (got %s, expected %s). Rebuilding.",
            agent_id,
            data.get("version"),
            CACHE_VERSION,
        )
        return None
    return data


async def _write_cache(redis, agent_id: str, data: dict) -> None:
    data["version"] = CACHE_VERSION
    await redis.setex(_cache_key(agent_id), CACHE_TTL_SECONDS, json.dumps(data))


# ──────────────────────────────────────────────────────────────────────────────
# Build cache
# ──────────────────────────────────────────────────────────────────────────────

async def build_intent_cache(redis) -> dict:
    embeddings = _embed(META_INTENTS)
    data = {
        "version"         : CACHE_VERSION,
        "meta_embeddings" : embeddings,
    }
    await redis.setex(INTENT_CACHE_KEY, CACHE_TTL_SECONDS, json.dumps(data))
    logger.info("Intent cache built (%s meta utterances)", len(META_INTENTS))
    return data

async def build_route_cache(
    db: AsyncSession,
    redis,
    agent_id: str,
) -> dict:
    tools     = await _fetch_active_tools(db, agent_id)
    documents = await _fetch_indexed_documents(db, agent_id)

    tool_names        = [t.name for t in tools]
    tool_descriptions = [t.description for t in tools]
    doc_names         = [d.name for d in documents]
    doc_descriptions  = [f'{d.name}: {d.description}' for d in documents]

    all_descriptions = tool_descriptions + doc_descriptions
    all_embeddings   = _embed(all_descriptions)

    n_tools = len(tool_descriptions)

    data = {
        "tool_names"        : tool_names,
        "tool_descriptions" : tool_descriptions,
        "tool_embeddings"   : all_embeddings[:n_tools],
        "doc_names"         : doc_names,
        "doc_descriptions"  : doc_descriptions,
        "doc_embeddings"    : all_embeddings[n_tools:],
    }

    await _write_cache(redis, agent_id, data)
    logger.debug("Route cache built for agent %s (%s tools, %s docs)",
                 agent_id, len(tool_names), len(doc_names))
    return data
# ──────────────────────────────────────────────────────────────────────────────
# Invalidate cache
# ──────────────────────────────────────────────────────────────────────────────


async def invalidate_route_cache(redis, agent_id: str) -> None:
    """
    Call this whenever:
        - a tool is created / updated / deleted
        - a document reaches status='indexed' or is deleted
    """
    await redis.delete(_cache_key(agent_id))
    logger.exception("Route cache invalidated for agent %s", agent_id)


# ──────────────────────────────────────────────────────────────────────────────
# Model warm-up (optional, call at app startup)
# ──────────────────────────────────────────────────────────────────────────────


def warmup_model() -> None:
    """
    Force MiniLM to load and JIT-compile at app startup instead of on the
    first real request. Call this from your FastAPI lifespan or startup event.

    Example:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            warmup_model()
            yield
    """
    _embed(["warmup"])
    logger.exception("MiniLM model warm-up complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Main router
# ──────────────────────────────────────────────────────────────────────────────


async def route(
    db: AsyncSession,
    redis,
    agent_id: str,
    user_message: str,
    conversation_context: str | None = None,
) -> RouteResult:

    # ── 1. Load agent cache ───────────────────────────────────────────────────
    cached = await _read_cache(redis, agent_id)
    if cached is None:
        cached = await build_route_cache(db, redis, agent_id)

    tool_embeddings: list = cached.get("tool_embeddings", [])
    doc_embeddings: list = cached.get("doc_embeddings", [])
    tool_names: list[str] = cached.get("tool_names", [])

    # ── 2. Embed user message ─────────────────────────────────────────────────
    raw_query = user_message.strip()
    query_text = _routing_query(raw_query, conversation_context)
    raw_query_embedding = _embed([raw_query or user_message])[0]
    query_embedding = (
        raw_query_embedding
        if query_text == raw_query
        else _embed([query_text])[0]
    )

    # ── 3. Meta intent check ──────────────────────────────────────────────────
    intent_cache = await _read_intent_cache(redis)
    if intent_cache is None:
        intent_cache = await build_intent_cache(redis)

    meta_embeddings = intent_cache.get("meta_embeddings", [])
    if meta_embeddings:
        meta_scores = _cosine_similarities(meta_embeddings, query_embedding)
        if meta_scores.max() >= INTENT_SIMILARITY_THRESHOLD:
            logger.debug("Meta intent matched for agent %s", agent_id)
            return RouteResult(
                route=Route.META,
                meta_context={
                    "tool_names": cached.get("tool_names", []),
                    "tool_descriptions": cached.get("tool_descriptions", []),
                    "doc_names": cached.get("doc_names", []),
                    "doc_descriptions": cached.get("doc_descriptions", []),
                },
            )

    # ── 4. Nothing configured → skip retrieval entirely ──────────────────────
    if not tool_embeddings and not doc_embeddings:
        return RouteResult(route=Route.NONE)

    # ── 5. Score against document descriptions ────────────────────────────────
    use_rag = False
    best_doc_score = 0.0
    if doc_embeddings:
        doc_scores = _cosine_similarities(doc_embeddings, query_embedding)
        best_doc_score = float(doc_scores.max())
        use_rag = bool(best_doc_score >= DOC_SIMILARITY_THRESHOLD)

        if (
            not use_rag
            and _is_information_request(raw_query)
            and best_doc_score >= DOC_INFORMATION_REQUEST_THRESHOLD
        ):
            use_rag = True

        if not use_rag and _is_contextual_follow_up(raw_query):
            use_rag = True

        if use_rag:
            query_text = "Meta data \n" + f"{cached.get('doc_names', [])[doc_scores.argmax()]} {cached.get('doc_descriptions', [])[doc_scores.argmax()]}" + "\n\n" + query_text

    # ── 6. Score against tool descriptions ───────────────────────────────────
    matched_tools: list[str] = []
    if tool_embeddings:
        tool_scores = _cosine_similarities(tool_embeddings, query_embedding)
        matched_tools = [
            tool_names[i]
        for i, score in enumerate(tool_scores)
            if score >= TOOL_SIMILARITY_THRESHOLD
        ]

    # ── 7. Decide route ───────────────────────────────────────────────────────
    use_tool = bool(matched_tools)

    if use_rag and use_tool:
        decision = Route.BOTH
    elif use_rag:
        decision = Route.RAG
    elif use_tool:
        decision = Route.TOOL
    else:
        decision = Route.NONE

    logger.debug(
        "Semantic route for agent %s: %s | matched tools: %s",
        agent_id,
        decision,
        matched_tools,
    )

    return RouteResult(route=decision, matched_tools=matched_tools, query_text=query_text)