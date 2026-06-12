"""
app/routing/router.py
---------------------
Semantic router — the single entry point for all routing decisions.

Determines the retrieval strategy for a user message before the ReAct
loop in planner.py runs. Runs on every chat turn; must be fast.

Decision flow
-------------
    1. Short-circuit: acknowledgements / small talk          → NONE
    2. Load (or build) the agent's route cache from Redis
    3. Embed the user message (+ conversation context if contextual)
    4. Meta intent check ("what can you do?")                → META
    5. No tools or docs configured — legitimate empty agent  → NONE
    6. Score user embedding against document embeddings      → use_rag
    7. Score user embedding against tool embeddings          → matched_tools
    8. Decide: BOTH / RAG / TOOL
       All scores missed but config exists → LLM fallback    → BOTH / RAG / TOOL / NONE

Thresholds
----------
Kept in this module — the only place they're consumed. Promote to
app/config.py (pydantic-settings) if you want env-var control without
a code deploy.

    TOOL_SIMILARITY_THRESHOLD         = 0.55
    DOC_SIMILARITY_THRESHOLD          = 0.50
    DOC_INFORMATION_REQUEST_THRESHOLD = 0.35

The two-level document threshold is intentional:
    - 0.50  catches strong topical matches
    - 0.35  catches weaker matches that are still worth retrieving when
            the message structure itself signals an information request
            (question mark, "explain", "tell me", etc.)
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.query_router.types import Route, RouteResult
from app.agent.query_router.embedder import embed, cosine_similarities, embed_async, blend_embeddings
from app.agent.query_router.classifier import (
    INTENT_SIMILARITY_THRESHOLD,
    build_routing_query,
    is_acknowledgement_or_smalltalk,
    is_contextual_follow_up,
    is_information_request,
    is_resource_listing_query,
)
from app.agent.query_router.cache import (
    RouteLayerData,
    get_or_build_intent_cache,
    get_or_build_route_cache,
    get_conversation_intent,
)
from app.agent.query_router.fallback import llm_route_fallback

logger = logging.getLogger(__name__)

TOOL_SIMILARITY_THRESHOLD         = 0.55
DOC_SIMILARITY_THRESHOLD          = 0.50
DOC_INFORMATION_REQUEST_THRESHOLD = 0.35


async def route(
    db: AsyncSession,
    redis,
    agent_id: str,
    conversation_id: str,
    user_message: str,
    conversation_context: str | None = None,
) -> RouteResult:
    """
    Decide the retrieval strategy for a single user message.

    Args:
        db:                   Async SQLAlchemy session (used only on cache miss).
        redis:                Redis connection (from FastAPI Depends).
        agent_id:             The agent handling this conversation.
        conversation_id:      The conversation ID (used to retrieve/store conversation intent).
        user_message:         The raw text from the end user.
        conversation_context: Optional recent history string. Injected into
                              the routing query when the message is a
                              contextual follow-up (pronouns, deictics).

    Returns:
        RouteResult with route, matched_tools, and query_text set.
    """

    # ── 1. Short-circuit: acknowledgements and small talk ─────────────────────
    # No retrieval intent — skip embedding entirely.
    if is_acknowledgement_or_smalltalk(user_message):
        return RouteResult(route=Route.NONE)

    # ── 2. Load agent route cache ─────────────────────────────────────────────
    # Returns cached data on hit; builds from Postgres + MiniLM on miss.
    cached: RouteLayerData = await get_or_build_route_cache(db, redis, agent_id)

    tool_embeddings: list[list[float]] = cached["tool_embeddings"]
    doc_embeddings:  list[list[float]] = cached["doc_embeddings"]
    tool_names:      list[str]         = cached["tool_names"]

    # ── 3. Embed user message ─────────────────────────────────────────────────
    # raw_query    → used for meta intent scoring (should be uncontextualised)
    # query_text   → used for tool/doc scoring (enriched with context if needed)
    # query_embedding → matches query_text so follow-ups embed the full topic
    raw_query   = user_message.strip()

    # Fast-path: resource listing queries bypass embedding entirely.
    #
    # "What documents u have?" scores ~0.29 against META_INTENTS embeddings
    # because informal 'u' shifts the vector enough to miss the 0.75 threshold.
    # The regex catches this class of query unconditionally and routes to META
    # so the planner can list available documents and tools from meta_context.
    #
    # This runs after the cache load (we need meta_context) but before any
    # embed() call (no point embedding if we already know the route).
    if is_resource_listing_query(raw_query):
        logger.debug(
            "Resource listing query fast-path matched for agent %s.", agent_id
        )
        return RouteResult(
            route=Route.META,
            meta_context={
                "tool_names":        cached["tool_names"],
                "tool_descriptions": cached["tool_descriptions"],
                "doc_names":         cached["doc_names"],
                "doc_descriptions":  cached["doc_descriptions"],
            },
        )

    query_text  = build_routing_query(raw_query, conversation_context)

    raw_embedding   = (await embed_async([raw_query]))[0]
    query_embedding = (
        raw_embedding
        if query_text == raw_query
        else (await embed_async([query_text]))[0]
    )
    
    # ── Load conversation intent and blend with current query ────────────────────
    # If prior turns exist, blend their intent with the current query.
    # This improves routing for ambiguous queries by leveraging multi-turn context.
    # Weighting: 70% current query, 30% conversation history.
    blended_query_embedding = query_embedding
    conversation_intent = await get_conversation_intent(redis, conversation_id)
    
    if conversation_intent:
        try:
            blended_query_embedding = blend_embeddings(
                [query_embedding, conversation_intent["conversation_embedding"]],
                weights=[0.7, 0.3],
            )
            logger.debug(
                "Blended query embedding with conversation intent (turn_count=%d) for agent %s.",
                conversation_intent["turn_count"], agent_id
            )
        except Exception as e:
            logger.warning(
                "Failed to blend embeddings for agent %s: %s — using current query only.",
                agent_id, str(e)
            )
            blended_query_embedding = query_embedding

    # ── 4. Meta intent check ──────────────────────────────────────────────────
    # Score raw_query (not context-enriched) — meta questions like
    # "what can you do?" should match on their own phrasing.
    intent_data    = await get_or_build_intent_cache(redis)
    meta_embeddings = intent_data["meta_embeddings"]

    if meta_embeddings:
        meta_scores = cosine_similarities(meta_embeddings + tool_embeddings + doc_embeddings, raw_embedding)
        if float(meta_scores.max()) >= INTENT_SIMILARITY_THRESHOLD:
            logger.debug("Meta intent matched for agent %s.", agent_id)
            return RouteResult(
                route=Route.META,
                meta_context={
                    "tool_names":        cached["tool_names"],
                    "tool_descriptions": cached["tool_descriptions"],
                    "doc_names":         cached["doc_names"],
                    "doc_descriptions":  cached["doc_descriptions"],
                },
            )

    # ── 5. No config — legitimate NONE ────────────────────────────────────────
    # No tools or documents are set up for this agent. There is nothing
    # to retrieve, so the LLM fallback would have nothing to offer either.
    # Return NONE immediately without invoking the fallback.
    if not tool_embeddings and not doc_embeddings:
        return RouteResult(route=Route.NONE)

    # ── 6. Score against document embeddings ──────────────────────────────────
    use_rag = False
    if doc_embeddings:
        doc_scores     = cosine_similarities(doc_embeddings, blended_query_embedding)
        best_doc_score = float(doc_scores.max())

        # Primary threshold — strong topical match
        use_rag = best_doc_score >= DOC_SIMILARITY_THRESHOLD

        # Lower threshold for explicit information requests ("explain X",
        # "what is Y?"). DOC_INFORMATION_REQUEST_THRESHOLD is only checked
        # here, not inside is_contextual_follow_up(), so the gate remains live.
        if not use_rag and is_information_request(raw_query):
            use_rag = best_doc_score >= DOC_INFORMATION_REQUEST_THRESHOLD

        # Contextual follow-ups ("what about it?", "tell me more about that")
        # reference a topic already in the conversation. If the conversation
        # context was injected into the query embedding, the topic similarity
        # is likely above threshold anyway. This branch covers the edge case
        # where the enriched embedding still scores below both thresholds.
        if not use_rag and is_contextual_follow_up(raw_query):
            use_rag = True

    # ── 7. Score against tool embeddings ─────────────────────────────────────
    matched_tools: list[str] = []
    if tool_embeddings:
        tool_scores   = cosine_similarities(tool_embeddings, blended_query_embedding)
        matched_tools = [
            tool_names[i]
            for i, score in enumerate(tool_scores)
            if float(score) >= TOOL_SIMILARITY_THRESHOLD
        ]

    # ── 8. Decide route ───────────────────────────────────────────────────────
    use_tool = bool(matched_tools)

    if use_rag and use_tool:
        decision = Route.BOTH
    elif use_rag:
        decision = Route.RAG
    elif use_tool:
        decision = Route.TOOL
    else:
        decision = Route.NONE

    # ── LLM fallback ──────────────────────────────────────────────────────────
    # Config exists but every embedding score missed the threshold.
    # This is a low-confidence NONE — the semantic router may have missed
    # the intent due to jargon, paraphrasing mismatch, or a short message.
    # Hand off to the LLM for a single structured routing call.
    #
    # This branch is NOT reached for the no-config NONE at step 5.
    if decision == Route.NONE:
        logger.debug(
            "Embedding router returned NONE with config present for agent %s "
            "— invoking LLM fallback.",
            agent_id,
        )
        result = await llm_route_fallback(
            user_message=raw_query,
            conversation_context=conversation_context,
            tool_names=cached["tool_names"],
            tool_descriptions=cached["tool_descriptions"],
            doc_names=cached["doc_names"],
            doc_descriptions=cached["doc_descriptions"],
        )
        # Set blended embedding for later conversation intent update
        result.conversation_intent_embedding = blended_query_embedding
        return result

    logger.debug(
        "Route for agent %s: %s | tools: %s", agent_id, decision, matched_tools
    )
    return RouteResult(
        route=decision,
        matched_tools=matched_tools,
        query_text=query_text,
        conversation_intent_embedding=blended_query_embedding,
    )