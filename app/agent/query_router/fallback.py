"""
app/routing/fallback.py
-----------------------
LLM-based fallback router.

Invoked by router.py only when:
    1. The embedding-based router scored Route.NONE
    2. AND tools or documents are actually configured for the agent

Meaning: the semantic similarity pass missed the user's intent. Common
causes include domain jargon MiniLM hasn't seen, paraphrasing that
doesn't match stored descriptions, or messages that are too short or
ambiguous to produce a strong embedding signal.

Strategy
--------
A single non-streaming gpt-4o-mini call with forced tool_choice produces
a structured routing decision. Forced tool_choice (rather than JSON mode)
guarantees the response always has the expected schema — no preamble,
no markdown fences, no reasoning prose before the JSON.

temperature=0 makes routing deterministic: the same ambiguous message
always hits the same path, which matters for debugging.

Hallucination guard
-------------------
Returned tool names are validated against the caller's known tool list
before any name reaches the planner. Names not in the known set are
dropped and logged at WARNING — a consistent pattern in the logs is a
signal to improve that tool's description.

Failure behaviour
-----------------
Any failure (API error, malformed response, JSON parse error) returns
RouteResult(route=Route.NONE). The planner always gets a usable result
and falls back to general LLM knowledge. The fallback fails open.
"""

from __future__ import annotations

import json
import logging

from app.agent.query_router.types import Route, RouteResult

logger = logging.getLogger(__name__)


# ── OpenAI client singleton ───────────────────────────────────────────────────
# Initialised on first fallback call; reused for the process lifetime.
# Lazy import keeps startup fast and avoids importing openai on services
# that never hit this path.

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        from app.config import settings
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info("AsyncOpenAI client initialised for LLM fallback router.")
    return _openai_client


# ── Tool schema ───────────────────────────────────────────────────────────────
# Forced tool_choice on this schema guarantees a structured response
# on every call. The LLM has no choice but to fill use_rag and matched_tools.

_ROUTE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "route_decision",
        "description": (
            "Decide which resources are relevant to answering the user's request. "
            "Only select resources that clearly apply — when in doubt, leave them out."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "use_rag": {
                    "type": "boolean",
                    "description": (
                        "True if at least one knowledge base document is relevant "
                        "and would improve the answer."
                    ),
                },
                "matched_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Names of tools to invoke, matching the names exactly as listed. "
                        "Empty array if no tool is relevant."
                    ),
                },
            },
            "required": ["use_rag", "matched_tools"],
        },
    },
}

_SYSTEM_PROMPT = (
    "You are a routing assistant for an AI agent. "
    "Given a user message and a list of available tools and knowledge base documents, "
    "decide which resources the agent should use to answer. "
    "Be conservative — only select resources that clearly apply to the request. "
    "Do not select anything speculatively or just in case."
)


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    user_message: str,
    conversation_context: str | None,
    tool_names: list[str],
    tool_descriptions: list[str],
    doc_names: list[str],
    doc_descriptions: list[str],
) -> str:
    """
    Build a compact user-turn prompt for the routing decision.

    Sections are only emitted when they have content — a tools-only
    agent produces no documents section, and vice versa. Conversation
    context is appended when present so the LLM can resolve follow-ups
    that reference prior topics.
    """
    parts: list[str] = []

    if tool_names:
        tool_lines = "\n".join(
            f"- {name}: {desc}"
            for name, desc in zip(tool_names, tool_descriptions)
        )
        parts.append(f"Available tools:\n{tool_lines}")

    if doc_names:
        doc_lines = "\n".join(
            f"- {name}: {desc}"
            for name, desc in zip(doc_names, doc_descriptions)
        )
        parts.append(f"Available knowledge base documents:\n{doc_lines}")

    if conversation_context:
        parts.append(f"Recent conversation:\n{conversation_context}")

    parts.append(f"User message: {user_message}")
    return "\n\n".join(parts)


# ── Entry point ───────────────────────────────────────────────────────────────

async def llm_route_fallback(
    user_message: str,
    conversation_context: str | None,
    tool_names: list[str],
    tool_descriptions: list[str],
    doc_names: list[str],
    doc_descriptions: list[str],
) -> RouteResult:
    """
    Ask gpt-4o-mini to make the routing decision.

    Parameters are passed explicitly (not as a cached dict) so the
    function has no dependency on the cache layer's internal structure
    and the type checker can verify each argument individually.
    """
    # Defensive guard — should not be called without any config
    if not tool_names and not doc_names:
        return RouteResult(route=Route.NONE)

    prompt = _build_prompt(
        user_message=user_message,
        conversation_context=conversation_context,
        tool_names=tool_names,
        tool_descriptions=tool_descriptions,
        doc_names=doc_names,
        doc_descriptions=doc_descriptions,
    )
        
    client = _get_openai_client()

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            temperature=0,
            tools=[_ROUTE_TOOL],
            tool_choice={"type": "function", "function": {"name": "route_decision"}},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
    except Exception as exc:
        logger.exception("LLM fallback API call failed: %s", exc)
        return RouteResult(route=Route.NONE)

    try:
        tool_call = response.choices[0].message.tool_calls[0]
        args      = json.loads(tool_call.function.arguments)
    except Exception as exc:
        logger.exception("LLM fallback: failed to parse tool_call response: %s", exc)
        return RouteResult(route=Route.NONE)

    use_rag: bool = bool(args.get("use_rag", False))

    # Validate returned tool names against the known list.
    # The model occasionally returns a plausible-sounding but non-existent
    # name. Passing an unknown name to the planner causes a tool-lookup
    # failure, so we drop and log instead.
    valid_tools   = set(tool_names)
    raw_tools     = set(args.get("matched_tools", []))
    matched_tools = [name for name in args.get("matched_tools", []) if name in valid_tools]
    hallucinated  = raw_tools - valid_tools

    if hallucinated:
        logger.warning(
            "LLM fallback returned unknown tool name(s) — dropped: %s", hallucinated
        )

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
        "LLM fallback decision: %s | matched_tools: %s", decision, matched_tools
    )
    return RouteResult(route=decision, matched_tools=matched_tools)