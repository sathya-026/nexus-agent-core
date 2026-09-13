"""
ReAct planner — orchestrates one full user turn.

Per-turn message persistence (updated design):
    user message          → saved before the loop
    assistant (tool-call) → saved at the start of each iteration that uses tools;
                            tool_calls rows are linked to this message's id
    assistant (final)     → saved after the loop completes

role="tool" is never written to messages.
Tool inputs and outputs live entirely in tool_calls.input / tool_calls.output,
joined back onto their assistant message at read time by load_memory().
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from typing import Any, AsyncGenerator

from app.ai.factory import get_provider
from app.ai.types import ContentDelta, ToolCallComplete, ToolSchema, UsageEvent

from app.common.constants import AnalyticEvent
from app.db.agents import load_agent
from app.db.analytics import log_event
from app.db.conversations import update_conversation_stats
from app.db.messages import save_message, update_message
from app.db.tool_call import save_tool_call
from app.db.tools import load_tools

from redis.asyncio.client import Redis

from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.retriever import retrieve, format_context_for_prompt
from app.agent.memory import (
    load_memory,
    update_conversation_intent,
)
from app.agent import tool_executor as Executor
from app.agent.query_router.router import route as semantic_router, Route

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class PlannerResult:
    response: str
    tokens_used: int
    iterations: int
    rag_hit: bool
    latency_ms: int


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run(
    db: AsyncSession,
    redis: Redis,
    agent_id: str,
    org_id: str,
    conversation_id: str,
    user_message: str,
) -> PlannerResult:
    """
    Batch entry point — collects all tokens and returns a PlannerResult.
    Used for testing and non-streaming contexts.
    """
    start = time.monotonic()
    tokens: list[str] = []

    async for token in _execute(
        db, redis, agent_id, org_id, conversation_id, user_message
    ):
        tokens.append(token)

    return PlannerResult(
        response="".join(tokens),
        latency_ms=int((time.monotonic() - start) * 1000),
    )


async def _execute(
    db: AsyncSession,
    redis: Redis,
    agent_id: str,
    org_id: str,
    conversation_id: str,
    user_message: str,
) -> AsyncGenerator[str, None]:

    start_ms = time.monotonic()

    # -- Memory + semantic routing --
    memory = await load_memory(db, conversation_id)
    routing_context = memory.to_context_string()

    route_result = await semantic_router(
        db=db,
        redis=redis,
        agent_id=agent_id,
        conversation_id=conversation_id,
        user_message=user_message,
        conversation_context=routing_context,
    )
    route = route_result.route
    meta_context = route_result.meta_context
    retrieval_query = route_result.query_text or user_message

    await log_event(
        db,
        org_id,
        agent_id,
        AnalyticEvent.SEMANTIC_ROUTING,
        {        
            "route": route.value,
            "meta_context": meta_context,
            "matched_tools": route_result.matched_tools,
            "contextualized_query": retrieval_query != user_message,
        },
        conversation_id=conversation_id
    )

    use_tools = route in (Route.TOOL, Route.BOTH)
    use_rag = route in (Route.RAG, Route.BOTH)

    # -- Load agent, tools, RAG --
    agent = await load_agent(db, agent_id, org_id)
    tool_rows = await load_tools(db, agent_id) if use_tools else []
    tool_map = {t.name: t for t in tool_rows}
    tools = [
        ToolSchema(
            name=t.name,
            description=t.description,
            parameters_schema=t.parameters_schema,
        )
        for t in tool_rows
    ]

    retrieved = (
        await retrieve(db, agent_id=agent_id, query=retrieval_query) if use_rag else []
    )
    rag_hit = bool(retrieved)
    rag_context = format_context_for_prompt(retrieved) if rag_hit else ""
    
    # Debug logging for RAG retrieval
    logger.debug(
        "RAG retrieval for agent %s: query='%s' | hit=%s | chunks=%d | context_len=%d",
        agent_id, retrieval_query, rag_hit, len(retrieved), len(rag_context)
    )

    if use_rag and not rag_hit:
        await log_event(
            db,
            org_id,
            agent_id,
            AnalyticEvent.RAG_MISS,
            {
                "query": retrieval_query,
            },
            conversation_id=conversation_id
        )

    # -- Persist user message --
    user_message_id = await save_message(
        db, conversation_id, role="user", content=user_message
    )

    # -- Build provider message list --
    provider = get_provider(provider=agent.llm_provider, model=agent.llm_model)

    # Route.META: append meta_context to the user turn before formatting
    effective_user_message = user_message
    if route == Route.META and meta_context:
        # Format meta_context as readable text (not JSON) so LLM understands it
        meta_text = "\n\nHere's what I can help you with:\n\n"

        # Add tools section
        if meta_context.get("tool_names"):
            meta_text += "**Tools I can use:**\n"
            for name, desc in zip(
                meta_context["tool_names"], meta_context.get("tool_descriptions", [])
            ):
                meta_text += f"- {name}: {desc}\n"
            meta_text += "\n"

        # Add documents section
        if meta_context.get("doc_names"):
            meta_text += "**Documents I can access:**\n"
            for name, desc in zip(
                meta_context["doc_names"], meta_context.get("doc_descriptions", [])
            ):
                meta_text += f"- {name}: {desc}\n"

        effective_user_message = user_message + meta_text

    messages = provider.format_messages(
        memory=memory.messages,
        system_prompt=agent.system_prompt,
        rag_context=rag_context,
    )
    provider.append_user_message(messages, effective_user_message)

    # -- ReAct loop --
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for _ in range(MAX_ITERATIONS):

        pending_tool_calls: list[ToolCallComplete] = []
        turn_content: list[str] = []

        async for event in provider.stream(messages, tools):
            if isinstance(event, ToolCallComplete):
                pending_tool_calls.append(event)
            elif isinstance(event, ContentDelta):
                turn_content.append(event.content)
                yield event.content
            elif isinstance(event, UsageEvent):
                prompt_tokens += event.prompt_tokens                
                completion_tokens += event.completion_tokens
                total_tokens += event.total_tokens

        assistant_msg_id = user_message_id
        if turn_content:
            assistant_msg_id = await save_message(
                db,
                conversation_id,
                role="assistant",
                content="".join(turn_content),
                tokens_used=completion_tokens,
                latency_ms=int((time.monotonic() - start_ms) * 1000),
            )

        if prompt_tokens > 0:
            await update_message(
                db,
                message_id=assistant_msg_id,
                tokens_used=completion_tokens,
            )

        # -- Tool-calling iteration --
        if pending_tool_calls:
            provider.append_assistant_tool_calls(messages, pending_tool_calls)

            results: list[dict[str, Any]] = []
            for tc in pending_tool_calls:
                tool_row = tool_map.get(tc.tool_name)
                if tool_row is None:
                    result = {"error": f"Tool '{tc.tool_name}' not found"}
                    status = "failed"
                    latency = 0
                else:
                    t0 = time.monotonic()
                    tool_result = await Executor.execute(tool_row, tc.arguments)
                    latency = int((time.monotonic() - t0) * 1000)
                    result = tool_result.output
                    status = tool_result.status

                results.append(result)
                await save_tool_call(
                    db,
                    message_id=assistant_msg_id,
                    tool_id=tool_row.id if tool_row else None,
                    input_data=tc.arguments,
                    output=result,
                    status=status,
                    latency_ms=latency,
                )
                if status != "success":
                    await log_event(
                        db,
                        org_id,
                        agent_id,
                        AnalyticEvent.TOOL_FAILURE,
                        {
                            "tool_name": tc.tool_name,
                            "status": status,
                        },
                        conversation_id=conversation_id,
                    )

            provider.append_tool_results(messages, pending_tool_calls, results)
            continue

        break  # no tool calls pending → end of this turn's assistant response

    else:
        await log_event(
            db,
            org_id,
            agent_id,
            AnalyticEvent.TOOL_ITERATION_LIMIT,
            {
                "iterations": MAX_ITERATIONS,
            },
            conversation_id=conversation_id,
        )
        fallback = (
            "I wasn't able to complete this in the allowed number of steps. "
            "Try rephrasing or breaking it into smaller questions."
        )
        yield fallback
        await save_message(db, conversation_id, role="assistant", content=fallback)

    # Update conversation intent for next turn's routing
    # Skip for META routes — they return tool/doc listings, not real conversation content
    # Building intent from META responses pollutes future routing decisions
    if route != Route.META:
        # Reload memory to get the latest messages (including this turn's response)
        updated_memory = await load_memory(db, conversation_id)
        asyncio.create_task(
            update_conversation_intent(redis, conversation_id, updated_memory.messages)
        )

    await log_event(
        db,
        org_id,
        agent_id,
        AnalyticEvent.MESSAGE_TURN,
        {
            "user_msg_id": user_message_id,
            "assistant_msg_id": assistant_msg_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        conversation_id=conversation_id,
    )

    await update_conversation_stats(db, conversation_id, tokens_delta=total_tokens)


async def stream(
    db: AsyncSession,
    redis: Redis,
    agent_id: str,
    org_id: str,
    conversation_id: str,
    user_message: str,
) -> AsyncGenerator[str, None]:
    """
    Production entry point — yields string tokens for the SSE response.
    Called by app/routers/chat.py after the retriever has run.
    """
    async for token in _execute(
        db, redis, agent_id, org_id, conversation_id, user_message
    ):
        yield token
