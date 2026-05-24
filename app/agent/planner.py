# app/agent/planner.py
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

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator

from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.retrieval.retriever import retrieve, format_context_for_prompt
from app.agent.memory import (
    load_memory,
    save_message,
    save_tool_call,
    update_conversation_stats,
)
from app.agent import tool_executor

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8
MODEL = "gpt-4o-mini"


def _get_client():
    _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


# ---------------------------------------------------------------------------
# Internal data classes
# ---------------------------------------------------------------------------


@dataclass
class _AgentRow:
    id: str
    org_id: str
    system_prompt: str


@dataclass
class _ToolRow:
    id: int
    name: str
    description: str
    endpoint_url: str
    http_method: str
    headers_encrypted: dict
    parameters_schema: dict


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
# Private helpers
# ---------------------------------------------------------------------------


async def _load_agent(db: AsyncSession, agent_id: str, org_id: str) -> _AgentRow:
    try:
        result = await db.execute(
            text("""
                SELECT id, org_id, system_prompt
                FROM   agents
                WHERE  id = :agent_id AND org_id = :org_id AND is_active = true
            """),
            {"agent_id": agent_id, "org_id": org_id},
        )
        row = result.fetchone()
        if not row:
            raise ValueError(f"Agent {agent_id} not found or inactive for org {org_id}")
        return _AgentRow(
            id=str(row.id), org_id=str(row.org_id), system_prompt=row.system_prompt
        )
    except Exception as e:
        logger.exception(f"Error during loading agent {str(e)}")
        await db.rollback()


async def _load_tools(db: AsyncSession, agent_id: str) -> list[_ToolRow]:
    try:
        result = await db.execute(
            text("""
                SELECT id, name, description, endpoint_url,
                       http_method, headers, parameters_schema
                FROM   tools
                WHERE  agent_id = :agent_id AND is_active = true
            """),
            {"agent_id": agent_id},
        )
        return [
            _ToolRow(
                id=row.id,
                name=row.name,
                description=row.description,
                endpoint_url=row.endpoint_url,
                http_method=row.http_method,
                headers_encrypted=row.headers or {},
                parameters_schema=row.parameters_schema or {},
            )
            for row in result.fetchall()
        ]
    except Exception as e:
        logger.exception(f"Error during loading tools {str(e)}")
        await db.rollback()


def _to_openai_tools(tools: list[_ToolRow]) -> list[dict]:
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        }
        for tool in tools
    ]


def _build_system_prompt(agent: _AgentRow, rag_context: str, has_tools: bool) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    context_section = (
        f"## Knowledge Base Context\n{rag_context}"
        if rag_context
        else "## Knowledge Base Context\n"
        "No relevant documents found. Answer from general knowledge if appropriate, "
        "or inform the user you don't have information on this topic."
    )

    guardrails = "\n".join(
        filter(
            None,
            [
                "## Instructions",
                f"- Today's date is {today}.",
                "- Prioritise the Knowledge Base Context when relevant.",
                "- Do not fabricate facts. If unsure, say so.",
                (
                    (
                        "- Use tools when you need real-time data or actions the knowledge base cannot satisfy.\n"
                        "- After a tool returns, incorporate its result — do not call the same tool twice "
                        "with identical arguments."
                    )
                    if has_tools
                    else None
                ),
            ],
        )
    )

    return "\n\n".join([agent.system_prompt.strip(), context_section, guardrails])


async def _log_event(
    db: AsyncSession,
    org_id: str,
    agent_id: str,
    conversation_id: str,
    event_type: str,
    payload: dict,
) -> None:
    try:
        await db.execute(
            text("""
                INSERT INTO analytics_events
                    (org_id, agent_id, conversation_id, event_type, payload)
                VALUES
                    (:org_id, :agent_id, :conv_id, :event_type, :payload)
            """),
            {
                "org_id": org_id,
                "agent_id": agent_id,
                "conv_id": conversation_id,
                "event_type": event_type,
                "payload": json.dumps(payload),
            },
        )
        await db.commit()
    except Exception as exc:
        logger.exception("Analytics logging failed (non-fatal): %s", exc)
        await db.rollback()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run(
    db: AsyncSession,
    agent_id: str,
    org_id: str,
    conversation_id: str,
    user_message: str,
) -> PlannerResult:
    """
    Execute one complete user turn and return the agent's final response.

    ReAct loop behaviour:
        Each iteration calls the LLM.
        If the LLM responds with tool_calls:
            - Persist an assistant message (content=None) and get its DB id.
            - Execute each requested tool.
            - Persist a tool_calls row per execution, linked to that assistant id.
            - Append the assistant message + tool results to working_messages.
            - Loop.
        If the LLM responds with plain text:
            - That is the final answer. Break.

    Only user messages and assistant messages are written to the messages table.
    Tool inputs/outputs live in tool_calls and are joined back at read time.
    """
    turn_start = int(time.monotonic() * 1000)

    # ------------------------------------------------------------------
    # 1. Load agent config and tools
    # ------------------------------------------------------------------
    agent = await _load_agent(db, agent_id, org_id)
    tools = await _load_tools(db, agent_id)
    tools_by_name = {t.name: t for t in tools}
    openai_tools = _to_openai_tools(tools) if tools else []

    # ------------------------------------------------------------------
    # 2. RAG retrieval
    # ------------------------------------------------------------------
    retrieved = await retrieve(db, agent_id=agent_id, query=user_message)
    rag_hit = bool(retrieved)
    rag_context = format_context_for_prompt(retrieved) if rag_hit else ""

    if not rag_hit:
        await _log_event(
            db,
            org_id,
            agent_id,
            conversation_id,
            event_type="RAG miss",
            payload={"query": user_message},
        )

    # ------------------------------------------------------------------
    # 3. Load memory (past user + assistant pairs)
    # ------------------------------------------------------------------
    memory = await load_memory(db, conversation_id)

    # ------------------------------------------------------------------
    # 4. Persist the incoming user message
    # ------------------------------------------------------------------
    seq = await save_message(
        db,
        conversation_id=conversation_id,
        role="user",
        content=user_message,
    )
    # ------------------------------------------------------------------
    # 5. Assemble initial working messages list
    # ------------------------------------------------------------------
    system_prompt = _build_system_prompt(agent, rag_context, has_tools=bool(tools))

    working: list[dict] = [
        {"role": "system", "content": system_prompt},
        *memory.to_openai_messages(),
        {"role": "user", "content": user_message},
    ]

    # ------------------------------------------------------------------
    # 6. ReAct loop
    # ------------------------------------------------------------------
    total_tokens = 0
    iterations = 0
    final_response = ""

    for iteration in range(MAX_ITERATIONS):
        iterations = iteration + 1
        logger.debug(
            "ReAct iteration %d/%d | conversation=%s",
            iterations,
            MAX_ITERATIONS,
            conversation_id,
        )

        call_kwargs: dict = {"model": MODEL, "messages": working}
        if openai_tools:
            call_kwargs["tools"] = openai_tools
            call_kwargs["tool_choice"] = "auto"
        _client = _get_client()
        completion = await _client.chat.completions.create(**call_kwargs)
        choice = completion.choices[0]
        total_tokens += completion.usage.total_tokens if completion.usage else 0

        # ── No tool calls → final answer ────────────────────────────────
        if not choice.message.tool_calls:
            final_response = choice.message.content or ""
            break

        # ── Tool calls requested ─────────────────────────────────────────
        # Persist the assistant's tool-calling message first so we have
        # its DB id to link tool_calls rows against.
        # content is None — this turn's value is in tool_calls.input/output.
        assistant_msg_id, _ = await save_message(
            db,
            conversation_id=conversation_id,
            role="assistant",
            content=None,
        )

        # Append to working list so the LLM sees it in the next iteration.
        working.append(choice.message.model_dump(exclude_unset=True))

        for tc in choice.message.tool_calls:
            tool_name = tc.function.name
            tool_config = tools_by_name.get(tool_name)

            # ── Unknown / hallucinated tool name ────────────────────────
            if not tool_config:
                logger.warning("LLM requested unknown tool '%s' — skipping", tool_name)
                working.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            {"error": f"Tool '{tool_name}' does not exist."}
                        ),
                    }
                )
                continue

            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            # ── Execute ─────────────────────────────────────────────────
            exec_start = int(time.monotonic() * 1000)
            exec_result = await tool_executor.execute(
                tool=tool_config, arguments=arguments
            )
            exec_latency = int(time.monotonic() * 1000) - exec_start

            # ── Persist tool_calls row linked to the assistant message ───
            # No role="tool" message written — tool_calls.output is the
            # source of truth, joined back by load_memory() at read time.
            await save_tool_call(
                db,
                message_id=assistant_msg_id,  # ← assistant, not a tool message
                tool_id=tool_config.id,
                input_data=arguments,
                output=exec_result.output,
                status=exec_result.status,
                latency_ms=exec_latency,
            )

            if exec_result.status != "success":
                await _log_event(
                    db,
                    org_id,
                    agent_id,
                    conversation_id,
                    event_type="tool failure",
                    payload={
                        "tool": tool_name,
                        "status": exec_result.status,
                        "arguments": arguments,
                    },
                )

            # ── Append tool result to working list ───────────────────────
            # tc.id is OpenAI's ephemeral call ID, valid only within this
            # turn's working list. Unrelated to our DB tool_calls.id.
            working.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(exec_result.output),
                }
            )

        # ── Hit ceiling without a clean final answer ─────────────────────
        if iterations == MAX_ITERATIONS:
            logger.error(
                "ReAct hit MAX_ITERATIONS (%d) for conversation %s",
                MAX_ITERATIONS,
                conversation_id,
            )
            await _log_event(
                db,
                org_id,
                agent_id,
                conversation_id,
                event_type="max iterations reached",
                payload={"iterations": MAX_ITERATIONS, "query": user_message},
            )
            final_response = (
                "I wasn't able to complete this in the allowed number of steps. "
                "Try rephrasing or breaking it into smaller questions."
            )

    # ------------------------------------------------------------------
    # 7. Persist final assistant response
    # ------------------------------------------------------------------
    turn_latency = int(time.monotonic() * 1000) - turn_start

    await save_message(
        db,
        conversation_id=conversation_id,
        role="assistant",
        content=final_response,
        tokens_used=total_tokens,
        latency_ms=turn_latency,
    )

    await update_conversation_stats(db, conversation_id, tokens_delta=total_tokens)

    return PlannerResult(
        response=final_response,
        tokens_used=total_tokens,
        iterations=iterations,
        rag_hit=rag_hit,
        latency_ms=turn_latency,
    )


# stream() alongside the existing run()
async def stream(
    db: AsyncSession,
    agent_id: str,
    org_id: str,
    conversation_id: str,
    user_message: str,
) -> AsyncGenerator[str, None]:
    """
    Streaming variant of run(). Yields string tokens as they arrive from
    the LLM on the final ReAct iteration.

    Tool-calling iterations are handled silently — the caller sees nothing
    until the agent has finished reasoning and begins its final response.

    Token-by-token yielding lets the widget render the response as it
    streams rather than waiting for the full completion.

    Usage data is only available in the last chunk of a streaming response.
    We request it explicitly via stream_options={"include_usage": True}.
    This is an OpenAI-specific option — adjust if switching providers.
    """

    # ------------------------------------------------------------------
    # Setup — identical to run()
    # ------------------------------------------------------------------
    agent = await _load_agent(db, agent_id, org_id)
    tools = await _load_tools(db, agent_id)
    tools_by_name = {t.name: t for t in tools}
    openai_tools = _to_openai_tools(tools) if tools else []

    retrieved = await retrieve(db, agent_id=agent_id, query=user_message)
    rag_hit = bool(retrieved)
    rag_context = format_context_for_prompt(retrieved) if rag_hit else ""

    if not rag_hit:
        await _log_event(
            db,
            org_id,
            agent_id,
            conversation_id,
            "RAG miss",
            {"query": user_message},
        )

    memory = await load_memory(db, conversation_id)

    user_message_id = await save_message(
        db,
        conversation_id=conversation_id,
        role="user",
        content=user_message,
    )

    system_prompt = _build_system_prompt(agent, rag_context, has_tools=bool(tools))

    working: list[dict] = [
        {"role": "system", "content": system_prompt},
        *memory.to_openai_messages(),
        {"role": "user", "content": user_message},
    ]

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------
    total_tokens = 0
    iterations = 0
    final_response = ""

    for iteration in range(MAX_ITERATIONS):
        iterations = iteration + 1
        logger.debug("ReAct stream iteration %d/%d", iterations, MAX_ITERATIONS)

        call_kwargs: dict = {
            "model": MODEL,
            "input": working,
            "stream": True,
        }
        if openai_tools:
            call_kwargs["tools"] = openai_tools
            call_kwargs["tool_choice"] = "auto"

        _client = _get_client()
        start = int(time.monotonic() * 1000)
        response_stream = await _client.responses.create(**call_kwargs)
        
        # Per-iteration accumulators
        accumulated_content = ""
        # Tool calls arrive as indexed chunks — build them up by index.
        # { 0: { id, type, function: { name, arguments } }, 1: { ... }, ... }
        accumulated_tool_calls: dict[int, dict] = {}
        has_tool_calls = False

        async for event in response_stream:

            event_type = event.type
            # ─────────────────────────────────────────────
            # TEXT STREAMING
            # ─────────────────────────────────────────────
            if event_type == "response.output_text.delta":

                delta = event.delta
                # logger.exception(f"TEXT: {delta}")
                if not has_tool_calls:
                    accumulated_content += delta
                    yield delta

            # ─────────────────────────────────────────────
            # FUNCTION CALL CREATED
            # ─────────────────────────────────────────────
            elif event_type == "response.output_item.added":

                item = event.item

                if item.type == "function_call":

                    has_tool_calls = True

                    accumulated_tool_calls[item.id] = {
                        "id": item.id,
                        "name": item.name,
                        "arguments": "",
                    }
                # logger.exception(f"FUNCTION CALL CREATED: {str(accumulated_tool_calls)}")

            # ─────────────────────────────────────────────
            # FUNCTION CALL ARGUMENT STREAMING
            # ─────────────────────────────────────────────
            elif event_type == "response.function_call_arguments.delta":
                has_tool_calls = True
                item_id = event.item_id

                if item_id not in accumulated_tool_calls:
                    accumulated_tool_calls[item_id] = {
                        "id": item_id,
                        "name": None,
                        "arguments": "",
                    }

                accumulated_tool_calls[item_id]["arguments"] += event.delta
                # logger.exception(f"FUNCTION CALL ARG STREAM: {str(accumulated_tool_calls)}")
            # ─────────────────────────────────────────────
            # FUNCTION CALL FINALIZED
            # ─────────────────────────────────────────────
            elif event_type == "response.output_item.done":

                item = event.item

                if item.type == "function_call":

                    has_tool_calls = True

                    accumulated_tool_calls[item.id] = {
                        "id": item.id,
                        "name": item.name,
                        "arguments": item.arguments,
                    }
                # logger.exception(f"FUNCTION CALL FINALIZED: {str(accumulated_tool_calls)}")
            # ─────────────────────────────────────────────
            # USAGE
            # ─────────────────────────────────────────────

            elif event_type == "response.completed":
                if event.response.usage:
                    total_tokens += event.response.usage.total_tokens
                end = int(time.monotonic() * 1000)

        # ─────────────────────────────────────────────
        # END ITERATION
        # ─────────────────────────────────────────────
        assistant_msg_id = user_message_id
        if accumulated_content:
            assistant_msg_id = await save_message(
                db,
                conversation_id=conversation_id,
                role="assistant",
                content=accumulated_content,
                tokens_used=total_tokens,
                latency_ms=(end - start)
            )

        if not has_tool_calls:
            final_response = accumulated_content
            break

        # ─────────────────────────────────────────────
        # EXECUTE TOOL CALLS
        # ─────────────────────────────────────────────

        tool_calls_list = list(accumulated_tool_calls.values())

        # Add assistant tool calls to context
        for tc in tool_calls_list:
            working.append(
                {
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                }
            )

        # Execute tools
        for tc in tool_calls_list:
            tool_name = tc["name"]
            tool_config = tools_by_name.get(tool_name)
            if not tool_config:
                logger.warning(
                    "LLM requested unknown tool '%s'",
                    tool_name,
                )
                working.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc["id"],
                        "output": json.dumps(
                            {"error": (f"Tool '{tool_name}' not found.")}
                        ),
                    }
                )
                continue

            try:
                arguments = json.loads(tc["arguments"])

            except json.JSONDecodeError:
                arguments = {}
            exec_start = int(time.monotonic() * 1000)
            logger.exception(f"FUNCTION CALL: {str(tool_config)} {arguments}")
            exec_result = await tool_executor.execute(
                tool=tool_config,
                arguments=arguments,
            )
            exec_latency = int(time.monotonic() * 1000) - exec_start
            await save_tool_call(
                db,
                message_id=assistant_msg_id,
                tool_id=tool_config.id,
                input_data=arguments,
                output=exec_result.output,
                status=exec_result.status,
                latency_ms=exec_latency,
            )
            if exec_result.status != "success":

                await _log_event(
                    db,
                    org_id,
                    agent_id,
                    conversation_id,
                    "tool failure",
                    {
                        "tool": tool_name,
                        "status": exec_result.status,
                    },
                )

            working.append(
                {
                    "type": "function_call_output",
                    "call_id": tc["id"],
                    "output": json.dumps(exec_result.output),
                }
            )

    # ─────────────────────────────────────────────
    # MAX ITERATIONS
    # ─────────────────────────────────────────────

    if iterations == MAX_ITERATIONS:

        logger.error(
            "stream() hit MAX_ITERATIONS (%d) " "for conversation %s",
            MAX_ITERATIONS,
            conversation_id,
        )

        await _log_event(
            db,
            org_id,
            agent_id,
            conversation_id,
            "max iterations reached",
            {"iterations": MAX_ITERATIONS},
        )

        final_response = (
            "I wasn't able to complete this in the "
            "allowed number of steps. Try rephrasing "
            "or breaking it into smaller questions."
        )

        yield final_response
