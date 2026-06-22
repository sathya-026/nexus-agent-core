"""
app/ai/openai.py

OpenAI implementation of AIProvider.

Responsibilities absorbed from the old planner.py:
  - Building the system prompt (persona + RAG context + guardrails)
  - Converting MemoryMessages to OpenAI message dicts, including
    reconstructing role=tool turns from ToolCallRecord data
  - Streaming from the OpenAI SDK
  - Accumulating character-by-character tool call deltas into
    complete ToolCallComplete events before yielding them
  - Extracting usage from the final chunk and emitting UsageEvent

The planner never imports openai directly.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI


from app.ai.base import AIProvider
from app.ai.types import (
    AIEvent,
    ContentDelta,
    MemoryMessage,
    ToolCallComplete,
    ToolSchema,
    UsageEvent,
)
from app.config import settings


class OpenAIProvider(AIProvider):

    def __init__(self, model: str) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    # ------------------------------------------------------------------
    # Message transformation
    # ------------------------------------------------------------------

    def format_messages(
        self,
        memory: list[MemoryMessage],
        system_prompt: str,
        rag_context: str,
    ) -> list[dict[str, Any]]:
        """
        Build the full OpenAI messages list from neutral inputs.

        Layout:
          [0]  system   — persona + RAG context + guardrails
          [1…] history  — converted from MemoryMessage list
                          assistant tool-calling turns are followed immediately
                          by their role=tool result turns, synthesised from
                          ToolCallRecord.output (never stored as DB rows)
        """
        messages: list[dict[str, Any]] = []
        messages.append(
            {
                "role": "system",
                "content": self._build_system_prompt(system_prompt, rag_context),
            }
        )

        for mem in memory:
            messages.extend(self._memory_message_to_openai(mem))

        return messages

    def append_user_message(
        self,
        messages: list[dict[str, Any]],
        content: str,
    ) -> None:
        messages.append({"role": "user", "content": content})

    def append_assistant_tool_calls(
        self,
        messages: list[Any],
        tool_calls: list[ToolCallComplete],
    ) -> None:
        for tc in tool_calls:
            messages.append(
                {
                    "type": "function_call",
                    "call_id": tc.call_id,
                    "name": tc.tool_name,
                    "arguments": json.dumps(tc.arguments),
                }
            )

    def append_tool_results(
        self,
        messages: list[Any],
        tool_calls: list[ToolCallComplete],
        results: list[dict[str, Any]],
    ) -> None:
        for tc, result in zip(tool_calls, results):
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": json.dumps(result),
                }
            )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[Any],
        tools: list[ToolSchema],
    ) -> AsyncGenerator[AIEvent, None]:
        openai_tools = (
            [self._tool_schema_to_openai(t) for t in tools] if tools else None
        )

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "input": messages,
            "stream": True,
        }
        if openai_tools:
            call_kwargs["tools"] = openai_tools
            call_kwargs["tool_choice"] = "auto"

        # accumulated_tool_calls keyed by item.id (string), not integer index
        accumulated_tool_calls: dict[str, dict[str, Any]] = {}
        has_tool_calls = False

        response_stream = await self._client.responses.create(**call_kwargs)

        async for event in response_stream:
            event_type = event.type

            # -- Text delta --
            if event_type == "response.output_text.delta":
                if not has_tool_calls:
                    yield ContentDelta(content=event.delta)

            # -- Function call created (name + id arrive here) --
            elif event_type == "response.output_item.added":
                if event.item.type == "function_call":
                    has_tool_calls = True
                    accumulated_tool_calls[event.item.id] = {
                        "id": event.item.id,
                        "name": event.item.name,
                        "arguments": "",
                    }

            # -- Argument streaming --
            elif event_type == "response.function_call_arguments.delta":
                has_tool_calls = True
                item_id = event.item_id
                if item_id not in accumulated_tool_calls:
                    # defensive — item.added should always precede this
                    accumulated_tool_calls[item_id] = {
                        "id": item_id,
                        "name": None,
                        "arguments": "",
                    }
                accumulated_tool_calls[item_id]["arguments"] += event.delta

            # -- Function call finalised (authoritative name + arguments) --
            elif event_type == "response.output_item.done":
                if event.item.type == "function_call":
                    has_tool_calls = True
                    slot = accumulated_tool_calls.get(event.item.id, {})
                    slot["name"] = event.item.name
                    slot["arguments"] = event.item.arguments  # complete, not streamed
                    accumulated_tool_calls[event.item.id] = slot

                    try:
                        arguments = json.loads(slot["arguments"])
                    except json.JSONDecodeError:
                        arguments = {"raw": slot["arguments"]}

                    yield ToolCallComplete(
                        index=len(accumulated_tool_calls) - 1,  # 0-based position
                        call_id=slot["id"],
                        tool_name=slot["name"],
                        arguments=arguments,
                    )

            # -- Usage + completion --
            elif event_type == "response.completed":
                if event.response.usage:
                    u = event.response.usage
                    yield UsageEvent(
                        prompt_tokens=u.input_tokens,
                        completion_tokens=u.output_tokens,
                        total_tokens=u.total_tokens,
                    )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, system_prompt: str, rag_context: str) -> str:
        """
        Compose the full system message from four sections:
          1. Agent persona / instructions
          2. RAG knowledge base context (omitted on RAG miss)
          3. Formatting instructions
          4. Behavioural guardrails
        """
        sections: list[str] = [system_prompt.strip()]
        if rag_context:
            sections.append(f"## Knowledge Base Context\n{rag_context}")
        sections.append(self.FORMATTING_INSTRUCTIONS)
        sections.append(self._guardrails())

        return "\n\n".join(sections)

    def _guardrails(self) -> str:
        return (
            "## Instructions\n"
            f"Today's date is {date.today().isoformat()}.\n"
            "Answer only from the knowledge base context when it is relevant. "
            "If relevant information is present, provide a complete answer without citing the context. "
            "If the context does not contain the answer, say so clearly — do not fabricate. "
            "Use the available tools when they can provide a better answer. "
            "Be concise and helpful."
        )

    def _memory_message_to_openai(self, mem: MemoryMessage) -> list[dict[str, Any]]:
        if mem.role == "user":
            return [{"role": "user", "content": mem.content}]

        if mem.role == "assistant":
            if not mem.tool_calls:
                return [{"role": "assistant", "content": mem.content}]

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": mem.content or None,  # Responses API: None if empty
                "tool_calls": [
                    {
                        "id": tc.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": tc.arguments,  # already a JSON string
                        },
                    }
                    for tc in mem.tool_calls
                ],
            }

            tool_result_msgs: list[dict[str, Any]] = [
                {
                    "role": "tool",
                    "tool_call_id": tc.tool_call_id,
                    "content": tc.output,  # already a JSON string
                }
                for tc in mem.tool_calls
            ]

            return [assistant_msg, *tool_result_msgs]

        return []

    def _tool_schema_to_openai(self, tool: ToolSchema) -> dict[str, Any]:
        """Convert neutral ToolSchema to OpenAI function-calling format."""
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        }