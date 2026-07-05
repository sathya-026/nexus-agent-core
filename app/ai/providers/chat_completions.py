"""
app/ai/providers/chat_completions.py

Generic Chat-Completions-wire-format provider. Any vendor whose API is
Chat-Completions-compatible (Fireworks, most local serving stacks — Ollama,
vLLM, llama.cpp server) subclasses this with just base_url/api_key/model,
inheriting message building, tool-call accumulation, and usage extraction
untouched.

NOT related to OpenAIProvider — that file implements the Responses API
(client.responses.create), a different, OpenAI-proprietary wire format.
This class exists because Chat Completions, not Responses, is the actual
interop surface every other vendor targets.
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


class ChatCompletionsProvider(AIProvider):
    """base_url/api_key are constructor args, not settings lookups — keeps
    this class vendor-agnostic. Subclasses own reading the right settings
    field and supplying defaults."""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # ------------------------------------------------------------------
    # Message transformation — Chat Completions wire format
    # ------------------------------------------------------------------

    def format_messages(
        self,
        memory: list[MemoryMessage],
        system_prompt: str,
        rag_context: str,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._build_system_prompt(system_prompt, rag_context),
            }
        ]
        for mem in memory:
            messages.extend(self._memory_message_to_chat(mem))
        return messages

    def append_user_message(
        self,
        messages: list[dict[str, Any]],
        content: str,
    ) -> None:
        messages.append({"role": "user", "content": content})

    def append_assistant_tool_calls(
        self,
        messages: list[dict[str, Any]],
        tool_calls: list[ToolCallComplete],
    ) -> None:
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

    def append_tool_results(
        self,
        messages: list[dict[str, Any]],
        tool_calls: list[ToolCallComplete],
        results: list[dict[str, Any]],
    ) -> None:
        for tc, result in zip(tool_calls, results):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.call_id,
                    "content": json.dumps(result),
                }
            )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema],
    ) -> AsyncGenerator[AIEvent, None]:
        chat_tools = [self._tool_schema_to_chat(t) for t in tools] if tools else None

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if chat_tools:
            call_kwargs["tools"] = chat_tools
            call_kwargs["tool_choice"] = "auto"

        # Keyed by index — Chat Completions streams tool_calls by position
        # in the batch; id/name only arrive on the first delta for that index.
        accumulated: dict[int, dict[str, Any]] = {}

        response = await self._client.chat.completions.create(**call_kwargs)

        async for chunk in response:
            if not chunk.choices:
                # Final usage-only chunk (stream_options.include_usage=True)
                if chunk.usage:
                    yield UsageEvent(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )
                continue

            delta = chunk.choices[0].delta

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    slot = accumulated.setdefault(
                        tc.index, {"id": None, "name": None, "arguments": ""}
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
            elif delta.content:
                yield ContentDelta(content=delta.content)

            if chunk.choices[0].finish_reason == "tool_calls":
                for index in sorted(accumulated):
                    slot = accumulated[index]
                    try:
                        arguments = json.loads(slot["arguments"])
                    except json.JSONDecodeError:
                        arguments = {"raw": slot["arguments"]}
                    yield ToolCallComplete(
                        index=index,
                        call_id=slot["id"],
                        tool_name=slot["name"],
                        arguments=arguments,
                    )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, system_prompt: str, rag_context: str) -> str:
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

    def _memory_message_to_chat(self, mem: MemoryMessage) -> list[dict[str, Any]]:
        if mem.role == "user":
            return [{"role": "user", "content": mem.content}]

        if mem.role == "assistant":
            if not mem.tool_calls:
                return [{"role": "assistant", "content": mem.content}]

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": mem.content or None,
                "tool_calls": [
                    {
                        "id": tc.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in mem.tool_calls
                ],
            }
            tool_result_msgs = [
                {"role": "tool", "tool_call_id": tc.tool_call_id, "content": tc.output}
                for tc in mem.tool_calls
            ]
            return [assistant_msg, *tool_result_msgs]

        return []

    def _tool_schema_to_chat(self, tool: ToolSchema) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            },
        }