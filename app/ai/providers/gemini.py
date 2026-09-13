"""
app/ai/gemini.py

Gemini implementation of AIProvider, using the google-genai SDK
(client.aio.models.generate_content_stream).

Key differences from OpenAIProvider that shape this implementation:

  1. Roles are "user" and "model", not "user"/"assistant"/"tool".
     Tool results are sent back as a "user"-role Content containing a
     function_response Part — there is no separate "tool" role.

  2. Function calls arrive WHOLE in a single chunk's Part, never streamed
     character-by-character. No delta accumulation is needed — each
     function_call Part is immediately a complete ToolCallComplete.

  3. The system prompt is NOT part of the contents list. It is a separate
     GenerateContentConfig.system_instruction parameter. Because base.py's
     format_messages() must still return one opaque object the planner
     passes straight to stream(), this provider returns a small dict
     wrapper: {"contents": [...], "system_instruction": "..."}.
     stream() unpacks it. The planner never inspects this — it stays opaque.

  4. Function call correlation uses an `id` field on both the function_call
     and function_response parts (not a string like OpenAI's "call_xyz").
     We still use ToolCallComplete.call_id for this — Gemini's id is a
     string token, same type as call_id, so no extra mapping is needed.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, AsyncGenerator

from google import genai
from google.genai import types as gtypes

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


class GeminiProvider(AIProvider):

    def __init__(self, model: str) -> None:
        self._model = model
        self._client = genai.Client(api_key=settings.gemini_api_key)

    # ------------------------------------------------------------------
    # Message transformation
    # ------------------------------------------------------------------

    def format_messages(
        self,
        memory: list[MemoryMessage],
        system_prompt: str,
        rag_context: str,
    ) -> dict[str, Any]:
        """
        Returns {"contents": list[gtypes.Content], "system_instruction": str}.

        This dict is opaque to the planner — it is only ever passed back
        into append_*() and stream() on this same provider instance.
        """
        contents: list[gtypes.Content] = []
        for mem in memory:
            contents.extend(self._memory_message_to_gemini(mem))

        return {
            "contents": contents,
            "system_instruction": self._build_system_prompt(system_prompt, rag_context),
        }

    def append_user_message(
        self,
        messages: dict[str, Any],
        content: str,
    ) -> None:
        messages["contents"].append(
            gtypes.Content(role="user", parts=[gtypes.Part.from_text(text=content)])
        )

    def append_assistant_tool_calls(
        self,
        messages: dict[str, Any],
        tool_calls: list[ToolCallComplete],
    ) -> None:
        """
        Append the model's tool-calling turn as a single Content(role="model")
        with one function_call Part per tool — mirrors how Gemini itself
        emits parallel calls within one turn.
        """
        messages["contents"].append(
            gtypes.Content(
                role="model",
                parts=[
                    gtypes.Part.from_function_call(
                        name=tc.tool_name,
                        args=tc.arguments,
                    )
                    for tc in tool_calls
                ],
            )
        )

    def append_tool_results(
        self,
        messages: dict[str, Any],
        tool_calls: list[ToolCallComplete],
        results: list[dict[str, Any]],
    ) -> None:
        """
        Tool results go back as role="user" Content containing
        function_response Parts. id correlates each response to its
        call, matching the id Gemini issued on the function_call Part.
        """
        messages["contents"].append(
            gtypes.Content(
                role="user",
                parts=[
                    gtypes.Part.from_function_response(
                        name=tc.tool_name,
                        response=result,
                        id=tc.call_id,
                    )
                    for tc, result in zip(tool_calls, results)
                ],
            )
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: dict[str, Any],
        tools: list[ToolSchema],
    ) -> AsyncGenerator[AIEvent, None]:
        """
        Stream from Gemini and yield normalised AIEvents.

        Unlike OpenAI, function_call Parts arrive complete — no accumulation
        across chunks is needed. Each chunk's candidate may contain a mix of
        text Parts and function_call Parts (though in practice a turn is
        either all-text or all-function-calls).

        Usage arrives on chunk.usage_metadata, present on multiple chunks
        as the running total — we only emit UsageEvent once, after the
        stream completes, using the last seen value.
        """
        gemini_tools = (
            [
                gtypes.Tool(
                    function_declarations=[
                        self._tool_schema_to_gemini(t) for t in tools
                    ]
                )
            ]
            if tools
            else None
        )

        config = gtypes.GenerateContentConfig(
            system_instruction=messages["system_instruction"],
            tools=gemini_tools,
        )

        last_usage: gtypes.GenerateContentResponseUsageMetadata | None = None

        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=messages["contents"],
            config=config,
        )

        async for chunk in stream:
            if chunk.usage_metadata:
                last_usage = chunk.usage_metadata

            if not chunk.candidates:
                continue

            parts = (
                chunk.candidates[0].content.parts if chunk.candidates[0].content else []
            )
            if not parts:
                continue

            for part in parts:
                # -- Complete function call, no accumulation needed --
                if part.function_call:
                    fc = part.function_call
                    yield ToolCallComplete(
                        index=0,  # Gemini doesn't expose ordinal position; id is authoritative
                        call_id=fc.id or fc.name,  # fall back to name if id is absent
                        tool_name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    )

                # -- Text delta --
                elif part.text:
                    yield ContentDelta(content=part.text)

        if last_usage:
            yield UsageEvent(
                prompt_tokens=last_usage.prompt_token_count or 0,
                completion_tokens=last_usage.candidates_token_count or 0,
                total_tokens=last_usage.total_token_count or 0,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, system_prompt: str, rag_context: str) -> str:
        """Identical composition to OpenAIProvider — kept in sync intentionally."""
        sections: list[str] = [system_prompt.strip()]

        if rag_context:
            sections.append(f"## Knowledge Base Context\n{rag_context}")

        sections.append(self._guardrails())
        return "\n\n".join(sections)

    def _guardrails(self) -> str:
        return (
            "## Instructions\n"
            f"Today's date is {date.today().isoformat()}.\n"
            "Answer only from the knowledge base context when it is relevant. "
            "If the context does not contain the answer, say so clearly — do not fabricate. "
            "Use the available tools when they can provide a better answer. "
            "Be concise and helpful."
        )

    def _memory_message_to_gemini(
        self,
        mem: MemoryMessage,
    ) -> list[gtypes.Content]:
        """
        Convert one neutral MemoryMessage into one or more gtypes.Content.

        user turn            -> Content(role="user")
        assistant, no tools  -> Content(role="model")
        assistant with tools -> Content(role="model", parts=[function_call, ...])
                                 + Content(role="user", parts=[function_response, ...])
                                 reconstructed from ToolCallRecord.output
        """
        if not mem.tool_calls:
            return [
                gtypes.Content(
                    role="model" if mem.role == "assistant" else "user",
                    parts=[gtypes.Part.from_text(text=mem.content)],
                )
            ]

        model_turn = gtypes.Content(
            role="model",
            parts=[
                gtypes.Part.from_function_call(
                    name=tc.tool_name,
                    args=json.loads(tc.arguments),  # stored as JSON string
                )
                for tc in mem.tool_calls
            ],
        )

        tool_result_turn = gtypes.Content(
            role="user",
            parts=[
                gtypes.Part.from_function_response(
                    name=tc.tool_name,
                    response=json.loads(tc.output),  # stored as JSON string
                    id=tc.tool_call_id,
                )
                for tc in mem.tool_calls
            ],
        )

        return [model_turn, tool_result_turn]

    def _tool_schema_to_gemini(self, tool: ToolSchema) -> gtypes.FunctionDeclaration:
        """Convert neutral ToolSchema to Gemini's FunctionDeclaration format."""
        return gtypes.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters_schema,
        )
