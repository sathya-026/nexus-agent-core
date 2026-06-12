"""
app/ai/types.py

Neutral, provider-agnostic types that flow between memory, the planner,
and every AI provider implementation.

Two categories:
  1. MemoryMessage  — the intermediate format memory.py produces.
                      Provider implementations consume this and convert it
                      to their own wire format inside format_messages().

  2. AIEvent        — the normalised stream of events that every provider
                      yields. The planner only ever sees these; it never
                      handles raw SDK chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Neutral memory types (replaces to_openai_messages output as source of truth)
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """
    A single tool invocation attached to an assistant message.
    tool_call_id is derived as f'call_{tool_calls.id}' — stable from BIGSERIAL.
    arguments and output are JSON strings, matching OpenAI wire format and DB storage.
    """
    tool_call_id: str        # "call_{tool_calls.id}"
    tool_name: str
    arguments: str           # JSON string — e.g. '{"city": "London"}'
    output: str              # JSON string — tool_calls.output serialised


@dataclass
class MemoryMessage:
    sequence_number: int
    role: str                # "user" | "assistant"
    content: str             # empty string for tool-only assistant turns (not None)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Neutral tool schema (provider format_messages() converts this to wire format)
# ---------------------------------------------------------------------------

@dataclass
class ToolSchema:
    """
    Provider-agnostic tool definition, built from the tools table row.
    Each provider converts this to its own function-calling format.
    """
    name: str
    description: str
    parameters_schema: dict[str, Any]    # JSON Schema — already stored in this format


# ---------------------------------------------------------------------------
# Normalised AI stream events
# ---------------------------------------------------------------------------

@dataclass
class ContentDelta:
    """
    A fragment of the final text response.
    Yielded during the last ReAct iteration only — the planner forwards
    these directly to the SSE stream.
    """
    content: str


@dataclass
class ToolCallComplete:
    """
    A fully accumulated tool call, ready to execute.

    The provider is responsible for accumulating character-by-character
    streaming deltas into a complete call before yielding this event.
    The planner never sees partial tool calls.

    index identifies position in a parallel tool-call batch (0-based).
    call_id is the provider's ephemeral ID (e.g. OpenAI's "call_abc123").
      — stored in tool_calls.id as BIGSERIAL; the provider ID is not persisted.
    """
    index: int
    call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class UsageEvent:
    """
    Token consumption for the completed LLM call.
    Yielded once, in the final chunk of a streaming response.
    prompt_tokens and completion_tokens are provided for granular logging;
    total_tokens is what update_conversation_stats() increments.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# AIEvent is the union type the planner iterates over.
AIEvent = ContentDelta | ToolCallComplete | UsageEvent