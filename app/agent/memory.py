"""
Conversation memory layer.

Loads the last N messages for a conversation from Postgres and reconstructs
an OpenAI-compatible messages array for the planner.

All three roles are replayed into context:
    "user"      → end-user turn
    "assistant" → agent response; may have an associated tool_calls array
                  if that turn triggered tools (joined from tool_calls table)
    "tool"      → tool result; requires tool_call_id matching the assistant
                  message that triggered it

tool_call_id strategy:
    OpenAI generates ephemeral IDs like "call_abc123" per request — we never
    store those. Instead we derive a stable ID from the BIGSERIAL tool_calls.id:
        f"call_{tool_calls.id}"
    This is deterministic across DB reloads and unique per tool invocation.
"""

import json
import logging
from dataclasses import dataclass, field
from itertools import groupby

from app.ai.types import MemoryMessage, ToolCallRecord
from app.db.messages import fetch_messages

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConversationMemory:
    """
    Reconstructed conversation history for a single inference call.
    Created fresh per request — stateless, safe for horizontal scaling.
    """

    conversation_id: str
    messages: list[MemoryMessage] = field(default_factory=list)

    def to_context_string(
        self,
        max_messages: int = 8,
        max_chars_per_message: int = 500,
    ) -> str:
        """
        Compact text-only history for routing and retrieval query expansion.
        """
        lines: list[str] = []

        for msg in self.messages[-max_messages:]:
            if msg.role not in {"user", "assistant"} or not msg.content:
                continue

            content = " ".join(str(msg.content).split())
            if not content:
                continue

            if len(content) > max_chars_per_message:
                content = content[:max_chars_per_message].rstrip() + "..."

            label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{label}: {content}")

        return "\n".join(lines)

    @property
    def is_empty(self) -> bool:
        return len(self.messages) == 0

    @property
    def turn_count(self) -> int:
        """Number of complete user→assistant pairs loaded."""
        return sum(1 for m in self.messages if m.role == "user")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

async def load_memory(
    db: AsyncSession,
    conversation_id: str,
    max_messages: int = 20,
) -> ConversationMemory:
    """
    Load the last `max_messages` user/assistant rows, with tool_calls
    joined directly onto the assistant messages that triggered them.

    tool_calls.message_id → assistant message (new design).
    One assistant message with N tool calls produces N rows from the JOIN,
    so we group by sequence_number in Python after fetching.
    """
    rows = []
    try:
        rows = await fetch_messages(db, conversation_id, max_messages)
    except Exception as e:
        logger.exception(f"Error during inserting message: {str(e)}")

    # ------------------------------------------------------------------
    # Group rows by sequence_number.
    # A user message → exactly one row (no tool_calls JOIN match).
    # An assistant message with N tool calls → N rows, same sequence_number.
    # ------------------------------------------------------------------
    messages: list[MemoryMessage] = []

    for seq_num, group in groupby(rows, key=lambda r: r.sequence_number):
        group = list(group)
        first = group[0]

        tool_calls = [
            ToolCallRecord(
                tool_call_id=f"call_{row.tc_id}",
                tool_name=row.tool_name,
                arguments=json.dumps(row.tc_input),
                output=json.dumps(row.tc_output),
            )
            for row in group
            if row.tc_id is not None  # LEFT JOIN — user rows have no match
        ]

        messages.append(
            MemoryMessage(
                sequence_number=first.sequence_number,
                role=first.role,
                content=first.content,
                tool_calls=tool_calls,
            )
        )

    logger.debug(
        "Loaded %d messages for conversation %s (window=%d, turns=%d)",
        len(messages),
        conversation_id,
        max_messages,
        sum(1 for m in messages if m.role == "user"),
    )

    return ConversationMemory(conversation_id=conversation_id, messages=messages)


async def update_conversation_intent(
    redis,
    conversation_id: str,
    recent_messages: list[MemoryMessage],
    window_size: int = 5,
) -> None:
    """
    Build and store conversation intent from recent assistant responses.
    
    Called after each planner turn to capture the intent of the conversation
    so far. Uses a sliding window of the last N assistant messages.
    
    The resulting embedding is stored in Redis and later blended with the
    current query during routing to improve decisions for ambiguous queries.
    
    Args:
        redis: Redis connection
        conversation_id: ID of the conversation
        recent_messages: Full message history (we'll extract assistant messages)
        window_size: Number of recent assistant responses to use (default 5)
    """
    from app.agent.query_router.embedder import embed_async, blend_embeddings
    from app.agent.query_router.cache import set_conversation_intent, ConversationIntentData
    
    # Extract assistant messages (skip tools, only take content)
    assistant_messages = [
        m.content
        for m in recent_messages
        if m.role == "assistant" and m.content
    ]
    
    # Use only recent turns to avoid stale intent
    window = assistant_messages[-window_size:]
    
    if not window:
        logger.debug(
            "No assistant messages to build conversation intent for %s",
            conversation_id
        )
        return
    
    try:
        # Embed recent assistant responses
        embeddings = await embed_async(window)
        
        # Blend with uniform weights to get an average intent
        conversation_embedding = blend_embeddings(
            embeddings,
            weights=[1.0 / len(embeddings)] * len(embeddings),
        )
        
        data: ConversationIntentData = {
            "conversation_embedding": conversation_embedding,
            "turn_count": len(window),
        }
        
        await set_conversation_intent(redis, conversation_id, data)
        
        logger.debug(
            "Updated conversation intent for %s with %d recent turn(s)",
            conversation_id, len(window)
        )
    except Exception as e:
        logger.warning(
            "Failed to update conversation intent for %s: %s",
            conversation_id, str(e)
        )
