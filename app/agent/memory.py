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
from typing import Optional
from itertools import groupby

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    """A single tool invocation attached to an assistant message."""

    tool_call_id: str  # "call_{tool_calls.id}" — stable derived ID
    tool_name: str
    arguments: str  # JSON string (matches OpenAI function.arguments)
    output: str  # JSON string stored in tool_calls.output


# Updated dataclass — drop tool_call_id, it was only for role="tool" rows
@dataclass
class MemoryMessage:
    sequence_number: int
    role: str
    content: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


@dataclass
class ConversationMemory:
    """
    Reconstructed conversation history for a single inference call.
    Created fresh per request — stateless, safe for horizontal scaling.
    """

    conversation_id: str
    messages: list[MemoryMessage] = field(default_factory=list)

    def to_openai_messages(self) -> list[dict]:
        """
            Serialise to OpenAI chat completions format.

            Assistant messages that triggered tools produce TWO entries each:
            1. { role: "assistant", tool_calls: [...] }
            2. { role: "tool", tool_call_id: ..., content: tc.output }
               — one per tool call, synthesised from tool_calls.output.
               Never read from a messages row; tool_calls is the source of truth.

        Assistant messages with no tool calls produce one plain entry.
        """

        result = []

        for msg in self.messages:

            if msg.role == "user":
                result.append({"role": "user", "content": msg.content})

            elif msg.role == "assistant":
                if msg.tool_calls:
                    # Part 1 — assistant's intent to call tools
                    result.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tc.tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.tool_name,
                                        "arguments": tc.arguments,
                                    },
                                }
                                for tc in msg.tool_calls
                            ],
                        }
                    )
                # Part 2 — synthesised tool results, one per call
                for tc in msg.tool_calls:
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.tool_call_id,
                            "content": tc.output,
                        }
                    )
            else:
                result.append({"role": "assistant", "content": msg.content})

        return result

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

    Tool results are NOT stored as messages rows — they are synthesised
    in to_openai_messages() from tool_calls.output at read time.
    """
    rows = []
    try:
        result = await db.execute(
            text(
                """
                SELECT
                    m.sequence_number,
                    m.role,
                    m.content,
                    tc.id        AS tc_id,
                    tc.input     AS tc_input,
                    tc.output    AS tc_output,
                    t.name       AS tool_name
                FROM (
                    SELECT id, sequence_number, role, content
                    FROM   messages
                    WHERE  conversation_id = :conv_id
                    ORDER  BY sequence_number DESC
                    LIMIT  :lim
                ) m
                LEFT JOIN tool_calls tc ON tc.message_id = m.id
                LEFT JOIN tools      t  ON t.id = tc.tool_id
                ORDER BY m.sequence_number ASC, tc.id ASC
            """
            ),
            {"conv_id": conversation_id, "lim": max_messages},
        )
        rows = result.fetchall()
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


async def save_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    tokens_used: int = 0,
    latency_ms: int = 0,
) -> int:
    """
    Persist one message. sequence_number is assigned by the Postgres
    BEFORE INSERT trigger — never set by application code.

    Returns the trigger-assigned sequence_number.
    """
    try:
        result = await db.execute(
            text(
                """
                INSERT INTO messages
                    (conversation_id, role, content, tokens_used, latency_ms)
                VALUES
                    (:conv_id, :role, :content, :tokens_used, :latency_ms)
                RETURNING sequence_number
            """
            ),
            {
                "conv_id": conversation_id,
                "role": role,
                "content": content,
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
            },
        )
        seq = result.fetchone().sequence_number
        await db.commit()
        return seq
    except Exception as e:
        logger.exception(f"Error during inserting message {str(e)}")
        await db.rollback()


async def save_tool_call(
    db: AsyncSession,
    message_id: int,
    tool_id: int,
    input_data: dict,
    output: dict,
    status: str,
    latency_ms: int,
) -> int:
    """
    Persist one tool_calls row after a tool has executed.

    message_id is the BIGSERIAL id of the role="tool" message this result
    belongs to (returned by save_message()).

    Returns the tool_calls.id (BIGSERIAL), which the planner uses to
    derive the stable tool_call_id string ("call_{id}") for the next
    turn's memory reconstruction.
    """
    try:
        result = await db.execute(
            text(
                """
                INSERT INTO tool_calls
                    (message_id, tool_id, input, output, status, latency_ms)
                VALUES
                    (:message_id, :tool_id, :input, :output, :status, :latency_ms)
                RETURNING id
            """
            ),
            {
                "message_id": message_id,
                "tool_id": tool_id,
                "input": json.dumps(input_data),
                "output": json.dumps(output),
                "status": status,
                "latency_ms": latency_ms,
            },
        )
        await db.commit()
        return result.fetchone().id
    except Exception as e:
        logger.exception(f"Error during saving tool call {str(e)}")
        await db.rollback()

async def update_conversation_stats(
    db: AsyncSession,
    conversation_id: str,
    tokens_delta: int,
) -> None:
    """
    Increment total_tokens + message_count, refresh last_message_at.
    Denormalized for O(1) dashboard queries — avoids COUNT/SUM over messages.
    """
    try:
        await db.execute(
            text(
                """
                UPDATE conversations
                SET
                    total_tokens    = total_tokens + :tokens,
                    message_count   = message_count + 1,
                    last_message_at = NOW()
                WHERE id = :conv_id
            """
            ),
            {"conv_id": conversation_id, "tokens": tokens_delta},
        )
        await db.commit()
    except Exception as e:
        logger.exception(f"Error during updating  conversation {str(e)}")
        await db.rollback()


async def get_or_create_conversation(
    db: AsyncSession,
    agent_id: str,
    session_id: str,
    end_user_id: Optional[str] = None,
) -> str:
    """
    Return the conversation_id for (agent_id, session_id), creating the row
    if this is the widget's first message in this browser session.

    ON CONFLICT DO NOTHING handles the race where two tab-duplicated requests
    arrive simultaneously — only one INSERT wins, both then SELECT the winner.
    """
    try:
        await db.execute(
            text(
                """
                INSERT INTO conversations
                    (agent_id, session_id, end_user_id, status,
                     total_tokens, message_count, started_at, last_message_at)
                VALUES
                    (:agent_id, :session_id, :end_user_id, 'active',
                     0, 0, NOW(), NOW())
                ON CONFLICT (session_id) DO NOTHING
            """
            ),
            {"agent_id": agent_id, "session_id": session_id, "end_user_id": end_user_id},
        )

        result = await db.execute(
            text("SELECT id FROM conversations WHERE session_id = :sid"),
            {"sid": session_id},
        )
        return str(result.fetchone().id)

        await db.commit()
    except Exception as e:
        logger.exception(f"Error during saving conversation {str(e)}")
        await db.rollback()    
