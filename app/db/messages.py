import logging

from app.ai.types import MemoryMessage
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def save_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    tokens_used: int = None,
    latency_ms: int = None,
) -> int:
    """
    Persist one message. sequence_number is assigned by the Postgres
    BEFORE INSERT trigger — never set by application code.

    Returns the trigger-assigned sequence_number.
    """
    try:
        result = await db.execute(
            text("""
                INSERT INTO messages
                    (conversation_id, role, content, tokens_used, latency_ms)
                VALUES
                    (:conv_id, :role, :content, :tokens_used, :latency_ms)
                RETURNING *
            """),
            {
                "conv_id": conversation_id,
                "role": role,
                "content": content,
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
            },
        )
        row = result.fetchone()
        await db.commit()
        return row.id
    except Exception as e:
        logger.exception(f"Error during inserting message {str(e)}")
        await db.rollback()

async def update_message(
    db: AsyncSession,
    message_id: int,
    tokens_used: int = None,
    latency_ms: int = None,
) -> int:
    """
    Persist one message. sequence_number is assigned by the Postgres
    BEFORE INSERT trigger — never set by application code.

    Returns the trigger-assigned sequence_number.
    """
    try:
        result = await db.execute(
            text("""
                UPDATE messages
                SET tokens_used = :tokens_used, latency_ms = :latency_ms
                WHERE id = :message_id
                RETURNING *
            """),
            {
                "message_id": message_id,
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
            },
        )
        row = result.fetchone()
        await db.commit()
        return row.id
    except Exception as e:
        logger.exception(f"Error during updating message {str(e)}")
        await db.rollback()


async def fetch_messages(
    db: AsyncSession, conversation_id: str, max_messages: int = 10
) -> list[MemoryMessage]:
    """
    Fetch messages with their tool calls, ordered by sequence_number ASC.

    The SQL query returns one row per tool call, so assistant messages with
    multiple tool calls will have multiple rows with the same sequence_number.
     - A user message → exactly one row (no tool_calls JOIN match).
     - An assistant message with N tool calls → N rows, same sequence_number.

    Group rows by sequence_number to reconstruct the original messages with
    their associated tool calls.

    Note: This function is synchronous and expects pre-fetched rows as input.
    """
    try:
        result = await db.execute(
            text("""
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
            """),
            {"conv_id": conversation_id, "lim": max_messages},
        )
        return result.fetchall()
    except Exception as e:
        logger.exception(f"Error during inserting message: {str(e)}")
        raise
