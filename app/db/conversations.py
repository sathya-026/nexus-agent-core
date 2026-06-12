
import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


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
            text("""
                INSERT INTO conversations
                    (agent_id, session_id, end_user_id, status,
                     total_tokens, message_count, started_at, last_message_at)
                VALUES
                    (:agent_id, :session_id, :end_user_id, 'active',
                     0, 0, NOW(), NOW())
                ON CONFLICT (session_id) DO NOTHING
            """),
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "end_user_id": end_user_id,
            },
        )

        result = await db.execute(
            text("SELECT id FROM conversations WHERE session_id = :sid"),
            {"sid": session_id},
        )
        await db.commit()
        return str(result.fetchone().id)

    except Exception as e:
        logger.exception(f"Error during saving conversation {str(e)}")
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
            text("""
                UPDATE conversations
                SET
                    total_tokens    = total_tokens + :tokens,
                    message_count   = message_count + 1,
                    last_message_at = NOW()
                WHERE id = :conv_id
            """),
            {"conv_id": conversation_id, "tokens": tokens_delta},
        )
        await db.commit()
    except Exception as e:
        logger.exception(f"Error during updating  conversation {str(e)}")
        await db.rollback()

