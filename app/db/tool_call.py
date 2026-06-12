import json
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


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
            text("""
                INSERT INTO tool_calls
                    (message_id, tool_id, input, output, status, latency_ms)
                VALUES
                    (:message_id, :tool_id, :input, :output, :status, :latency_ms)
                RETURNING id
            """),
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
