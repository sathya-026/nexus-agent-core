import json
import logging

from app.common.constants import AnalyticEvent

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def log_event(
    db: AsyncSession,
    org_id: str,
    agent_id: str,
    event_type: AnalyticEvent,
    payload: dict,
) -> None:
    try:
        await db.execute(
            text("""
                INSERT INTO analytics_events
                    (org_id, agent_id, event_type, payload)
                VALUES
                    (:org_id, :agent_id, :event_type, :payload)
            """),
            {
                "org_id": org_id,
                "agent_id": agent_id,
                "event_type": event_type,
                "payload": json.dumps(payload),
            },
        )
        await db.commit()
    except Exception as exc:
        logger.exception("Analytics logging failed (non-fatal): %s", exc)
        await db.rollback()
