
import logging

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

@dataclass
class _AgentRow:
    id: str
    org_id: str
    system_prompt: str
    llm_provider: str
    llm_model: str


async def load_agent(db: AsyncSession, agent_id: str, org_id: str) -> _AgentRow:
    try:
        result = await db.execute(
            text("""
                SELECT id, org_id, system_prompt, llm_provider, llm_model
                FROM   agents
                WHERE  id = :agent_id AND org_id = :org_id AND is_active = true
            """),
            {"agent_id": agent_id, "org_id": org_id},
        )
        row = result.fetchone()
        if not row:
            raise ValueError(f"Agent {agent_id} not found or inactive for org {org_id}")
        return _AgentRow(
            id=str(row.id),
            org_id=str(row.org_id),
            system_prompt=row.system_prompt,
            llm_provider=row.llm_provider,
            llm_model=row.llm_model,
        )
    except Exception as e:
        logger.exception(f"Error during loading agent {str(e)}")
        await db.rollback()
