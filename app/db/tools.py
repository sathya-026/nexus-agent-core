
import logging

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

@dataclass
class _ToolRow:
    id: int
    name: str
    description: str
    endpoint_url: str
    http_method: str
    headers_encrypted: dict
    parameters_schema: dict

async def load_tools(db: AsyncSession, agent_id: str) -> list[_ToolRow]:
    try:
        result = await db.execute(
            text("""
                SELECT id, name, description, endpoint_url,
                       http_method, headers, parameters_schema
                FROM   tools
                WHERE  agent_id = :agent_id AND is_active = true
            """),
            {"agent_id": agent_id},
        )
        return [
            _ToolRow(
                id=row.id,
                name=row.name,
                description=row.description,
                endpoint_url=row.endpoint_url,
                http_method=row.http_method,
                headers_encrypted=row.headers or {},
                parameters_schema=row.parameters_schema or {},
            )
            for row in result.fetchall()
        ]
    except Exception as e:
        logger.exception(f"Error during loading tools {str(e)}")
        await db.rollback()

