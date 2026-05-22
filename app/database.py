"""
Database connection for the agent-core.

We use SQLAlchemy's async engine (via asyncpg driver) so that DB calls
don't block FastAPI's event loop. Every await db.execute(...) yields
control back while Postgres is thinking — letting other requests run.

We do NOT use SQLAlchemy ORM models here. The NestJS side owns schema
migrations. Here we talk to the DB using raw SQL via text() — it's
simpler, easier to debug, and the queries are straightforward enough
that an ORM adds no value.

pgvector: The `vector` type is not a native Postgres type from SQLAlchemy's
perspective. We register it at connection time so asyncpg knows how to
serialize/deserialize Python lists ↔ Postgres vector columns.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.engine.url import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text
from pgvector.asyncpg import register_vector

from app.config import settings
from sqlalchemy import event

logger = logging.getLogger(__name__)

# ── Engine ───────────────────────────────────────────────────────────────────
#
# pool_size=10: up to 10 persistent connections
# max_overflow=20: up to 20 extra connections when the pool is full
# pool_pre_ping=True: test connections before using them (handles DB restarts)

def new_async_engine(uri: URL) -> AsyncEngine:
    return create_async_engine(
        uri,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30.0,
        pool_recycle=600,
    )

_ASYNC_ENGINE = new_async_engine(settings.database_url)
_ASYNC_SESSIONMAKER = async_sessionmaker(_ASYNC_ENGINE, expire_on_commit=False)


@event.listens_for(_ASYNC_ENGINE.sync_engine, "connect")
def register_vector_event(dbapi_connection, connection_record):
    dbapi_connection.run_async(register_vector)


# ── Dependency ────────────────────────────────────────────────────────────────
#
# FastAPI routes declare `db: AsyncSession = Depends(get_db)`.
# This function yields a session, runs the route, then commits or rolls back.


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    session = _ASYNC_SESSIONMAKER()    
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise
    finally:
        await session.close()


# ── Health check ──────────────────────────────────────────────────────────────


async def check_db_connection() -> bool:
    try:
        session = _ASYNC_SESSIONMAKER() 
        await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
