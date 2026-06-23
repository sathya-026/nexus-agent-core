import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import check_db_connection, _ASYNC_ENGINE
from app.routers import indexing, chat

# from app.agent.query_router import warmup_model
from app.core.redis import create_redis_client

# ── Lifespan ──────────────────────────────────────────────────────────────────
#
# Code before `yield` runs at startup, code after runs at shutdown.
# This is where you'd warm up connection pools, load models into memory, etc.

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Nexus agent-core starting...")
    db_ok = await check_db_connection()
    if not db_ok:
        raise RuntimeError("Cannot connect to Postgres — check DATABASE_URL")
    print("✅ Database connected")
    print(
        f"🤖 Using models: embed={settings.embedding_model}, chat={settings.chat_model}"
    )
    print(logging.getLevelName(logger.getEffectiveLevel()))
    # Redis
    app.state.redis = create_redis_client(settings.redis_url)
    logger.info("Redis client created.")

    # MiniLM — warm up at startup, not on first request
    # warmup_model()
    yield
    # Shutdown
    await _ASYNC_ENGINE.dispose()
    await app.state.redis.aclose()
    logger.info("disposed database engine and closed connections...")
    print("👋 Nexus agent-core shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nexus Agent Core",
    description="RAG pipeline, ReAct planner, tool executor",
    version="0.1.0",
    lifespan=lifespan,
    root_path="/api",
    # Only expose docs in development
    docs_url="/docs" if settings.environment == "development" else None,
    redoc_url=None,
)

# CORS — only NestJS and the widget need to reach this service

allowed_origins = [settings.nestjs_url, settings.frontend_url]
allowed_origins.extend([origin.strip() for origin in settings.cors_allowed_origins if origin.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(indexing.router, prefix="/indexing", tags=["RAG Indexing"])
app.include_router(chat.router)


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Health"])
async def health():
    db_ok = await check_db_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
        "models": {
            "embedding": settings.embedding_model,
            "chat": settings.chat_model,
        },
    }
