from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Setting(BaseSettings):
    # App
    port: int = 8000
    environment: str = "development"

    frontend_url: str = "http://localhost:5173"

    # Database
    database_url: str

    #redis
    redis_url: str

    # OpenAI
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    # Gemini
    gemini_api_key: str

    # AWS S3
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket_name: str = "nexus-documents"

    # RAG tuning
    chunk_size: int = 400  # tokens
    chunk_overlap: int = 80  # tokens
    rag_top_k: int = 4  # chunks retrieved per query

    # Internal service auth
    nestjs_url: str = "http://localhost:3001"
    internal_secret: str = "temporary"

    tool_header_encryption_key: str
    widget_jwt_secret: str
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Single instance imported everywhere — settings are read once at startup
@lru_cache
def Settings():
    return Setting()


settings = Settings()
