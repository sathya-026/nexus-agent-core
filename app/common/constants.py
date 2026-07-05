from enum import Enum

class AnalyticEvent(str, Enum):
    SEMANTIC_ROUTING = "semantic routing"
    RAG_HIT = "rag hit"
    RAG_MISS = "rag miss"
    TOOL_CALL = "tool call"
    TOOL_FAILURE = "tool failure"
    TOOL_ITERATION_LIMIT = "tool iteration limit reached"
    DOCUMENT_INDEXED = "document indexed"
    DOCUMENT_INDEXING_FAILED = "document indexing failed"
    MODEL_ROUTING = "model routing"
    LOCAL_MODEL_FALLBACK = "local model fallback"

class AIProviderType(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    AZURE_OPENAI = "azure_openai"
    FIREWORKS = "fireworks"
    LOCAL = "local"
