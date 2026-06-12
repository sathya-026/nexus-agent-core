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

class AIProviderType(str, Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
