from app.core.error_codes import ErrorCode

ERROR_MESSAGES: dict[str, str] = {
    ErrorCode.AGENT_NOT_FOUND: "Agent not found",
    ErrorCode.AUTH_INTERNAL_SECRET_INVALID: "Invalid internal secret",
    ErrorCode.AGENT_CORE_LLM_TIMEOUT: "Language model timed out",
    ErrorCode.AGENT_CORE_LLM_UNAVAILABLE: "Language model unavailable",
    ErrorCode.VALIDATION_FAILED: "Validation failed",
    ErrorCode.RESOURCE_NOT_FOUND: "Resource not found",
    ErrorCode.RESOURCE_CONFLICT: "Resource already exists",
    ErrorCode.INTERNAL_UNEXPECTED: "Unexpected error",
}