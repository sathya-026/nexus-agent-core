from typing import Optional

from app.core.error_codes import ErrorCode
from app.core.error_messages import ERROR_MESSAGES


class AppException(Exception):
    """Mirrors NestJS's AppException — same {errorCode, message} contract."""

    def __init__(self, error_code: str, status_code: int, message: Optional[str] = None):
        self.error_code = error_code
        self.status_code = status_code
        self.message = message or ERROR_MESSAGES.get(error_code, "Unexpected error")
        super().__init__(self.message)