import logging
import traceback
from functools import wraps
from fastapi import HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def catch_and_log_exceptions(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request: Request = kwargs.get("request")

        try:
            return await func(*args, **kwargs)

        except Exception as e:
            logger.exception(
                f"Error in {request.method if request else ''} "
                f"{request.url if request else ''}"
            )

            raise HTTPException(
                status_code=500,
                detail="Internal Server Error"
            )

    return wrapper