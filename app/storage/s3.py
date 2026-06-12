"""
S3 client — async wrapper around boto3.

boto3 is synchronous. All public functions here offload blocking I/O to a
thread pool via run_in_executor so callers get a clean await interface without
stalling FastAPI's event loop.

The boto3 client is a module-level singleton, initialised once on import and
shared across all calls. boto3 clients are thread-safe for concurrent reads.
"""

import asyncio
import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings

logger = logging.getLogger(__name__)


class S3Error(RuntimeError):
    """
    Raised when an S3 operation fails.

    Wraps boto3's BotoCoreError / ClientError so callers don't need to import
    boto3 just to catch storage failures.
    """


# ── Client singleton ──────────────────────────────────────────────────────────

_s3 = boto3.client(
    "s3",
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
)


# ── Public API ────────────────────────────────────────────────────────────────

async def download_file(s3_key: str) -> bytes:
    """
    Download a file from S3 and return its raw bytes.

    Args:
        s3_key: The S3 object key to fetch.

    Returns:
        Raw file bytes.

    Raises:
        S3Error: If the download fails for any boto3 reason (key not found,
                 credentials invalid, network error, etc.).
    """
    loop = asyncio.get_running_loop()

    def _sync_download() -> bytes:
        try:
            response = _s3.get_object(Bucket=settings.s3_bucket_name, Key=s3_key)
            return response["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            raise S3Error(f"S3 download failed for key '{s3_key}': {exc}") from exc

    return await loop.run_in_executor(None, _sync_download)