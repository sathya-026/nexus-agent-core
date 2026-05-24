# app/agent/tool_executor.py
"""
Tool executor — makes the outbound HTTP call for a single tool invocation.

Responsibilities:
    - Decrypt tool headers (AES-256-GCM, encrypted at rest in the tools table)
    - Build and fire the HTTP request with the LLM-supplied arguments
    - Enforce a per-call timeout
    - Normalise every outcome (success, HTTP error, timeout, unexpected) into
      a ToolResult so the planner never has to handle raw exceptions

Explicitly NOT responsible for:
    - DB persistence  (planner calls save_tool_call after execute() returns)
    - Retries         (tools are user-configured webhooks; retrying blindly
                       could cause duplicate side-effects e.g. placing two orders)
    - Analytics       (planner logs failures via _log_event)
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64

from app.config import settings

logger = logging.getLogger(__name__)

TOOL_TIMEOUT_SECONDS = 10


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    output: dict  # always a dict — normalised from whatever the tool returns
    status: str  # "success" | "failed" | "timeout"


# ---------------------------------------------------------------------------
# Header decryption
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

class DecryptionError(Exception):
    """Raised when encrypted payload cannot be decrypted."""


def _decrypt_headers(stored: dict[str, Any]) -> dict[str, Any]:
    """
    Decrypts data stored in the format:
    base64(iv):base64(tag):base64(ciphertext)

    Returns the original dictionary if '__encrypted' is not present.
    Raises DecryptionError for malformed or invalid encrypted payloads.
    """
    encryption_key = bytes.fromhex(settings.tool_header_encryption_key)
    if not isinstance(stored, dict):
        raise TypeError("stored must be a dictionary")

    encrypted = stored.get("__encrypted")

    if not encrypted:
        return stored

    try:
        # Validate payload structure
        parts = encrypted.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Invalid encrypted payload format. "
                "Expected 'iv:tag:ciphertext'."
            )

        iv_b64, tag_b64, enc_b64 = parts

        # Decode base64 components
        try:
            iv = base64.b64decode(iv_b64, validate=True)
            tag = base64.b64decode(tag_b64, validate=True)
            ciphertext = base64.b64decode(enc_b64, validate=True)
        except Exception as exc:
            raise ValueError("Invalid base64 encoding in encrypted payload") from exc

        # AES-GCM validation
        if len(iv) not in (12, 16):
            raise ValueError("Invalid IV length for AES-GCM")

        if len(tag) != 16:
            raise ValueError("Invalid authentication tag length")

        if len(encryption_key) not in (16, 24, 32):
            raise ValueError(
                "Invalid AES key length. "
                "Must be 16, 24, or 32 bytes."
            )

        # cryptography AESGCM expects ciphertext + tag
        encrypted_data = ciphertext + tag

        aesgcm = AESGCM(encryption_key)

        decrypted = aesgcm.decrypt(
            nonce=iv,
            data=encrypted_data,
            associated_data=None
        )

        try:
            return json.loads(decrypted.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Decryption succeeded but payload is not valid UTF-8 JSON"
            ) from exc

    except InvalidTag as exc:
        logger.warning("AES-GCM authentication failed")
        raise DecryptionError(
            "Failed to decrypt payload: authentication failed"
        ) from exc

    except Exception as exc:
        logger.exception("Decryption failed")
        raise DecryptionError(
            f"Failed to decrypt payload: {exc}"
        ) from exc# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def execute(tool, arguments: dict) -> ToolResult:
    """
    Execute a single tool call and return a normalised ToolResult.

    Args:
        tool:      _ToolRow from the planner (has endpoint_url, http_method,
                   headers_encrypted, etc.). Typed as Any to avoid a circular
                   import — tool_executor must not import from planner.
        arguments: Dict parsed from LLM's function call arguments JSON.

    Returns:
        ToolResult with .output (dict) and .status ("success"|"failed"|"timeout").
        Never raises — all failure modes are captured into ToolResult.

    HTTP method conventions:
        GET  → arguments become query parameters
        POST → arguments become the JSON request body
    """
    try:
        headers = _decrypt_headers(tool.headers_encrypted)
    except RuntimeError as exc:
        return ToolResult(output={"error": str(exc)}, status="failed")

    headers.setdefault("Content-Type", "application/json")

    try:
        async with httpx.AsyncClient(timeout=TOOL_TIMEOUT_SECONDS) as client:

            method = tool.http_method.upper()

            if method == "GET":
                response = await client.get(
                    tool.endpoint_url,
                    params=arguments,
                    headers=headers,
                )
            elif method == "POST":
                response = await client.post(
                    tool.endpoint_url,
                    json=arguments,
                    headers=headers,
                )
            else:
                # PUT / PATCH / DELETE — send body as JSON
                response = await client.request(
                    method=method,
                    url=tool.endpoint_url,
                    json=arguments,
                    headers=headers,
                )

            # ── Parse response ───────────────────────────────────────────
            # Accept any 2xx as success.
            # Non-2xx is a "failed" status — the output contains the error
            # detail so the LLM can reason about it ("The booking API returned
            # 404 — the item does not exist").
            if response.is_success:
                try:
                    output = response.json()
                    if not isinstance(output, dict):
                        # Some tools return a plain list or string — wrap it
                        # so tool_calls.output is always a JSON object.
                        output = {"result": output}
                except Exception:
                    output = {"result": response.text}

                return ToolResult(output=output, status="success")

            else:
                logger.warning(
                    "Tool '%s' returned HTTP %d: %s",
                    tool.name,
                    response.status_code,
                    response.text[:200],
                )
                return ToolResult(
                    output={
                        "error": f"HTTP {response.status_code}",
                        "detail": response.text[:500],
                    },
                    status="failed",
                )

    except httpx.TimeoutException:
        logger.warning("Tool '%s' timed out after %ds", tool.name, TOOL_TIMEOUT_SECONDS)
        return ToolResult(
            output={"error": f"Tool did not respond within {TOOL_TIMEOUT_SECONDS}s."},
            status="timeout",
        )

    except Exception as exc:
        logger.exception("Unexpected error executing tool '%s': %s", tool.name, exc)
        return ToolResult(
            output={"error": "An unexpected error occurred while calling the tool."},
            status="failed",
        )