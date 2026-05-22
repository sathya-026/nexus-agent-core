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
from dataclasses import dataclass, field

import httpx
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
    output: dict          # always a dict — normalised from whatever the tool returns
    status: str           # "success" | "failed" | "timeout"


# ---------------------------------------------------------------------------
# Header decryption
# ---------------------------------------------------------------------------

def _decrypt_headers(encrypted_headers: dict) -> dict:
    """
    Decrypt AES-256-GCM encrypted header values.

    Storage format per header value:
        { "iv": "<base64>", "ciphertext": "<base64>" }

    The NestJS ToolsModule encrypts on write using the same key and format.
    Plain-text header names are stored as-is; only values are encrypted.
    """
    if not encrypted_headers:
        return {}

    key = base64.b64decode(settings.tool_header_encryption_key)  # 32 bytes
    aesgcm = AESGCM(key)

    decrypted = {}
    for header_name, payload in encrypted_headers.items():
        try:
            iv         = base64.b64decode(payload["iv"])
            ciphertext = base64.b64decode(payload["ciphertext"])
            plaintext  = aesgcm.decrypt(iv, ciphertext, None)
            decrypted[header_name] = plaintext.decode()
        except Exception as exc:
            # Log the header name but never the value — it may be an API key.
            logger.error("Failed to decrypt header '%s': %s", header_name, exc)
            raise RuntimeError(f"Header decryption failed for '{header_name}'") from exc

    return decrypted


# ---------------------------------------------------------------------------
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
                    tool.name, response.status_code, response.text[:200],
                )
                return ToolResult(
                    output={
                        "error":       f"HTTP {response.status_code}",
                        "detail":      response.text[:500],
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