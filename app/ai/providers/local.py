"""
app/ai/providers/local.py

Any self-hosted Chat-Completions-compatible server (Ollama, vLLM, llama.cpp
server). base_url comes from settings, not hardcoded — the real endpoint
depends on how the standardized scoring environment serves it, unknown
until kickoff.
"""

from __future__ import annotations

from app.ai.providers.chat_completions import ChatCompletionsProvider
from app.config import settings


class LocalProvider(ChatCompletionsProvider):

    def __init__(self, model: str, supports_tool_calling: bool = False) -> None:
        """supports_tool_calling defaults False — don't assume an unknown,
        hardware-constrained local model can reliably tool-call. Flip once
        the kickoff-revealed model is confirmed to support it.

        Known risk, not fixed here: some local serving stacks may reject
        stream_options.include_usage as an unrecognized field — verify
        against the actual target stack once chosen. If it's rejected, the
        fallback is computing usage client-side via the same tiktoken
        tokenizer model_router already uses, not changed preemptively."""
        super().__init__(
            model=model,
            api_key=settings.local_model_api_key or "not-needed",
            base_url=settings.local_model_base_url,
        )
        self.supports_tool_calling = supports_tool_calling