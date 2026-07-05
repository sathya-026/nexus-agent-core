"""
app/ai/providers/fireworks.py

Fireworks AI — Chat-Completions-compatible (confirmed via their docs: same
client shape, base_url=https://api.fireworks.ai/inference/v1, model ids
follow accounts/fireworks/models/<name>).
"""

from __future__ import annotations

from app.ai.providers.chat_completions import ChatCompletionsProvider
from app.config import settings


class FireworksProvider(ChatCompletionsProvider):

    BASE_URL = "https://api.fireworks.ai/inference/v1"

    def __init__(self, model: str, supports_tool_calling: bool = True) -> None:
        """supports_tool_calling defaults True — most Fireworks-served
        instruct/chat models support function calling. get_provider()
        doesn't pass this through yet, so every factory-built instance
        gets the default until the kickoff-revealed model says otherwise."""
        super().__init__(
            model=model, api_key=settings.fireworks_api_key, base_url=self.BASE_URL
        )
        self.supports_tool_calling = supports_tool_calling