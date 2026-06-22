"""
app/ai/factory.py

Single entry point for obtaining an AIProvider instance.
The planner imports only this function — never a concrete provider class.

Adding a new provider:
  1. Implement AIProvider in app/ai/<provider_name>.py
  2. Add its import and a branch in get_provider() below
  3. No other file needs to change
"""

from __future__ import annotations

from app.ai.base import AIProvider
from app.common.constants import AIProviderType


def get_provider(provider: str, model: str) -> AIProvider:
    """
    Return the AIProvider implementation for the given provider string.

    Parameters
    ----------
    provider:
        The agents.llm_provider value — e.g. "openai", "anthropic".
        Matched case-insensitively.
    model:
        The agents.llm_model value — passed through to the provider
        so one implementation can serve multiple models (e.g. gpt-4o,
        gpt-4o-mini) without separate classes.

    Raises
    ------
    ValueError
        If the provider string is not recognised. This surfaces as a 500
        on the chat endpoint, which is appropriate — it indicates a
        misconfigured agent record that should never reach production.
    """
    match provider.lower():
        case AIProviderType.OPENAI:
            from app.ai.providers.openai import OpenAIProvider
            return OpenAIProvider(model=model)

        case AIProviderType.GEMINI:
            from app.ai.providers.gemini import GeminiProvider
            return GeminiProvider(model=model)

        case _:
            raise ValueError(
                f"Unknown AI provider '{provider}'. "
                f"Supported providers: openai, gemini"
            )