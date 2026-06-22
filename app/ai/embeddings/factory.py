"""
app/ai/embeddings/factory.py

Single entry point for obtaining an EmbeddingProvider instance.
Mirrors app/ai/factory.py's pattern exactly, but is intentionally a
separate function — conflating chat-provider and embedding-provider
selection would force every future chat provider to also implement
embeddings, which isn't true (e.g. a provider might do chat only).

Adding a new embedding provider:
  1. Implement EmbeddingProvider in app/ai/embeddings/<provider_name>.py
  2. Add its import and a branch in get_embedding_provider() below
  3. No other file needs to change
"""

from __future__ import annotations

from app.ai.embeddings.base import EmbeddingProvider


def get_embedding_provider(provider: str) -> EmbeddingProvider:
    """
    Return the EmbeddingProvider implementation for the given provider string.

    Parameters
    ----------
    provider:
        "openai" or "gemini". Matched case-insensitively.

        Unlike get_provider() for chat, there's no per-agent model string
        threaded through here yet — each provider currently hardcodes its
        model (text-embedding-3-small for OpenAI, gemini-embedding-001 for
        Gemini). If/when RAG embeddings become agent-configurable, this
        signature will need a model parameter the same way chat's does —
        deferred until that migration is actually being built.

    Raises
    ------
    ValueError
        If the provider string is not recognised.
    """
    match provider.lower():
        case "openai":
            from app.ai.embeddings.providers.openai import OpenAIEmbeddingProvider
            return OpenAIEmbeddingProvider()

        case "gemini":
            from app.ai.embeddings.providers.gemini import GeminiEmbeddingProvider
            return GeminiEmbeddingProvider()

        case _:
            raise ValueError(
                f"Unknown embedding provider '{provider}'. "
                f"Supported providers: openai, gemini"
            )