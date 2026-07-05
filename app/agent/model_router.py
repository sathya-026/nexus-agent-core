"""
app/agent/model_router.py

Decides which provider+model serves a single ReAct turn, before the loop
starts. Pure function of its inputs — no db, no org_id, no side effects —
so it works identically inside Nexus's _execute() and inside a standalone
harness that imports it directly with zero Nexus infra in the loop.

Strategy: decide once, call once. No try-local-first/escalate path — that
double-spends tokens on every escalation, and it's unconfirmed whether
local-model tokens are excluded from external token-count scoring. Safer
to assume they count and never double-generate.

Routing signal, in priority order:
  1. Tool-calling requirement is a hard gate, not a heuristic. If this turn
     needs tools and the candidate local model can't reliably call them,
     it's remote — full stop.
  2. Otherwise, token count of (user_message + rag_context) vs. a
     threshold. rag_context is included because it's often the larger of
     the two and is already computed by retrieve() before routing happens.
     Conversation history length is NOT counted yet — known gap, revisit
     once real tasks show what actually matters.

Local-model timeout/connectivity fallback to remote is NOT handled here —
that's a runtime concern, not a decision concern, and lives in _execute()
as a wrap-and-swap around provider.stream(). Keeping it out of this module
means choose() stays a pure function, and the fallback can be "sticky" for
the rest of a turn without this module needing to know what a turn is.
"""

from __future__ import annotations

import tiktoken

from app.ai.types import ModelChoice

def _supports_tool_calling(provider: str, model: str) -> bool:
    """Real capability check — constructs the actual provider and reads its
    supports_tool_calling flag (set per-model in __init__, since capability
    depends on the configured model, not just the vendor). Construction is
    cheap — an AsyncOpenAI client just stores config, no network call — so
    this isn't worth avoiding via a separate lookup table."""
    from app.ai.factory import get_provider

    return get_provider(provider=provider, model=model).supports_tool_calling


# Calibrate once real tasks are revealed at kickoff. Conservative default —
# short, contextless turns stay local; anything with real bulk goes remote.
_LOCAL_TOKEN_THRESHOLD = 200

_encoding = tiktoken.get_encoding("cl100k_base")  # same tokenizer as chunker.py


def _count_tokens(text: str) -> int:
    return len(_encoding.encode(text)) if text else 0


def choose(
    *,
    user_message: str,
    rag_context: str = "",
    use_tools: bool,
    local_provider: str = "local",
    local_model: str,
    remote_provider: str = "fireworks",
    remote_model: str,
) -> ModelChoice:
    """
    Decide provider+model for one ReAct turn.

    user_message / rag_context are token-counted together — deliberately
    NOT the full provider-formatted `messages` list, since building that
    requires a provider to already be chosen.

    local_provider/local_model, remote_provider/remote_model are passed in
    rather than read from config here, so a standalone harness can supply
    kickoff-revealed models without touching this file.
    """
    if use_tools and not _supports_tool_calling(local_provider, local_model):
        return ModelChoice(
            provider=remote_provider,
            model=remote_model,
            reason="tool_calling_required",
        )

    token_count = _count_tokens(user_message) + _count_tokens(rag_context)

    if token_count <= _LOCAL_TOKEN_THRESHOLD:
        return ModelChoice(
            provider=local_provider,
            model=local_model,
            reason=f"short_turn:{token_count}_tokens",
        )

    return ModelChoice(
        provider=remote_provider,
        model=remote_model,
        reason=f"long_turn:{token_count}_tokens",
    )