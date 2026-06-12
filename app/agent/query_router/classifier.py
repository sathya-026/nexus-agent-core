"""
app/routing/classifier.py
-------------------------
Pure heuristics for classifying user messages before embedding.

All functions are stateless and have no I/O — they read only the
message string (and optional conversation context). This makes them
trivially unit-testable in isolation.

Design notes
------------
_is_contextual_follow_up() covers ONLY pronoun/deictic references
("it", "that", "above", "the previous one"). It deliberately does NOT
include _is_information_request(). Merging the two would make the
DOC_INFORMATION_REQUEST_THRESHOLD gate in router.py dead code — any
information request would unconditionally trigger RAG regardless of
the document similarity score.
"""

from __future__ import annotations

import re

# ── Meta intent utterances ────────────────────────────────────────────────────
# Matched via embedding similarity in router.py; defined here because they
# are a content/heuristic decision, not embedding infrastructure.
#
# Two groups:
#   General capabilities — "what can you do?"
#   Resource listing     — "what documents do you have?"
#
# The resource listing group exists because queries about available documents
# or tools sit in a different embedding region from general capability questions.
# Without them, "what documents do you have?" scores ~0.29 against general
# capability utterances — well below the 0.75 threshold.
#
# The is_resource_listing_query() regex below handles informal phrasing
# ("what documents u have?") that still might score below threshold even
# with these additions. Both mechanisms are complementary.

META_INTENTS: list[str] = [
    # General capabilities
    "what can you do",
    "what are your capabilities",
    "how can you help me",
    "what are you able to do",
    "show me what you can do",
    "list your features",
    "what are your functions",
    "what kind of tasks can you handle",
    "give me an overview of your abilities",
    "what are you good at",
    # Document / knowledge base listing
    "what documents do you have",
    "what files do you have",
    "what pdfs do you have",
    "list your documents",
    "show me your documents",
    "what documents are available",
    "what files are available",
    "do you have any documents",
    "what is in your knowledge base",
    "what knowledge base documents are there",
    # Tool listing
    "what tools do you have",
    "what tools are available",
    "list your tools",
    "what apis do you have access to",
    "what actions can you perform",
]

# Strict — meta intent must be unambiguous before we short-circuit to META route
INTENT_SIMILARITY_THRESHOLD = 0.75


# ── Short-circuit word sets ───────────────────────────────────────────────────
# Normalised to lowercase, punctuation stripped before comparison.

ACKNOWLEDGEMENT_MESSAGES: frozenset[str] = frozenset({
    "ok", "okay", "k", "kk", "alright", "all right", "got it",
    "thanks", "thank you", "thx", "cool", "fine", "sure",
    "yes", "yep", "no", "nope",
})

SMALLTALK_MESSAGES: frozenset[str] = frozenset({
    "hi", "hello", "hey",
    "good morning", "good afternoon", "good evening",
    "how are you", "how are you doing",
    "whats up", "what's up",
})


# ── Regex patterns ────────────────────────────────────────────────────────────

QUESTION_RE = re.compile(
    r"\b(who|what|when|where|why|how|which|whose|whom|"
    r"can|could|would|should|do|does|did|is|are|was|were|"
    r"will|has|have|had)\b",
    re.IGNORECASE,
)

INFO_REQUEST_RE = re.compile(
    r"\b(tell me|explain|describe|summari[sz]e|list|show|give me|"
    r"find|search|look up|retrieve|compare)\b",
    re.IGNORECASE,
)

# Pronoun/deictic references only — not general information requests.
# See module docstring for why this is intentionally narrow.
CONTEXT_REFERENCE_RE = re.compile(
    r"\b(he|him|his|she|her|hers|they|them|their|theirs|it|its|"
    r"this|that|these|those|there|same|above|previous|earlier|"
    r"former|latter)\b",
    re.IGNORECASE,
)

# Matches resource-listing queries: "what documents do you have?", "list your files",
# "show me your pdfs", "do you have any tools?", "any knowledge base documents?"
# Excludes content queries via negative lookahead: "what document covers X?"
#
# Patterns:
#   A — listing verbs ("list", "show") + resource noun
#   B — "what/which" + resource noun + word boundary, NOT followed by a content word
#       Word boundary AFTER s? is required to prevent backtracking where the engine
#       drops the 's' from "documents" to bypass the lookahead (e.g. "documents about"
#       → tries s?=0, lands at 's' which is not 'about' → incorrectly passes).
#   C — "do you have" + resource noun
#   D — "any" directly before a resource noun
RESOURCE_LISTING_RE = re.compile(
    r"""
    (?:
        \b(?:list|show)\b .{0,30} \b(?:documents?|files?|pdfs?|tools?|apis?|knowledge(?:\s+base)?|resources?)\b
      |
        \b(?:what|which)\b \s* (?:documents?|files?|pdfs?|tools?|apis?|resources?|knowledge(?:\s+base)?)\b
        (?!\s*(?:about|on\s|cover|contain|regarding|discuss|explain|for\s))
      |
        \bdo\s+you\s+have\b .{0,30} \b(?:documents?|files?|pdfs?|tools?|apis?|knowledge|resources?)\b
      |
        \bany\b \s+ (?:documents?|files?|pdfs?|tools?|apis?|resources?)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation (except apostrophes), collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


# ── Classifiers ───────────────────────────────────────────────────────────────

def is_acknowledgement_or_smalltalk(message: str) -> bool:
    """
    True for empty messages, single-word acknowledgements ("ok", "thanks"),
    and common greetings ("hi", "good morning").
    These carry no retrieval intent and can short-circuit the router entirely.
    """
    normalised = _normalise(message)
    if not normalised:
        return True
    return normalised in ACKNOWLEDGEMENT_MESSAGES or normalised in SMALLTALK_MESSAGES


def is_information_request(message: str) -> bool:
    """
    True if the message contains a question word, a question mark, or
    an explicit information-seeking verb ("explain", "tell me", "find", …).
    Used in router.py with a lower similarity threshold (DOC_INFORMATION_REQUEST_THRESHOLD)
    to catch borderline RAG cases that the main threshold would miss.
    """
    text = message.strip()
    if not text or is_acknowledgement_or_smalltalk(text):
        return False
    return bool(
        "?" in text
        or QUESTION_RE.search(text)
        or INFO_REQUEST_RE.search(text)
    )


def is_contextual_follow_up(message: str) -> bool:
    """
    True if the message references a prior topic via a pronoun or deictic
    ("it", "that", "the previous one", "above", etc.).

    Deliberately narrow — does NOT include is_information_request().
    The router handles information requests with a score gate; this function
    handles the case where the score gate wouldn't fire (the message itself
    is short and ambiguous) but context makes the topic clear.
    """
    return bool(CONTEXT_REFERENCE_RE.search(message))


def is_resource_listing_query(message: str) -> bool:
    """
    True if the message is asking what documents, files, or tools the agent has.

    Runs as a regex fast-path in router.py before embedding, which makes it
    immune to informal phrasing that reduces similarity scores. "What documents
    u have?" scores ~0.29 against the META_INTENTS embeddings because 'u'
    instead of 'you' shifts the embedding enough to miss the 0.75 threshold.
    This function catches it unconditionally.

    Complements (does not replace) the expanded META_INTENTS — embedding
    handles natural variations; regex handles informal/abbreviated phrasing.
    """
    return bool(RESOURCE_LISTING_RE.search(message.strip()))


def build_routing_query(user_message: str, conversation_context: str | None) -> str:
    """
    Produce the query string to embed for retrieval scoring.

    For contextual follow-ups ("what about it?"), prepends the recent
    conversation so the embedding captures the actual topic being discussed.
    For all other messages, returns the raw user message unchanged.
    """
    message = user_message.strip()
    if conversation_context and is_contextual_follow_up(message):
        return (
            "Recent conversation:\n"
            f"{conversation_context}\n\n"
            f"Current user request: {message}"
        )
    return message