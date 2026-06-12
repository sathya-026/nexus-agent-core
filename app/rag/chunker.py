"""
RAG Step 1 — Chunker

Splits raw document text into overlapping, token-bounded chunks that
are then passed to the embedder.

Key decisions explained inline:
  - Why tiktoken, not len(): token counts must match what the embedding
    model sees. len() wildly over/undercounts.
  - Why sentence snapping: cutting mid-sentence degrades retrieval quality.
    A chunk ending in "The return policy is 30 da" matches nothing.
  - Why overlap: answers often span chunk boundaries. Overlap ensures
    boundary-straddling context appears in at least one complete chunk.
"""

import re
from dataclasses import dataclass

import tiktoken

from app.config import settings


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    content: str  # The actual text this chunk contains
    chunk_index: int  # Position in the document (0-based)
    token_count: int  # Exact token count (useful for billing estimates)
    char_start: int  # Character offset in original text (for debugging)
    char_end: int


# ── Tokenizer setup ───────────────────────────────────────────────────────────
#
# tiktoken is the same tokenizer OpenAI uses internally. We load it once at
# module level — it's a ~1MB vocabulary file, not something to reload per doc.
#
# "cl100k_base" is the encoding for:
#   - text-embedding-3-small  (our embedding model)
#   - text-embedding-ada-002
#   - gpt-4, gpt-4o, gpt-3.5-turbo
# If you ever switch model families, update this encoding string too.

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the exact number of tokens in a string."""
    return len(_tokenizer.encode(text))


# ── Sentence splitting ────────────────────────────────────────────────────────
#
# We split on sentence boundaries so chunks always end on a complete thought.
#
# The regex uses a lookbehind (?<=...) — it matches whitespace that is
# PRECEDED by sentence-ending punctuation, without consuming the punctuation.
# Result: the punctuation stays attached to the sentence before the split.
#
#   "Hello world. How are you?" → ["Hello world.", "How are you?"]
#
# [\"\'\)]? handles closing quotes/parens: e.g. 'He said "done." Next...'
#
# This is intentionally simple. A production system might use spaCy for better
# handling of abbreviations ("Dr. Smith"), but this covers >95% of real docs.

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])(?:["\')])?\s+')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping punctuation with its sentence."""
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ── Core chunker ──────────────────────────────────────────────────────────────


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Split `text` into overlapping token-bounded chunks.

    Args:
        text:          Raw document text (already extracted from PDF/docx/etc.)
        chunk_size:    Target token count per chunk. Defaults to settings.chunk_size.
        chunk_overlap: Token overlap between consecutive chunks.
                       Defaults to settings.chunk_overlap.

    Returns:
        List of Chunk objects in document order.

    Algorithm:
        1. Split the full text into sentences.
        2. Greedily add sentences to the current chunk until adding the next
           sentence would exceed chunk_size.
        3. When the chunk is full, record it and start a new chunk seeded
           with the last `chunk_overlap` tokens of the current chunk.
        4. Repeat until all sentences are consumed.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not text or not text.strip():
        return []

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_token_count = 0
    char_cursor = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # Edge case: a single sentence exceeds chunk_size.
        # We can't split sentences without breaking meaning, so we allow the
        # oversized chunk. This is rare but happens with long tables or code blocks.
        if sentence_tokens > chunk_size and not current_sentences:
            chunk_content = sentence
            chunks.append(
                Chunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    token_count=sentence_tokens,
                    char_start=char_cursor,
                    char_end=char_cursor + len(chunk_content),
                )
            )
            char_cursor += len(chunk_content) + 1
            chunk_index += 1
            continue

        # Would adding this sentence push us over the limit?
        joining_space = 1 if current_sentences else 0
        projected_tokens = current_token_count + sentence_tokens + joining_space

        if projected_tokens > chunk_size and current_sentences:
            # ── Flush the current chunk ──────────────────────────────────────
            chunk_content = " ".join(current_sentences)
            char_end = char_cursor + len(chunk_content)

            chunks.append(
                Chunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    token_count=current_token_count,
                    char_start=char_cursor,
                    char_end=char_end,
                )
            )
            chunk_index += 1

            # ── Build the overlap window ─────────────────────────────────────
            # Walk backwards through the just-flushed sentences, collecting
            # them until we hit the overlap token budget.
            # These sentences become the START of the next chunk — giving it
            # context about what came just before the boundary.
            overlap_sentences: list[str] = []
            overlap_tokens = 0

            for s in reversed(current_sentences):
                s_tokens = count_tokens(s)
                if overlap_tokens + s_tokens > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens

            current_sentences = overlap_sentences
            current_token_count = overlap_tokens
            char_cursor = char_end + 1  # Advance past the flushed chunk

        current_sentences.append(sentence)
        current_token_count += sentence_tokens + (
            1 if len(current_sentences) > 1 else 0
        )

    # Flush the final chunk (the loop ends without triggering the flush above)
    if current_sentences:
        chunk_content = " ".join(current_sentences)
        chunks.append(
            Chunk(
                content=chunk_content,
                chunk_index=chunk_index,
                token_count=current_token_count,
                char_start=char_cursor,
                char_end=char_cursor + len(chunk_content),
            )
        )

    return chunks
