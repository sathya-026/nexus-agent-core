"""
app/routing/types.py
--------------------
Public data types for the routing package.

Kept in their own module so any internal file can import them
without triggering circular imports — types have no logic and
no dependencies within this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Route(str, Enum):
    RAG  = "rag"   # retrieve from knowledge base only
    TOOL = "tool"  # call tools only
    BOTH = "both"  # retrieve + call tools
    META = "meta"  # capabilities / "what can you do?" query
    NONE = "none"  # skip retrieval, go straight to LLM


@dataclass
class RouteResult:
    route        : Route
    matched_tools: list[str]   = field(default_factory=list)
    query_text   : str | None  = None
    # query_text: enriched routing query (includes conversation context when
    # the message is a contextual follow-up). Passed to the retriever.

    meta_context : dict | None = None
    # meta_context is set only when route == Route.META:
    # {
    #   "tool_names"        : list[str],
    #   "tool_descriptions" : list[str],
    #   "doc_names"         : list[str],
    #   "doc_descriptions"  : list[str],
    # }
    # The planner uses this to describe available capabilities to the user.
    
    conversation_intent_embedding: list[float] | None = None
    # conversation_intent_embedding: rolling weighted embedding of recent turns.
    # Set when the router has blended current query with conversation history.
    # Used downstream to update conversation intent after planner completes.