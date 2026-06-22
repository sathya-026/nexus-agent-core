"""
app/routing
-----------
Public API for the semantic routing package.

Import from here — never from internal sub-modules directly.
Internal layout is free to change; this surface stays stable.

    from app.routing import route, RouteResult, Route
    from app.routing import invalidate_route_cache
    from app.routing import warmup_model
"""

from app.agent.query_router.types import Route, RouteResult
from app.agent.query_router.router import route
from app.agent.query_router.cache import invalidate_route_cache, build_route_cache

__all__ = [
    # Types
    "Route",
    "RouteResult",
    # Core function
    "route",
    # Cache management (called by NestJS-facing indexing router)
    "invalidate_route_cache",
    "build_route_cache",
    # Startup
    "warmup_model",
]