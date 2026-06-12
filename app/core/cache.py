"""
app/core/cache.py
-----------------
Generic typed cache client over Redis.

CacheClient[T] owns four things:
  - namespace  : string prefix scoping all keys for a domain
  - version    : embedded in the stored envelope, invalidates stale data on read
  - default_ttl: expiry in seconds, overridable per write
  - T          : the value type; JSON-serialised on write, deserialised on read

Envelope pattern
----------------
Values are stored as:
    { "__v": "<version>", "d": <value> }

On read, a version mismatch returns None (treat as cache miss).
The caller always receives exactly what they stored — no metadata leaks
into the domain dict and no stripping needed on the read side.

Redis as a parameter
--------------------
The client holds configuration only. Every method accepts redis as a
parameter so the client can be a module-level constant (created before
Redis is ready) while working naturally with FastAPI dependency injection.

Usage
-----
    from app.core.cache import CacheClient

    my_cache: CacheClient[MyData] = CacheClient(
        namespace="mymodule:things",
        version="v1",
        default_ttl=3_600,
    )

    # Redis arrives via FastAPI Depends — client doesn't care when it was created
    data: MyData | None = await my_cache.get(redis, some_key)
    await my_cache.set(redis, some_key, data)
    await my_cache.delete(redis, some_key)
"""

from __future__ import annotations

import json
import logging
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_ENVELOPE_VERSION_KEY = "__v"
_ENVELOPE_DATA_KEY    = "d"


class CacheClient(Generic[T]):
    """
    Typed, namespaced, versioned Redis cache client.

    Stateless — holds configuration only. Redis is injected at call time.
    """

    def __init__(self, namespace: str, version: str, default_ttl: int) -> None:
        """
        Args:
            namespace:   Key prefix for all entries in this store, e.g.
                         "routing:route_layer" or "agents:config".
                         Dots and colons are both fine separators.
            version:     Arbitrary string embedded in every stored envelope.
                         Change it to invalidate all existing entries for this
                         store without touching Redis directly. Convention:
                         "v1:model-name" ties it to a dependency version.
            default_ttl: Default expiry in seconds. Individual .set() calls
                         can override with the ttl parameter.
        """
        self.namespace   = namespace
        self.version     = version
        self.default_ttl = default_ttl

    # ── Key construction ──────────────────────────────────────────────────────

    def make_key(self, key: str) -> str:
        """
        Full Redis key: '{namespace}:{key}'.
        Useful for logging and debugging. Internal callers use this implicitly.
        """
        return f"{self.namespace}:{key}"

    # ── Core operations ───────────────────────────────────────────────────────

    async def get(self, redis, key: str) -> T | None:
        """
        Fetch and deserialise a cached value.

        Returns None when:
            - The key doesn't exist or has expired
            - The stored version doesn't match self.version (stale data)
            - The payload can't be deserialised (corrupted entry)

        All miss cases are silent — callers treat None as "not in cache"
        regardless of the underlying reason.
        """
        raw = await redis.get(self.make_key(key))
        if not raw:
            return None

        try:
            envelope = json.loads(raw)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                "Cache deserialisation error for key '%s': %s",
                self.make_key(key), exc,
            )
            return None

        if envelope.get(_ENVELOPE_VERSION_KEY) != self.version:
            logger.debug(
                "Cache version mismatch for '%s' — got '%s', want '%s'. Treating as miss.",
                self.make_key(key),
                envelope.get(_ENVELOPE_VERSION_KEY),
                self.version,
            )
            return None

        return envelope.get(_ENVELOPE_DATA_KEY)

    async def set(
        self,
        redis,
        key: str,
        value: T,
        ttl: int | None = None,
    ) -> None:
        """
        Serialise and store value in the versioned envelope.

        Args:
            redis: Redis connection.
            key:   Domain key (namespace is prepended automatically).
            value: Any JSON-serialisable value.
            ttl:   Expiry override in seconds. Falls back to self.default_ttl.
        """
        envelope = {
            _ENVELOPE_VERSION_KEY: self.version,
            _ENVELOPE_DATA_KEY:    value,
        }

        try:
            serialised = json.dumps(envelope)
        except (TypeError, Exception) as exc:
            logger.error(
                "Cache serialisation error for key '%s': %s",
                self.make_key(key), exc,
            )
            return

        await redis.setex(
            self.make_key(key),
            ttl if ttl is not None else self.default_ttl,
            serialised,
        )

    async def delete(self, redis, key: str) -> None:
        """Delete a cached entry. No-op if the key doesn't exist."""
        await redis.delete(self.make_key(key))

    async def exists(self, redis, key: str) -> bool:
        """
        Return True if the key is present in Redis.
        Does not deserialise or version-check — purely an existence probe.
        Use get() when you need the value; use exists() only when you
        need a fast boolean without the payload.
        """
        return bool(await redis.exists(self.make_key(key)))