# tests/storage/providers/test_redis.py
"""Redis-backed session store.

Compatible with **both** synchronous ``redis.Redis`` and asynchronous
``redis.asyncio.Redis`` clients.  We inspect the provided client once and then
route every Redis call through :py:meth:`_call`, which either *await*s the
coroutine or pushes the blocking call into an executor so the event loop stays
free.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from typing import Dict, List, Optional, Type, TypeVar

from chuk_ai_session_manager.models.session import Session

T = TypeVar("T", bound=Session)


class RedisSessionStore:  # not tying to a specific interface to stay decoupled
    """Store sessions in Redis with transparent sync/async support."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(
        self,
        redis_client,
        *,
        key_prefix: str = "session:",
        expiration_seconds: Optional[int] = None,
        auto_save: bool = True,
        session_class: Type[T] = Session,
    ) -> None:
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.expiration_seconds = expiration_seconds
        self.auto_save = auto_save
        self.session_class = session_class
        self._cache: Dict[str, T] = {}

        # Detect if the supplied client is coroutine-based.
        self._async = inspect.iscoroutinefunction(getattr(redis_client, "set", None))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_key(self, session_id: str) -> str:
        return f"{self.key_prefix}{session_id}"

    async def _call(self, method: str, *args):
        """Invoke *method* on the underlying client correctly."""
        fn = getattr(self.redis, method)
        if self._async:
            return await fn(*args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save(self, session: T) -> None:
        """Cache and optionally persist *session* immediately."""
        self._cache[session.id] = session
        if not self.auto_save:
            return
        await self._persist(session)

    async def _persist(self, session: T) -> None:
        payload = session.json() if hasattr(session, "json") else json.dumps(session.to_dict())
        key = self._make_key(session.id)
        if self.expiration_seconds is not None:
            await self._call("setex", key, self.expiration_seconds, payload)
        else:
            await self._call("set", key, payload)

    # ------------------------------------------------------------------
    async def get(self, session_id: str) -> Optional[T]:
        if session_id in self._cache:
            return self._cache[session_id]

        raw = await self._call("get", self._make_key(session_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()

        try:
            session = self.session_class.parse_raw(raw)  # type: ignore[attr-defined]
        except AttributeError:
            # Fallback if .parse_raw not available (pydantic <2)
            session = self.session_class.model_validate(json.loads(raw))  # type: ignore[arg-type]

        self._cache[session_id] = session
        return session

    # ------------------------------------------------------------------
    async def delete(self, session_id: str) -> None:
        self._cache.pop(session_id, None)
        await self._call("delete", self._make_key(session_id))

    # ------------------------------------------------------------------
    async def list_sessions(self, prefix: str = "") -> List[str]:
        pattern = f"{self.key_prefix}{prefix}*"
        keys = await self._call("keys", pattern)
        out: List[str] = []
        for k in keys:
            if isinstance(k, bytes):
                k = k.decode()
            out.append(k.replace(self.key_prefix, "", 1))
        return out

    # ------------------------------------------------------------------
    async def set_expiration(self, session_id: str, ttl: int) -> None:
        await self._call("expire", self._make_key(session_id), ttl)

    # ------------------------------------------------------------------
    async def flush(self) -> None:
        if not self._cache:
            return
        await asyncio.gather(*[self._persist(s) for s in self._cache.values()])

    # ------------------------------------------------------------------
    async def clear_cache(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Factory helper - imports *redis.asyncio* lazily (helps unit-testing)
# ---------------------------------------------------------------------------

async def create_redis_session_store(
    *,
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    key_prefix: str = "session:",
    expiration_seconds: Optional[int] = None,
    auto_save: bool = True,
    session_class: Type[T] = Session,
    **redis_kwargs,
) -> RedisSessionStore:
    redis_asyncio = importlib.import_module("redis.asyncio")
    client = redis_asyncio.Redis(host=host, port=port, db=db, password=password, **redis_kwargs)
    return RedisSessionStore(
        redis_client=client,
        key_prefix=key_prefix,
        expiration_seconds=expiration_seconds,
        auto_save=auto_save,
        session_class=session_class,
    )
