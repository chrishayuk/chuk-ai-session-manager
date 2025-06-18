# chuk_ai_session_manager/session_aware_tool_processor.py
#!/usr/bin/env python3
"""Session-aware Tool-processor for chuk_tool_processor 0.1.x.

* Converts OpenAI `tool_calls` → `ToolCall` objects.
* Executes them with **ToolProcessor().executor.execute**.
* Adds caching / retry.
* Logs every call into the session tree, storing the **string-repr**
  of the result (this is what the prompt-builder currently expects)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List

from chuk_tool_processor.core.processor import ToolProcessor
from chuk_tool_processor.models.tool_call import ToolCall
from chuk_tool_processor.models.tool_result import ToolResult

from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore

logger = logging.getLogger(__name__)


class SessionAwareToolProcessor:
    """Run tool-calls, add retry/caching, and log into a session."""

    # ─────────────────────────── construction ──────────────────────────
    def __init__(
        self,
        session_id: str,
        *,
        enable_caching: bool = True,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.session_id     = session_id
        self.enable_caching = enable_caching
        self.max_retries    = max_retries
        self.retry_delay    = retry_delay
        self.cache: Dict[str, ToolResult] = {}

        self._tp = ToolProcessor()
        if not hasattr(self._tp, "executor"):
            raise AttributeError("Installed chuk_tool_processor is too old - missing `.executor`")

    @classmethod
    async def create(cls, session_id: str, **kwargs):
        backend = get_backend()
        store = ChukSessionsStore(backend)
        if not await store.get(session_id):
            raise ValueError(f"Session {session_id} not found")
        return cls(session_id=session_id, **kwargs)

    # ─────────────────────────── internals ─────────────────────────────
    async def _maybe_await(self, val: Any) -> Any:
        return await val if asyncio.iscoroutine(val) else val

    async def _exec_calls(self, calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Convert dicts → ToolCall and drive the executor."""
        tool_calls: list[ToolCall] = []
        for c in calls:
            fn   = c.get("function", {})
            name = fn.get("name", "tool")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {"raw": fn.get("arguments")}
            tool_calls.append(ToolCall(tool=name, arguments=args))

        results = await self._tp.executor.execute(tool_calls)
        for r in results:
            r.result = await self._maybe_await(r.result)
        return results

    async def _log_event(
        self,
        session,
        parent_id: str,
        res: ToolResult,
        attempt: int,
        *,
        cached: bool,
        failed: bool = False,
    ) -> None:
        """Persist TOOL_CALL with *string* result (prompt-friendly)."""
        result_str = str(res.result) if res.result is not None else "null"

        ev = await SessionEvent.create_with_tokens(
            message={
                "tool":      res.tool,
                "arguments": getattr(res, "arguments", None),
                "result":    result_str,
                "error":     res.error,
                "cached":    cached,
            },
            prompt=f"{res.tool}({json.dumps(getattr(res, 'arguments', None), default=str)})",
            completion=result_str,
            model="tool-execution",
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL,
        )
        await ev.update_metadata("parent_event_id", parent_id)
        await ev.update_metadata("call_id", getattr(res, "id", "cid"))
        await ev.update_metadata("attempt", attempt)
        if failed:
            await ev.update_metadata("failed", True)
        await session.add_event_and_save(ev)

    # ─────────────────────────── public API ────────────────────────────
    async def process_llm_message(self, llm_msg: Dict[str, Any], _) -> List[ToolResult]:
        backend = get_backend()
        store = ChukSessionsStore(backend)
        session = await store.get(self.session_id)
        if not session:
            raise ValueError(f"Session {self.session_id} not found")

        parent_evt = await SessionEvent.create_with_tokens(
            message=llm_msg,
            prompt="",
            completion=json.dumps(llm_msg, ensure_ascii=False),
            model="gpt-4o-mini",
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
        await session.add_event_and_save(parent_evt)

        calls = llm_msg.get("tool_calls", [])
        if not calls:
            return []

        out: list[ToolResult] = []
        for call in calls:
            fn   = call.get("function", {})
            name = fn.get("name", "tool")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {"raw": fn.get("arguments")}

            cache_key = (
                hashlib.md5(f"{name}:{json.dumps(args, sort_keys=True)}".encode()).hexdigest()
                if self.enable_caching else None
            )

            # 1) cache hit --------------------------------------------------
            if cache_key and (cached := self.cache.get(cache_key)):
                await self._log_event(session, parent_evt.id, cached, 1, cached=True)
                out.append(cached)
                continue

            # 2) execute with retry ----------------------------------------
            last_err: str | None = None
            for attempt in range(1, self.max_retries + 2):
                try:
                    res = (await self._exec_calls([call]))[0]
                    if cache_key:
                        self.cache[cache_key] = res
                    await self._log_event(session, parent_evt.id, res, attempt, cached=False)
                    out.append(res)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = str(exc)
                    if attempt <= self.max_retries:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    err_res = ToolResult(tool=name, result=None, error=last_err)  # type: ignore[arg-type]
                    await self._log_event(
                        session, parent_evt.id, err_res, attempt,
                        cached=False, failed=True
                    )
                    out.append(err_res)

        return out