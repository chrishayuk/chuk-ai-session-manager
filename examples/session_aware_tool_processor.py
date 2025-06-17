#!/usr/bin/env python3
# examples/session_aware_tool_processor.py
"""
Session-aware tool processor (async, chuk-tool-processor 0.1.x).

See doc-string at top of file for full details.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List

from chuk_tool_processor.core.processor import ToolProcessor
from chuk_tool_processor.models.tool_call import ToolCall
from chuk_tool_processor.models.tool_result import ToolResult

from chuk_ai_session_manager.storage.providers.memory import InMemorySessionStore
from chuk_ai_session_manager.storage import SessionStoreProvider
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.session_prompt_builder import build_prompt_from_session

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


# ─────────────────────────── core processor ──────────────────────────
class SessionAwareToolProcessor:
    """Run tool-calls, add retry/caching, and log them into a session."""

    def __init__(
        self,
        session_id: str,
        *,
        enable_caching: bool = True,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.session_id = session_id
        self.enable_caching = enable_caching
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._cache: Dict[str, Any] = {}

        self._tp = ToolProcessor()
        if not hasattr(self._tp, "executor"):
            raise AttributeError("chuk_tool_processor is too old - no .executor")

    # ── factory ───────────────────────────────────────────────────────
    @classmethod
    async def create(cls, session_id: str, **kw):
        store = SessionStoreProvider.get_store()
        if not await store.get(session_id):
            raise ValueError(f"Session {session_id!r} not found")
        return cls(session_id=session_id, **kw)

    # ── helpers ───────────────────────────────────────────────────────
    async def _await(self, v: Any) -> Any:
        return await v if asyncio.iscoroutine(v) else v

    def _cache_key(self, tool: str, args: Dict[str, Any]) -> str:
        blob = f"{tool}:{json.dumps(args, sort_keys=True)}"
        return hashlib.md5(blob.encode()).hexdigest()

    async def _run_calls(self, raw_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        tcalls: List[ToolCall] = []
        for c in raw_calls:
            fn = c["function"]
            try:
                args = json.loads(fn["arguments"])
            except json.JSONDecodeError:
                args = {"raw": fn["arguments"]}
            tcalls.append(ToolCall(tool=fn["name"], arguments=args))

        results = await self._tp.executor.execute(tcalls)
        for r in results:
            r.result = await self._await(r.result)
        return results

    async def _log_event(
        self,
        session: Session,
        parent: SessionEvent,
        res: ToolResult,
        attempt: int,
        *,
        cached: bool,
        failed: bool = False,
    ):
        # JSON-serialisable copies
        safe_args = getattr(res, "arguments", None)
        safe_res: Any = res.result
        if hasattr(safe_res, "model_dump"):
            safe_res = safe_res.model_dump()

        try:
            completion = json.dumps(safe_res, ensure_ascii=False)
        except TypeError:
            completion = json.dumps(str(safe_res), ensure_ascii=False)

        ev = await SessionEvent.create_with_tokens(
            message={
                "tool": res.tool,
                "arguments": safe_args,
                "result": safe_res,
                "error":  res.error,
                "cached": cached,
            },
            prompt=f"{res.tool}({json.dumps(safe_args, default=str)})",
            completion=completion if safe_res is not None else "",
            model="tool-execution",
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL,
        )
        await ev.update_metadata("parent_event_id", parent.id)
        await ev.update_metadata("attempt", attempt)
        if failed:
            await ev.update_metadata("failed", True)
        await session.add_event_and_save(ev)

    # ── public entry ────────────────────────────────────────────────
    async def process_llm_message(self, llm_msg: Dict[str, Any]) -> List[ToolResult]:
        store = SessionStoreProvider.get_store()
        session = await store.get(self.session_id)
        if not session:
            raise ValueError(f"Session {self.session_id!r} not found")

        parent = await SessionEvent.create_with_tokens(
            message=llm_msg,
            prompt="",
            completion=json.dumps(llm_msg),
            model="gpt-4o-mini",
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
        await session.add_event_and_save(parent)

        calls = llm_msg.get("tool_calls", [])
        if not calls:
            return []

        out: List[ToolResult] = []
        for raw_call in calls:
            fn = raw_call["function"]
            name = fn["name"]
            try:
                args = json.loads(fn["arguments"])
            except json.JSONDecodeError:
                args = {"raw": fn["arguments"]}

            ck = self._cache_key(name, args) if self.enable_caching else None
            if ck and ck in self._cache:
                res: ToolResult = self._cache[ck]
                await self._log_event(session, parent, res, 1, cached=True)
                out.append(res)
                continue

            last_err: str | None = None
            for attempt in range(1, self.max_retries + 2):
                try:
                    res = (await self._run_calls([raw_call]))[0]
                    if ck:
                        self._cache[ck] = res
                    await self._log_event(session, parent, res, attempt, cached=False)
                    out.append(res)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = str(exc)
                    if attempt <= self.max_retries:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    err_res = ToolResult(tool=name, result=None, error=last_err)
                    await self._log_event(
                        session, parent, err_res, attempt, cached=False, failed=True
                    )
                    out.append(err_res)
        return out


# ─────────────────────────── demo harness ───────────────────────────
async def _demo() -> None:
    """Minimal self-test when the file is executed directly."""
    # 1) In-memory store & session
    store = InMemorySessionStore()
    SessionStoreProvider.set_store(store)
    session = await Session.create()

    # 2) sample weather tool (auto-registers on import)
    from chuk_ai_session_manager.sample_tools import WeatherTool  # noqa: F401

    proc = await SessionAwareToolProcessor.create(session.id)

    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": json.dumps({"location": "London"}),
                },
            }
        ],
    }

    results = await proc.process_llm_message(assistant_msg)
    print("\nTool execution results:")
    for r in results:
        val = r.result
        if hasattr(val, "model_dump"):
            val = val.model_dump()
        try:
            print(json.dumps(val, indent=2, ensure_ascii=False))
        except TypeError:
            print(str(val))

    print("\nHierarchical Session Events:")

    async def _tree(evt: SessionEvent, depth=0):
        pad = "  " * depth
        print(f"{pad}• {evt.type.value:9} id={evt.id}")
        for ch in [e for e in session.events
                   if await e.get_metadata("parent_event_id") == evt.id]:
            await _tree(ch, depth + 1)

    roots = [e for e in session.events
             if not await e.get_metadata("parent_event_id")]
    for root in roots:
        await _tree(root)

    nxt = await build_prompt_from_session(session)
    print("\nNext-turn prompt that would be sent to the LLM:")
    print(json.dumps(nxt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(_demo())
