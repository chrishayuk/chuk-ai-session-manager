# tests/test_session_aware_tool_processor.py
"""
Async tests for chuk-style SessionAwareToolProcessor.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.storage import (
    InMemorySessionStore,
    SessionStoreProvider,
)
from chuk_ai_session_manager.session_aware_tool_processor import (
    SessionAwareToolProcessor,
    ToolResult,
)

# ───────────────────────── fixtures ──────────────────────────
@pytest_asyncio.fixture
async def sid():
    store = InMemorySessionStore()
    SessionStoreProvider.set_store(store)
    sess = Session()
    await store.save(sess)
    return sess.id


# ───────────────────────── helpers ───────────────────────────
def _dummy_msg():
    return {
        "tool_calls": [
            {
                "id": "cid",
                "type": "function",
                "function": {"name": "t", "arguments": "{}"},
            }
        ]
    }


async def _noop_llm(_: str) -> dict:  # placeholder callback
    return {}


# ───────────────────────── tests ─────────────────────────────
@pytest.mark.asyncio
async def test_process_tool_calls(sid):
    proc = await SessionAwareToolProcessor.create(session_id=sid)
    with patch.object(
        proc,
        "_exec_calls",
        AsyncMock(return_value=[ToolResult(tool="t", result={"ok": True})]),
    ):
        res = await proc.process_llm_message(_dummy_msg(), _noop_llm)
        assert res[0].result == {"ok": True}


@pytest.mark.asyncio
async def test_cache_behavior(sid):
    proc = await SessionAwareToolProcessor.create(session_id=sid, enable_caching=True)

    with patch.object(
        proc,
        "_exec_calls",
        AsyncMock(return_value=[ToolResult(tool="t", result={"v": 1})]),
    ) as first_call:
        await proc.process_llm_message(_dummy_msg(), _noop_llm)
        first_call.assert_awaited()

    with patch.object(proc, "_exec_calls", AsyncMock()) as second_call:
        out = await proc.process_llm_message(_dummy_msg(), _noop_llm)
        second_call.assert_not_called()
        assert out[0].result == {"v": 1}


@pytest.mark.asyncio
async def test_retry_behavior(sid):
    proc = await SessionAwareToolProcessor.create(
        session_id=sid, max_retries=2, retry_delay=0.001
    )

    with patch.object(
        proc,
        "_exec_calls",
        AsyncMock(side_effect=[Exception("fail"), [ToolResult(tool="t", result={"v": 1})]]),
    ):
        out = await proc.process_llm_message(_dummy_msg(), _noop_llm)
        assert out[0].result == {"v": 1}

    sess = await SessionStoreProvider.get_store().get(sid)
    # successful TOOL_CALL should have been logged on attempt 2
    tc_events = [e for e in sess.events if e.type == EventType.TOOL_CALL]
    assert len(tc_events) == 1
    assert tc_events[0].metadata["attempt"] == 2


@pytest.mark.asyncio
async def test_max_retries_exceeded(sid):
    proc = await SessionAwareToolProcessor.create(
        session_id=sid, max_retries=1, retry_delay=0.001
    )

    with patch.object(proc, "_exec_calls", AsyncMock(side_effect=Exception("boom"))):
        out = await proc.process_llm_message(_dummy_msg(), _noop_llm)
        assert out[0].error and "boom" in out[0].error
