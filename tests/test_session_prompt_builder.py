# tests/test_session_prompt_builder.py
"""
Minimal async tests for session_prompt_builder.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.session_prompt_builder import (
    build_prompt_from_session,
    PromptStrategy,
    truncate_prompt_to_token_limit,
)
from chuk_ai_session_manager.models.token_usage import TokenUsage


@pytest.mark.asyncio
async def test_empty_session():
    assert await build_prompt_from_session(Session()) == []


@pytest.mark.asyncio
async def test_minimal_basic():
    s = Session()
    await s.add_event(SessionEvent(message="q", source=EventSource.USER, type=EventType.MESSAGE))
    await s.add_event(SessionEvent(message="a", source=EventSource.LLM, type=EventType.MESSAGE))
    p = await build_prompt_from_session(s, PromptStrategy.MINIMAL)
    assert p[0]["role"] == "user" and p[1]["content"] is None


@pytest.mark.asyncio
async def test_tool_and_retry_and_strategies():
    s = Session()
    u = SessionEvent(message="q", source=EventSource.USER, type=EventType.MESSAGE)
    a = SessionEvent(message="a", source=EventSource.LLM, type=EventType.MESSAGE)
    await s.add_event(u); await s.add_event(a)
    t = SessionEvent(
        message={"tool": "t", "result": {"x": 1}},
        source=EventSource.SYSTEM,
        type=EventType.TOOL_CALL,
        metadata={"parent_event_id": a.id},
    )
    await s.add_event(t)

    p = await build_prompt_from_session(s, PromptStrategy.MINIMAL)
    assert any(m["role"] == "tool" for m in p)

    long_prompt = [{"role": "user", "content": "u"}] * 10
    # patch the *synchronous* count_tokens so it returns an int
    with patch.object(TokenUsage, "count_tokens", return_value=1000):
        out = await truncate_prompt_to_token_limit(long_prompt, max_tokens=1)
    assert len(out) < len(long_prompt)
