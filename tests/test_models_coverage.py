# tests/test_models_coverage.py
"""
Comprehensive test suite targeting >90% coverage for ALL model files:
  - models/session.py
  - models/session_event.py
  - models/token_usage.py
  - models/session_metadata.py
  - models/session_run.py
  - models/event_source.py
  - models/event_type.py
  - models/__init__.py
"""

import importlib
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.session_metadata import SessionMetadata
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.token_usage import TokenUsage, TokenSummary
from chuk_ai_session_manager.models.session_run import SessionRun, RunStatus


# ===========================================================================
# EventSource tests
# ===========================================================================


class TestEventSource:
    """Tests for EventSource enum covering all members and string behaviour."""

    def test_user_value(self):
        assert EventSource.USER == "user"
        assert EventSource.USER.value == "user"

    def test_llm_value(self):
        assert EventSource.LLM == "llm"
        assert EventSource.LLM.value == "llm"

    def test_system_value(self):
        assert EventSource.SYSTEM == "system"
        assert EventSource.SYSTEM.value == "system"

    def test_is_str_subclass(self):
        assert isinstance(EventSource.USER, str)

    def test_iteration(self):
        members = list(EventSource)
        assert len(members) == 3


# ===========================================================================
# EventType tests
# ===========================================================================


class TestEventType:
    """Tests for EventType enum covering all members."""

    def test_message(self):
        assert EventType.MESSAGE.value == "message"

    def test_summary(self):
        assert EventType.SUMMARY.value == "summary"

    def test_tool_call(self):
        assert EventType.TOOL_CALL.value == "tool_call"

    def test_reference(self):
        assert EventType.REFERENCE.value == "reference"

    def test_context_bridge(self):
        assert EventType.CONTEXT_BRIDGE.value == "context_bridge"

    def test_tool_trace(self):
        assert EventType.TOOL_TRACE.value == "tool_trace"

    def test_tool_pattern(self):
        assert EventType.TOOL_PATTERN.value == "tool_pattern"

    def test_tool_fix(self):
        assert EventType.TOOL_FIX.value == "tool_fix"

    def test_is_str_subclass(self):
        assert isinstance(EventType.MESSAGE, str)

    def test_iteration(self):
        members = list(EventType)
        assert len(members) == 8


# ===========================================================================
# TokenUsage tests
# ===========================================================================


class TestTokenUsageCoverage:
    """Thorough tests for TokenUsage model."""

    # --- Initialisation / auto-calculation ---

    def test_defaults(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.model == ""
        assert usage.estimated_cost_usd is None  # no model -> no cost calc

    def test_auto_total_tokens(self):
        usage = TokenUsage(prompt_tokens=60, completion_tokens=40)
        assert usage.total_tokens == 100

    def test_explicit_total_tokens_zero_override(self):
        """When prompt+completion > 0 but total_tokens left at default 0, it auto-calculates."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        assert usage.total_tokens == 15

    def test_auto_cost_when_model_present(self):
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, model="gpt-4")
        assert usage.estimated_cost_usd is not None
        assert usage.estimated_cost_usd > 0

    def test_no_cost_when_no_model(self):
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
        assert usage.estimated_cost_usd is None

    # --- Cost calculation for every known model ---

    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
        ],
    )
    def test_cost_known_models(self, model_name):
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, model=model_name)
        assert usage.estimated_cost_usd is not None
        assert usage.estimated_cost_usd > 0

    def test_cost_unknown_model_uses_default(self):
        usage = TokenUsage(
            prompt_tokens=1000, completion_tokens=500, model="mystery-model-9000"
        )
        assert usage.estimated_cost_usd is not None
        assert usage.estimated_cost_usd > 0

    def test_cost_zero_tokens(self):
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, model="gpt-4")
        assert usage.estimated_cost_usd == 0.0

    def test_gpt4_more_expensive_than_gpt35(self):
        gpt4 = TokenUsage(prompt_tokens=1000, completion_tokens=500, model="gpt-4")
        gpt35 = TokenUsage(
            prompt_tokens=1000, completion_tokens=500, model="gpt-3.5-turbo"
        )
        assert gpt4.estimated_cost_usd > gpt35.estimated_cost_usd

    # --- _calculate_cost_sync directly ---

    def test_calculate_cost_sync_directly(self):
        usage = TokenUsage(prompt_tokens=500, completion_tokens=250, model="gpt-4")
        cost = usage._calculate_cost_sync()
        assert isinstance(cost, float)
        assert cost > 0

    # --- async calculate_cost ---

    async def test_calculate_cost_async(self):
        usage = TokenUsage(prompt_tokens=500, completion_tokens=250, model="gpt-4")
        cost = await usage.calculate_cost()
        assert isinstance(cost, float)
        assert cost > 0

    # --- _update_sync ---

    def test_update_sync(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        original_cost = usage.estimated_cost_usd
        usage._update_sync(prompt_tokens=100, completion_tokens=50)
        assert usage.prompt_tokens == 200
        assert usage.completion_tokens == 100
        assert usage.total_tokens == 300
        assert usage.estimated_cost_usd > original_cost

    def test_update_sync_no_model(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        usage._update_sync(prompt_tokens=5, completion_tokens=5)
        assert usage.prompt_tokens == 15
        assert usage.completion_tokens == 10
        assert usage.total_tokens == 25
        # No model, so estimated_cost_usd stays None
        assert usage.estimated_cost_usd is None

    # --- async update ---

    async def test_update_async(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        await usage.update(prompt_tokens=100, completion_tokens=50)
        assert usage.prompt_tokens == 200
        assert usage.completion_tokens == 100
        assert usage.total_tokens == 300
        assert usage.estimated_cost_usd is not None

    async def test_update_async_no_model(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        await usage.update(prompt_tokens=5)
        assert usage.prompt_tokens == 15
        assert usage.total_tokens == 20

    # --- _count_tokens_sync ---

    def test_count_tokens_sync_none(self):
        assert TokenUsage._count_tokens_sync(None) == 0

    def test_count_tokens_sync_empty_string(self):
        assert TokenUsage._count_tokens_sync("") == 0

    def test_count_tokens_sync_regular_text(self):
        tokens = TokenUsage._count_tokens_sync("Hello world, this is a test")
        assert tokens > 0

    def test_count_tokens_sync_non_string_object(self):
        """Non-string objects get str() called on them."""
        tokens = TokenUsage._count_tokens_sync(12345)
        assert tokens > 0

    def test_count_tokens_sync_with_known_model(self):
        tokens = TokenUsage._count_tokens_sync("Hello world", "gpt-4")
        assert tokens > 0

    def test_count_tokens_sync_with_unknown_model(self):
        """Unknown model falls back to cl100k_base or approximation."""
        tokens = TokenUsage._count_tokens_sync("Hello world", "unknown-model-xyz")
        assert tokens > 0

    # --- async count_tokens ---

    async def test_count_tokens_async(self):
        tokens = await TokenUsage.count_tokens("Hello world test", "gpt-3.5-turbo")
        assert isinstance(tokens, int)
        assert tokens > 0

    async def test_count_tokens_async_none(self):
        tokens = await TokenUsage.count_tokens(None)
        assert tokens == 0

    async def test_count_tokens_async_empty(self):
        tokens = await TokenUsage.count_tokens("")
        assert tokens == 0

    # --- _from_text_sync ---

    def test_from_text_sync_prompt_only(self):
        usage = TokenUsage._from_text_sync("Hello world")
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens == 0
        assert usage.model == "gpt-3.5-turbo"

    def test_from_text_sync_prompt_and_completion(self):
        usage = TokenUsage._from_text_sync("Hello", "World", "gpt-4")
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.model == "gpt-4"

    # --- async from_text ---

    async def test_from_text_async(self):
        usage = await TokenUsage.from_text("Hello world", "Goodbye", "gpt-4")
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.model == "gpt-4"
        assert usage.estimated_cost_usd is not None

    async def test_from_text_async_no_completion(self):
        usage = await TokenUsage.from_text("Hello world", model="gpt-3.5-turbo")
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens == 0

    # --- __add__ ---

    def test_add_two_usages(self):
        a = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        b = TokenUsage(prompt_tokens=200, completion_tokens=75, model="gpt-4")
        c = a + b
        assert c.prompt_tokens == 300
        assert c.completion_tokens == 125
        assert c.total_tokens == 425
        assert c.model == "gpt-4"

    def test_add_first_has_no_model(self):
        a = TokenUsage(prompt_tokens=10, completion_tokens=5)
        b = TokenUsage(prompt_tokens=20, completion_tokens=10, model="gpt-4")
        c = a + b
        assert c.model == "gpt-4"

    def test_add_second_has_no_model(self):
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, model="gpt-4")
        b = TokenUsage(prompt_tokens=20, completion_tokens=10)
        c = a + b
        assert c.model == "gpt-4"

    def test_add_neither_has_model(self):
        a = TokenUsage(prompt_tokens=10, completion_tokens=5)
        b = TokenUsage(prompt_tokens=20, completion_tokens=10)
        c = a + b
        assert c.model == ""

    # --- tiktoken unavailable fallback ---

    def test_count_tokens_without_tiktoken(self):
        """When tiktoken is not available, falls back to len(text)/4 approximation."""
        with patch(
            "chuk_ai_session_manager.models.token_usage.TIKTOKEN_AVAILABLE", False
        ):
            tokens = TokenUsage._count_tokens_sync("Hello world this is a test string")
            expected = len("Hello world this is a test string") // 4
            assert tokens == expected


# ===========================================================================
# TokenSummary tests
# ===========================================================================


class TestTokenSummaryCoverage:
    """Thorough tests for TokenSummary model."""

    def test_defaults(self):
        summary = TokenSummary()
        assert summary.total_prompt_tokens == 0
        assert summary.total_completion_tokens == 0
        assert summary.total_tokens == 0
        assert summary.total_estimated_cost_usd == 0.0
        assert summary.usage_by_model == {}

    # --- _add_usage_sync ---

    def test_add_usage_sync_single(self):
        summary = TokenSummary()
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        summary._add_usage_sync(usage)
        assert summary.total_prompt_tokens == 100
        assert summary.total_completion_tokens == 50
        assert summary.total_tokens == 150
        assert summary.total_estimated_cost_usd > 0
        assert "gpt-4" in summary.usage_by_model

    def test_add_usage_sync_multiple_same_model(self):
        summary = TokenSummary()
        u1 = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        u2 = TokenUsage(prompt_tokens=200, completion_tokens=75, model="gpt-4")
        summary._add_usage_sync(u1)
        summary._add_usage_sync(u2)
        assert summary.total_prompt_tokens == 300
        assert summary.total_completion_tokens == 125
        assert summary.total_tokens == 425
        assert len(summary.usage_by_model) == 1
        # model entry should be aggregated
        assert summary.usage_by_model["gpt-4"].prompt_tokens == 300

    def test_add_usage_sync_different_models(self):
        summary = TokenSummary()
        u1 = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        u2 = TokenUsage(prompt_tokens=200, completion_tokens=75, model="gpt-3.5-turbo")
        summary._add_usage_sync(u1)
        summary._add_usage_sync(u2)
        assert len(summary.usage_by_model) == 2

    def test_add_usage_sync_no_model(self):
        summary = TokenSummary()
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        summary._add_usage_sync(usage)
        assert summary.total_prompt_tokens == 100
        # Empty model string means no entry in usage_by_model
        assert len(summary.usage_by_model) == 0

    def test_add_usage_sync_no_cost(self):
        summary = TokenSummary()
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        # estimated_cost_usd is None since no model
        summary._add_usage_sync(usage)
        assert summary.total_estimated_cost_usd == 0.0

    # --- async add_usage ---

    async def test_add_usage_async_single(self):
        summary = TokenSummary()
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        await summary.add_usage(usage)
        assert summary.total_prompt_tokens == 100
        assert summary.total_completion_tokens == 50
        assert summary.total_tokens == 150

    async def test_add_usage_async_multiple_same_model(self):
        summary = TokenSummary()
        u1 = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        u2 = TokenUsage(prompt_tokens=200, completion_tokens=75, model="gpt-4")
        await summary.add_usage(u1)
        await summary.add_usage(u2)
        assert summary.total_prompt_tokens == 300
        assert summary.total_completion_tokens == 125
        assert len(summary.usage_by_model) == 1
        # aggregated via async update
        assert summary.usage_by_model["gpt-4"].prompt_tokens == 300

    async def test_add_usage_async_different_models(self):
        summary = TokenSummary()
        u1 = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        u2 = TokenUsage(
            prompt_tokens=200, completion_tokens=75, model="claude-3-sonnet"
        )
        await summary.add_usage(u1)
        await summary.add_usage(u2)
        assert len(summary.usage_by_model) == 2
        assert "gpt-4" in summary.usage_by_model
        assert "claude-3-sonnet" in summary.usage_by_model

    async def test_add_usage_async_no_model(self):
        summary = TokenSummary()
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        await summary.add_usage(usage)
        assert summary.total_prompt_tokens == 100
        assert len(summary.usage_by_model) == 0

    async def test_add_usage_async_no_cost(self):
        summary = TokenSummary()
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        await summary.add_usage(usage)
        assert summary.total_estimated_cost_usd == 0.0


# ===========================================================================
# SessionMetadata tests
# ===========================================================================


class TestSessionMetadataCoverage:
    """Thorough tests for SessionMetadata."""

    def test_defaults(self):
        meta = SessionMetadata()
        assert meta.created_at is not None
        assert meta.updated_at is not None
        assert meta.properties == {}

    async def test_create_classmethod(self):
        meta = await SessionMetadata.create(properties={"foo": "bar"})
        assert meta.properties == {"foo": "bar"}
        assert meta.created_at is not None

    async def test_create_classmethod_no_properties(self):
        meta = await SessionMetadata.create()
        assert meta.properties == {}

    async def test_set_property(self):
        meta = SessionMetadata()
        old_updated = meta.updated_at
        await meta.set_property("key", "value")
        assert meta.properties["key"] == "value"
        assert meta.updated_at >= old_updated

    async def test_get_property(self):
        meta = SessionMetadata(properties={"x": 42})
        assert await meta.get_property("x") == 42
        assert await meta.get_property("missing") is None

    async def test_update_timestamp(self):
        meta = SessionMetadata()
        old_ts = meta.updated_at
        await meta.update_timestamp()
        assert meta.updated_at >= old_ts


# ===========================================================================
# SessionRun tests
# ===========================================================================


class TestSessionRunCoverage:
    """Extra coverage for SessionRun beyond existing test_models.py."""

    def test_run_status_enum_values(self):
        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"
        assert RunStatus.CANCELLED.value == "cancelled"

    async def test_create_default(self):
        run = await SessionRun.create()
        assert run.status == RunStatus.PENDING
        assert run.metadata == {}

    async def test_create_with_metadata(self):
        run = await SessionRun.create(metadata={"k": "v"})
        assert await run.get_metadata("k") == "v"

    async def test_mark_running(self):
        run = SessionRun()
        await run.mark_running()
        assert run.status == RunStatus.RUNNING

    async def test_mark_completed(self):
        run = SessionRun()
        await run.mark_running()
        await run.mark_completed()
        assert run.status == RunStatus.COMPLETED
        assert run.ended_at is not None

    async def test_mark_failed_with_reason(self):
        run = SessionRun()
        await run.mark_failed("boom")
        assert run.status == RunStatus.FAILED
        assert await run.get_metadata("failure_reason") == "boom"

    async def test_mark_failed_no_reason(self):
        run = SessionRun()
        await run.mark_failed()
        assert run.status == RunStatus.FAILED
        assert await run.has_metadata("failure_reason") is False

    async def test_mark_cancelled_with_reason(self):
        run = SessionRun()
        await run.mark_cancelled("no longer needed")
        assert run.status == RunStatus.CANCELLED
        assert await run.get_metadata("cancel_reason") == "no longer needed"

    async def test_mark_cancelled_no_reason(self):
        run = SessionRun()
        await run.mark_cancelled()
        assert run.status == RunStatus.CANCELLED
        assert await run.has_metadata("cancel_reason") is False

    async def test_get_duration_none_when_not_ended(self):
        run = SessionRun()
        d = await run.get_duration()
        assert d is None

    async def test_get_duration_after_completion(self):
        run = SessionRun()
        await run.mark_running()
        await run.mark_completed()
        d = await run.get_duration()
        assert isinstance(d, float)
        assert d >= 0

    async def test_metadata_operations(self):
        run = SessionRun()
        await run.set_metadata("a", 1)
        assert await run.get_metadata("a") == 1
        assert await run.has_metadata("a") is True
        await run.remove_metadata("a")
        assert await run.has_metadata("a") is False
        # remove_metadata on missing key is safe
        await run.remove_metadata("nonexistent")

    async def test_get_metadata_default(self):
        run = SessionRun()
        assert await run.get_metadata("missing", "fallback") == "fallback"

    async def test_add_tool_call(self):
        run = SessionRun()
        await run.add_tool_call("tc-1")
        await run.add_tool_call("tc-2")
        await run.add_tool_call("tc-1")  # duplicate
        assert run.tool_calls == ["tc-1", "tc-2"]

    async def test_get_tool_calls(self):
        run = SessionRun()
        await run.add_tool_call("evt-1")
        mock_session = MagicMock()
        mock_session.events = [
            MagicMock(id="evt-1"),
            MagicMock(id="evt-2"),
        ]
        result = await run.get_tool_calls(mock_session)
        assert len(result) == 1

    async def test_to_dict_pending(self):
        run = SessionRun()
        d = await run.to_dict()
        assert d["status"] == "pending"
        assert "ended_at" not in d
        assert "duration" not in d

    async def test_to_dict_completed(self):
        run = SessionRun()
        await run.mark_completed()
        d = await run.to_dict()
        assert d["status"] == "completed"
        assert "ended_at" in d
        assert "duration" in d


# ===========================================================================
# SessionEvent tests
# ===========================================================================


class TestSessionEventCoverage:
    """Thorough tests for SessionEvent model."""

    # --- Basic creation ---

    def test_defaults(self):
        event = SessionEvent(message="hi")
        assert event.message == "hi"
        assert event.source == EventSource.SYSTEM
        assert event.type == EventType.MESSAGE
        assert event.metadata == {}
        assert event.parent_event_id is None
        assert event.task_id is None
        assert event.token_usage is None
        assert event.id is not None
        assert event.timestamp is not None

    def test_custom_fields(self):
        event = SessionEvent(
            message="test",
            source=EventSource.USER,
            type=EventType.TOOL_CALL,
            parent_event_id="parent-1",
            task_id="task-1",
            metadata={"k": "v"},
        )
        assert event.source == EventSource.USER
        assert event.type == EventType.TOOL_CALL
        assert event.parent_event_id == "parent-1"
        assert event.task_id == "task-1"

    # --- create_with_tokens ---

    async def test_create_with_tokens_prompt_only(self):
        event = await SessionEvent.create_with_tokens(
            message="msg",
            prompt="the prompt",
            model="gpt-3.5-turbo",
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
        assert event.token_usage is not None
        assert event.token_usage.prompt_tokens > 0
        assert event.token_usage.completion_tokens == 0
        assert event.token_usage.model == "gpt-3.5-turbo"
        assert event.source == EventSource.USER

    async def test_create_with_tokens_prompt_and_completion(self):
        event = await SessionEvent.create_with_tokens(
            message="msg",
            prompt="prompt text",
            completion="completion text",
            model="gpt-4",
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
        assert event.token_usage.prompt_tokens > 0
        assert event.token_usage.completion_tokens > 0
        assert event.token_usage.model == "gpt-4"

    async def test_create_with_tokens_extra_kwargs(self):
        event = await SessionEvent.create_with_tokens(
            message="msg",
            prompt="p",
            task_id="task-xyz",
            parent_event_id="parent-xyz",
        )
        assert event.task_id == "task-xyz"
        assert event.parent_event_id == "parent-xyz"

    # --- update_token_usage ---

    async def test_update_token_usage_from_text(self):
        event = SessionEvent(message="test")
        await event.update_token_usage(
            prompt="hello", completion="world", model="gpt-4"
        )
        assert event.token_usage is not None
        assert event.token_usage.prompt_tokens > 0
        assert event.token_usage.completion_tokens > 0
        assert event.token_usage.model == "gpt-4"

    async def test_update_token_usage_prompt_only(self):
        event = SessionEvent(message="test")
        await event.update_token_usage(prompt="hello")
        assert event.token_usage is not None
        assert event.token_usage.prompt_tokens > 0

    async def test_update_token_usage_completion_only(self):
        """When completion is given but prompt is None, prompt defaults to empty."""
        event = SessionEvent(message="test")
        await event.update_token_usage(completion="some output", model="gpt-4")
        assert event.token_usage is not None
        assert event.token_usage.completion_tokens > 0

    async def test_update_token_usage_from_counts_new(self):
        """Pass prompt_tokens/completion_tokens directly on event with no prior usage."""
        event = SessionEvent(message="test")
        await event.update_token_usage(
            prompt_tokens=100, completion_tokens=50, model="gpt-4"
        )
        assert event.token_usage is not None
        assert event.token_usage.prompt_tokens == 100
        assert event.token_usage.completion_tokens == 50
        assert event.token_usage.total_tokens == 150
        assert event.token_usage.estimated_cost_usd is not None

    async def test_update_token_usage_from_counts_existing(self):
        """Pass prompt_tokens directly on event that already has token_usage."""
        event = SessionEvent(
            message="test",
            token_usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, model="gpt-4"
            ),
        )
        await event.update_token_usage(prompt_tokens=200, model="gpt-4")
        assert event.token_usage.prompt_tokens == 200
        assert event.token_usage.completion_tokens == 5
        assert event.token_usage.total_tokens == 205

    async def test_update_token_usage_completion_tokens_only(self):
        event = SessionEvent(message="test")
        await event.update_token_usage(completion_tokens=75, model="gpt-3.5-turbo")
        assert event.token_usage is not None
        assert event.token_usage.completion_tokens == 75
        assert event.token_usage.prompt_tokens == 0
        assert event.token_usage.total_tokens == 75

    async def test_update_token_usage_no_args(self):
        """When neither text nor counts are provided, nothing changes."""
        event = SessionEvent(message="test")
        await event.update_token_usage()
        assert event.token_usage is None

    # --- metadata operations ---

    async def test_set_get_metadata(self):
        event = SessionEvent(message="test")
        await event.set_metadata("key", "value")
        assert await event.get_metadata("key") == "value"

    async def test_get_metadata_default(self):
        event = SessionEvent(message="test")
        assert await event.get_metadata("missing", "fallback") == "fallback"

    async def test_has_metadata(self):
        event = SessionEvent(message="test")
        await event.set_metadata("a", 1)
        assert await event.has_metadata("a") is True
        assert await event.has_metadata("b") is False

    async def test_remove_metadata(self):
        event = SessionEvent(message="test", metadata={"x": 1, "y": 2})
        await event.remove_metadata("x")
        assert await event.has_metadata("x") is False
        assert await event.has_metadata("y") is True
        # remove nonexistent is safe
        await event.remove_metadata("zzz")

    async def test_update_metadata(self):
        event = SessionEvent(message="test")
        await event.update_metadata("key", "val")
        assert event.metadata["key"] == "val"

    async def test_merge_metadata(self):
        event = SessionEvent(message="test", metadata={"a": 1})
        await event.merge_metadata({"b": 2, "c": 3})
        assert event.metadata == {"a": 1, "b": 2, "c": 3}

    async def test_clear_metadata(self):
        event = SessionEvent(message="test", metadata={"a": 1, "b": 2})
        await event.clear_metadata()
        assert event.metadata == {}

    # --- calculate_tokens ---

    async def test_calculate_tokens_with_existing_usage(self):
        event = SessionEvent(
            message="test",
            token_usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, model="gpt-4"
            ),
        )
        result = await event.calculate_tokens()
        assert result == 15

    async def test_calculate_tokens_no_usage_string_message(self):
        event = SessionEvent(message="Hello world, this is a test")
        result = await event.calculate_tokens("gpt-3.5-turbo")
        assert result > 0

    async def test_calculate_tokens_no_usage_non_string_message(self):
        event = SessionEvent(message=12345)
        result = await event.calculate_tokens()
        assert result > 0

    # --- is_child_of / is_part_of_task ---

    def test_is_child_of_true(self):
        event = SessionEvent(message="test", parent_event_id="parent-1")
        assert event.is_child_of("parent-1") is True

    def test_is_child_of_false(self):
        event = SessionEvent(message="test", parent_event_id="parent-1")
        assert event.is_child_of("parent-2") is False

    def test_is_child_of_no_parent(self):
        event = SessionEvent(message="test")
        assert event.is_child_of("any") is False

    def test_is_part_of_task_true(self):
        event = SessionEvent(message="test", task_id="task-1")
        assert event.is_part_of_task("task-1") is True

    def test_is_part_of_task_false(self):
        event = SessionEvent(message="test", task_id="task-1")
        assert event.is_part_of_task("task-2") is False

    def test_is_part_of_task_no_task(self):
        event = SessionEvent(message="test")
        assert event.is_part_of_task("any") is False

    # --- to_dict ---

    async def test_to_dict_minimal(self):
        event = SessionEvent(message="hello")
        d = await event.to_dict()
        assert d["message"] == "hello"
        assert d["source"] == "system"
        assert d["type"] == "message"
        assert d["id"] == event.id
        assert "timestamp" in d
        assert "parent_event_id" not in d
        assert "task_id" not in d
        assert "token_usage" not in d

    async def test_to_dict_with_parent_event_id(self):
        event = SessionEvent(message="hi", parent_event_id="p-1")
        d = await event.to_dict()
        assert d["parent_event_id"] == "p-1"

    async def test_to_dict_with_task_id(self):
        event = SessionEvent(message="hi", task_id="t-1")
        d = await event.to_dict()
        assert d["task_id"] == "t-1"

    async def test_to_dict_with_token_usage(self):
        event = SessionEvent(
            message="hi",
            token_usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, model="gpt-4"
            ),
        )
        d = await event.to_dict()
        assert "token_usage" in d
        assert d["token_usage"]["prompt_tokens"] == 10
        assert d["token_usage"]["completion_tokens"] == 5
        assert d["token_usage"]["total_tokens"] == 15
        assert d["token_usage"]["model"] == "gpt-4"
        assert d["token_usage"]["estimated_cost_usd"] is not None

    async def test_to_dict_full(self):
        event = SessionEvent(
            message="full",
            source=EventSource.USER,
            type=EventType.TOOL_CALL,
            parent_event_id="p",
            task_id="t",
            metadata={"m": 1},
            token_usage=TokenUsage(prompt_tokens=5, completion_tokens=3, model="gpt-4"),
        )
        d = await event.to_dict()
        assert d["source"] == "user"
        assert d["type"] == "tool_call"
        assert d["parent_event_id"] == "p"
        assert d["task_id"] == "t"
        assert d["metadata"] == {"m": 1}
        assert "token_usage" in d


# ===========================================================================
# Session tests
# ===========================================================================


class TestSessionCoverage:
    """Thorough tests for Session model."""

    # --- Basic construction ---

    def test_session_default_construction(self):
        session = Session()
        assert session.id is not None
        assert session.metadata is not None
        assert session.parent_id is None
        assert session.child_ids == []
        assert session.task_ids == []
        assert session.runs == []
        assert session.events == []
        assert session.state == {}
        assert isinstance(session.token_summary, TokenSummary)

    def test_session_with_custom_id(self):
        session = Session(id="custom-id")
        assert session.id == "custom-id"

    def test_session_with_parent_id(self):
        session = Session(parent_id="parent-1")
        assert session.parent_id == "parent-1"

    # --- _sync_hierarchy model_validator ---

    def test_sync_hierarchy_validator_runs(self):
        """The model_validator runs but just returns the model unchanged."""
        session = Session(parent_id="some-parent")
        assert session.parent_id == "some-parent"

    # --- Properties ---

    def test_last_update_time_no_events(self):
        session = Session()
        assert session.last_update_time == session.metadata.created_at

    def test_last_update_time_with_events(self):
        session = Session()
        e1 = SessionEvent(message="a")
        e2 = SessionEvent(message="b")
        session.events.extend([e1, e2])
        assert session.last_update_time == max(e1.timestamp, e2.timestamp)

    def test_active_run_none(self):
        session = Session()
        assert session.active_run is None

    def test_active_run_with_running_run(self):
        session = Session()
        r1 = SessionRun(status=RunStatus.COMPLETED)
        r2 = SessionRun(status=RunStatus.RUNNING)
        session.runs = [r1, r2]
        assert session.active_run == r2

    def test_active_run_no_running(self):
        session = Session()
        r1 = SessionRun(status=RunStatus.COMPLETED)
        r2 = SessionRun(status=RunStatus.FAILED)
        session.runs = [r1, r2]
        assert session.active_run is None

    def test_active_run_last_running(self):
        """If multiple RUNNING, returns the last one (reversed iteration)."""
        session = Session()
        r1 = SessionRun(status=RunStatus.RUNNING)
        r2 = SessionRun(status=RunStatus.RUNNING)
        session.runs = [r1, r2]
        assert session.active_run == r2

    def test_total_tokens_property(self):
        session = Session()
        session.token_summary.total_tokens = 42
        assert session.total_tokens == 42

    def test_total_cost_property(self):
        session = Session()
        session.token_summary.total_estimated_cost_usd = 1.23
        assert session.total_cost == 1.23

    # --- async_init ---

    async def test_async_init_no_parent(self):
        session = Session()
        await session.async_init()
        # No parent, so nothing happens (no exceptions)

    async def test_async_init_with_parent_found(self):
        parent = Session(id="parent-id")
        child = Session(id="child-id", parent_id="parent-id")

        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_store.get = AsyncMock(return_value=parent)
            mock_store.save = AsyncMock()
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            await child.async_init()

            assert "child-id" in parent.child_ids
            mock_store.save.assert_called_once_with(parent)

    async def test_async_init_with_parent_not_found(self):
        child = Session(id="child-id", parent_id="parent-id")

        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_store.get = AsyncMock(return_value=None)
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            await child.async_init()
            # No error, and save not called since parent not found

    async def test_async_init_child_already_registered(self):
        parent = Session(id="parent-id", child_ids=["child-id"])
        child = Session(id="child-id", parent_id="parent-id")

        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_store.get = AsyncMock(return_value=parent)
            mock_store.save = AsyncMock()
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            await child.async_init()

            # child_id already in parent, so save should not be called
            mock_store.save.assert_not_called()

    # --- add_child / remove_child ---

    async def test_add_child(self):
        session = Session()
        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            await session.add_child("child-1")
            assert "child-1" in session.child_ids
            mock_store.save.assert_called_once()

    async def test_add_child_duplicate(self):
        session = Session(child_ids=["child-1"])
        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            await session.add_child("child-1")
            # Should not add duplicate
            assert session.child_ids.count("child-1") == 1
            mock_store.save.assert_not_called()

    async def test_remove_child(self):
        session = Session(child_ids=["child-1", "child-2"])
        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            await session.remove_child("child-1")
            assert "child-1" not in session.child_ids
            assert "child-2" in session.child_ids
            mock_store.save.assert_called_once()

    async def test_remove_child_not_present(self):
        session = Session(child_ids=["child-1"])
        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            await session.remove_child("not-here")
            assert session.child_ids == ["child-1"]
            mock_store.save.assert_not_called()

    # --- ancestors ---

    async def test_ancestors_no_parent(self):
        session = Session()
        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            result = await session.ancestors()
            assert result == []

    async def test_ancestors_chain(self):
        grandparent = Session(id="gp")
        parent = Session(id="p", parent_id="gp")
        child = Session(id="c", parent_id="p")

        store_data = {"gp": grandparent, "p": parent}

        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_store.get = AsyncMock(side_effect=lambda sid: store_data.get(sid))
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            result = await child.ancestors()
            assert len(result) == 2
            assert result[0].id == "p"
            assert result[1].id == "gp"

    async def test_ancestors_missing_parent(self):
        child = Session(id="c", parent_id="missing")

        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_store.get = AsyncMock(return_value=None)
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            result = await child.ancestors()
            assert result == []

    # --- descendants ---

    async def test_descendants_no_children(self):
        session = Session()
        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            result = await session.descendants()
            assert result == []

    async def test_descendants_tree(self):
        child1 = Session(id="c1", child_ids=["gc1"])
        child2 = Session(id="c2")
        grandchild = Session(id="gc1")
        parent = Session(id="p", child_ids=["c1", "c2"])

        store_data = {"c1": child1, "c2": child2, "gc1": grandchild}

        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_store.get = AsyncMock(side_effect=lambda sid: store_data.get(sid))
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            result = await parent.descendants()
            ids = [s.id for s in result]
            assert "c1" in ids or "c2" in ids
            assert "gc1" in ids
            assert len(result) == 3

    async def test_descendants_missing_child(self):
        parent = Session(id="p", child_ids=["missing"])

        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_store.get = AsyncMock(return_value=None)
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            result = await parent.descendants()
            assert result == []

    # --- add_event ---

    async def test_add_event_no_token_usage(self):
        session = Session()
        event = SessionEvent(message="no tokens")
        await session.add_event(event)
        assert len(session.events) == 1
        assert session.token_summary.total_tokens == 0

    async def test_add_event_with_token_usage(self):
        session = Session()
        event = SessionEvent(
            message="with tokens",
            token_usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, model="gpt-4"
            ),
        )
        await session.add_event(event)
        assert len(session.events) == 1
        assert session.token_summary.total_tokens == 15

    # --- add_event_and_save ---

    async def test_add_event_and_save(self):
        session = Session()
        event = SessionEvent(message="save me")

        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            await session.add_event_and_save(event)
            assert len(session.events) == 1
            mock_store.save.assert_called_once_with(session)

    # --- get_token_usage_by_source ---

    async def test_get_token_usage_by_source_empty(self):
        session = Session()
        result = await session.get_token_usage_by_source()
        assert result == {}

    async def test_get_token_usage_by_source_mixed(self):
        session = Session()
        e1 = SessionEvent(
            message="user msg",
            source=EventSource.USER,
            token_usage=TokenUsage(
                prompt_tokens=10, completion_tokens=0, model="gpt-4"
            ),
        )
        e2 = SessionEvent(
            message="llm msg",
            source=EventSource.LLM,
            token_usage=TokenUsage(
                prompt_tokens=0, completion_tokens=20, model="gpt-4"
            ),
        )
        e3 = SessionEvent(
            message="no tokens",
            source=EventSource.SYSTEM,
        )
        session.events = [e1, e2, e3]

        result = await session.get_token_usage_by_source()
        assert "user" in result
        assert "llm" in result
        assert "system" not in result  # no token_usage on e3

    async def test_get_token_usage_by_source_aggregates(self):
        """Multiple events from same source get aggregated."""
        session = Session()
        e1 = SessionEvent(
            message="m1",
            source=EventSource.USER,
            token_usage=TokenUsage(prompt_tokens=5, completion_tokens=3, model="gpt-4"),
        )
        e2 = SessionEvent(
            message="m2",
            source=EventSource.USER,
            token_usage=TokenUsage(
                prompt_tokens=10, completion_tokens=7, model="gpt-4"
            ),
        )
        session.events = [e1, e2]
        result = await session.get_token_usage_by_source()
        assert result["user"].total_tokens == 25  # 5+3+10+7

    async def test_get_token_usage_by_source_string_source(self):
        """Verifies source.value is used as the key (EventSource is str enum)."""
        session = Session()
        e = SessionEvent(
            message="msg",
            source=EventSource.USER,
            token_usage=TokenUsage(prompt_tokens=5, completion_tokens=3, model="gpt-4"),
        )
        session.events = [e]
        result = await session.get_token_usage_by_source()
        assert "user" in result

    # --- get_token_usage_by_run ---

    async def test_get_token_usage_by_run_empty(self):
        session = Session()
        result = await session.get_token_usage_by_run()
        assert "no_run" in result
        assert result["no_run"].total_tokens == 0

    async def test_get_token_usage_by_run_mixed(self):
        session = Session()
        e1 = SessionEvent(
            message="msg",
            task_id="run-1",
            token_usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, model="gpt-4"
            ),
        )
        e2 = SessionEvent(
            message="msg",
            task_id=None,
            token_usage=TokenUsage(
                prompt_tokens=20, completion_tokens=10, model="gpt-4"
            ),
        )
        e3 = SessionEvent(
            message="no tokens",
            task_id="run-1",
        )
        session.events = [e1, e2, e3]

        result = await session.get_token_usage_by_run()
        assert "run-1" in result
        assert "no_run" in result
        assert result["run-1"].total_tokens == 15
        assert result["no_run"].total_tokens == 30

    async def test_get_token_usage_by_run_multiple_runs(self):
        session = Session()
        e1 = SessionEvent(
            message="msg",
            task_id="run-1",
            token_usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, model="gpt-4"
            ),
        )
        e2 = SessionEvent(
            message="msg",
            task_id="run-2",
            token_usage=TokenUsage(
                prompt_tokens=20, completion_tokens=10, model="gpt-4"
            ),
        )
        session.events = [e1, e2]

        result = await session.get_token_usage_by_run()
        assert "run-1" in result
        assert "run-2" in result
        assert "no_run" in result

    # --- count_message_tokens ---

    async def test_count_message_tokens_string(self):
        session = Session()
        count = await session.count_message_tokens("Hello world!")
        assert isinstance(count, int)
        assert count > 0

    async def test_count_message_tokens_dict_with_content(self):
        session = Session()
        count = await session.count_message_tokens(
            {"role": "user", "content": "Hello world!"}, model="gpt-4"
        )
        assert count > 0

    async def test_count_message_tokens_dict_without_content(self):
        session = Session()
        count = await session.count_message_tokens({"role": "user"})
        # Falls through to str() conversion
        assert count > 0

    async def test_count_message_tokens_other_object(self):
        session = Session()
        count = await session.count_message_tokens(12345)
        assert count > 0

    # --- state management ---

    async def test_set_state(self):
        session = Session()
        await session.set_state("key", "value")
        assert session.state["key"] == "value"

    async def test_get_state(self):
        session = Session(state={"x": 42})
        assert await session.get_state("x") == 42

    async def test_get_state_default(self):
        session = Session()
        assert await session.get_state("missing", "default") == "default"

    async def test_has_state_true(self):
        session = Session(state={"a": 1})
        assert await session.has_state("a") is True

    async def test_has_state_false(self):
        session = Session()
        assert await session.has_state("missing") is False

    async def test_remove_state(self):
        session = Session(state={"a": 1, "b": 2})
        await session.remove_state("a")
        assert "a" not in session.state
        assert "b" in session.state

    async def test_remove_state_nonexistent(self):
        session = Session()
        await session.remove_state("nothing")
        # No error

    # --- Session.create classmethod ---

    async def test_create_no_args(self):
        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_store.get = AsyncMock(return_value=None)
            mock_store.save = AsyncMock()
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            session = await Session.create()
            assert session.id is not None
            assert session.parent_id is None
            mock_store.save.assert_called()

    async def test_create_with_session_id(self):
        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_store.get = AsyncMock(return_value=None)
            mock_store.save = AsyncMock()
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            session = await Session.create(session_id="my-id")
            assert session.id == "my-id"

    async def test_create_with_parent_id(self):
        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_store.get = AsyncMock(return_value=None)
            mock_store.save = AsyncMock()
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            session = await Session.create(parent_id="parent-1")
            assert session.parent_id == "parent-1"

    async def test_create_with_session_id_and_parent(self):
        parent = Session(id="parent-1")
        with (
            patch("chuk_ai_session_manager.session_storage.get_backend") as mock_gb,
            patch(
                "chuk_ai_session_manager.session_storage.ChukSessionsStore"
            ) as mock_sc,
        ):
            mock_store = AsyncMock()
            mock_store.get = AsyncMock(return_value=parent)
            mock_store.save = AsyncMock()
            mock_gb.return_value = AsyncMock()
            mock_sc.return_value = mock_store

            session = await Session.create(session_id="child-1", parent_id="parent-1")
            assert session.id == "child-1"
            assert session.parent_id == "parent-1"
            assert "child-1" in parent.child_ids


# ===========================================================================
# models/__init__.py tests
# ===========================================================================


class TestModelsInit:
    """Tests for models/__init__.py import handling and __all__."""

    def test_all_exports_present(self):
        """Verify that __all__ contains expected names when imports succeed."""
        import chuk_ai_session_manager.models as models_pkg

        for name in [
            "EventSource",
            "EventType",
            "SessionEvent",
            "SessionMetadata",
            "SessionRun",
            "RunStatus",
            "Session",
        ]:
            assert name in models_pkg.__all__, f"{name} not in __all__"
            assert hasattr(models_pkg, name), f"{name} not accessible on module"

    def test_event_source_accessible(self):
        from chuk_ai_session_manager.models import EventSource as ES

        assert ES.USER == "user"

    def test_event_type_accessible(self):
        from chuk_ai_session_manager.models import EventType as ET

        assert ET.MESSAGE == "message"

    def test_session_event_accessible(self):
        from chuk_ai_session_manager.models import SessionEvent as SE

        event = SE(message="hi")
        assert event.message == "hi"

    def test_session_metadata_accessible(self):
        from chuk_ai_session_manager.models import SessionMetadata as SM

        meta = SM()
        assert meta.properties == {}

    def test_session_run_accessible(self):
        from chuk_ai_session_manager.models import SessionRun as SR

        run = SR()
        assert run.status == RunStatus.PENDING

    def test_run_status_accessible(self):
        from chuk_ai_session_manager.models import RunStatus as RS

        assert RS.PENDING == "pending"

    def test_session_accessible(self):
        from chuk_ai_session_manager.models import Session as S

        s = S()
        assert s.events == []

    def test_import_failure_event_source(self):
        """When event_source import fails, module still loads without crashing."""
        with patch.dict(
            sys.modules, {"chuk_ai_session_manager.models.event_source": None}
        ):
            import chuk_ai_session_manager.models as models_pkg

            importlib.reload(models_pkg)

    def test_import_failure_event_type(self):
        with patch.dict(
            sys.modules, {"chuk_ai_session_manager.models.event_type": None}
        ):
            import chuk_ai_session_manager.models as models_pkg

            importlib.reload(models_pkg)

    def test_import_failure_session_event(self):
        with patch.dict(
            sys.modules, {"chuk_ai_session_manager.models.session_event": None}
        ):
            import chuk_ai_session_manager.models as models_pkg

            importlib.reload(models_pkg)

    def test_import_failure_session_metadata(self):
        with patch.dict(
            sys.modules, {"chuk_ai_session_manager.models.session_metadata": None}
        ):
            import chuk_ai_session_manager.models as models_pkg

            importlib.reload(models_pkg)

    def test_import_failure_session_run(self):
        with patch.dict(
            sys.modules, {"chuk_ai_session_manager.models.session_run": None}
        ):
            import chuk_ai_session_manager.models as models_pkg

            importlib.reload(models_pkg)

    def test_import_failure_session(self):
        with patch.dict(sys.modules, {"chuk_ai_session_manager.models.session": None}):
            import chuk_ai_session_manager.models as models_pkg

            importlib.reload(models_pkg)

    def test_reload_restores_all(self):
        """After failures, reloading recovers everything."""
        import chuk_ai_session_manager.models as models_pkg

        importlib.reload(models_pkg)
        for name in [
            "EventSource",
            "EventType",
            "SessionEvent",
            "SessionMetadata",
            "SessionRun",
            "RunStatus",
            "Session",
        ]:
            assert name in models_pkg.__all__
