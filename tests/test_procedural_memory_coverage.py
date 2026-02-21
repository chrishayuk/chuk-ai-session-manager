# tests/test_procedural_memory_coverage.py
"""
Comprehensive tests for the procedural_memory subsystem.

Covers:
  - procedural_memory/models.py   (ToolOutcome, ToolFixRelation, ToolLogEntry,
                                    ErrorPattern, SuccessPattern, ToolPattern,
                                    ProceduralMemory, _abbrev, _describe_fix)
  - procedural_memory/manager.py  (ToolMemoryManager)
  - procedural_memory/formatter.py (ProceduralContextFormatter, FormatterConfig)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chuk_ai_session_manager.procedural_memory.formatter import (
    FormatterConfig,
    ProceduralContextFormatter,
)
from chuk_ai_session_manager.procedural_memory.manager import ToolMemoryManager
from chuk_ai_session_manager.procedural_memory.models import (
    ErrorPattern,
    ProceduralMemory,
    SuccessPattern,
    ToolFixRelation,
    ToolLogEntry,
    ToolOutcome,
    ToolPattern,
    _abbrev,
    _describe_fix,
)


# ---------------------------------------------------------------------------
# Helper: lightweight mock Session used by manager tests
# ---------------------------------------------------------------------------
class MockSession:
    """Minimal session mock with async get_state / set_state."""

    def __init__(self, session_id: str = "test-session"):
        self.id = session_id
        self._state: dict = {}

    async def get_state(self, key: str):
        return self._state.get(key)

    async def set_state(self, key: str, value):
        self._state[key] = value


# ===========================================================================
# 1. models.py
# ===========================================================================


class TestToolOutcome:
    """ToolOutcome enum values."""

    def test_success_value(self):
        assert ToolOutcome.SUCCESS == "success"

    def test_failure_value(self):
        assert ToolOutcome.FAILURE == "failure"

    def test_partial_value(self):
        assert ToolOutcome.PARTIAL == "partial"

    def test_timeout_value(self):
        assert ToolOutcome.TIMEOUT == "timeout"

    def test_cancelled_value(self):
        assert ToolOutcome.CANCELLED == "cancelled"

    def test_is_str_subclass(self):
        assert isinstance(ToolOutcome.SUCCESS, str)


class TestToolFixRelation:
    """ToolFixRelation creation and defaults."""

    def test_creation_minimal(self):
        rel = ToolFixRelation(failed_call_id="c1", success_call_id="c2")
        assert rel.failed_call_id == "c1"
        assert rel.success_call_id == "c2"
        assert rel.delta_args == {}

    def test_creation_with_delta(self):
        delta = {"added": {"key": "val"}}
        rel = ToolFixRelation(failed_call_id="c1", success_call_id="c2", delta_args=delta)
        assert rel.delta_args == delta


class TestToolLogEntry:
    """ToolLogEntry methods and formatting."""

    @staticmethod
    def _make_entry(**overrides) -> ToolLogEntry:
        defaults = {
            "id": "call-0001",
            "tool_name": "solver",
            "outcome": ToolOutcome.SUCCESS,
            "result_summary": "ok",
        }
        defaults.update(overrides)
        return ToolLogEntry(**defaults)

    # -- predicates --

    def test_is_failure_with_failure(self):
        e = self._make_entry(outcome=ToolOutcome.FAILURE)
        assert e.is_failure() is True

    def test_is_failure_with_timeout(self):
        e = self._make_entry(outcome=ToolOutcome.TIMEOUT)
        assert e.is_failure() is True

    def test_is_failure_with_success(self):
        e = self._make_entry(outcome=ToolOutcome.SUCCESS)
        assert e.is_failure() is False

    def test_is_success_true(self):
        e = self._make_entry(outcome=ToolOutcome.SUCCESS)
        assert e.is_success() is True

    def test_is_success_false(self):
        e = self._make_entry(outcome=ToolOutcome.FAILURE)
        assert e.is_success() is False

    def test_was_fixed_true(self):
        e = self._make_entry(fixed_by="call-0002")
        assert e.was_fixed() is True

    def test_was_fixed_false(self):
        e = self._make_entry()
        assert e.was_fixed() is False

    def test_is_fix_true(self):
        e = self._make_entry(fix_for="call-0000")
        assert e.is_fix() is True

    def test_is_fix_false(self):
        e = self._make_entry()
        assert e.is_fix() is False

    # -- format_compact --

    def test_format_compact_success(self):
        e = self._make_entry(outcome=ToolOutcome.SUCCESS, result_summary="solved")
        text = e.format_compact()
        assert "solver" in text
        assert "solved" in text

    def test_format_compact_failure(self):
        e = self._make_entry(outcome=ToolOutcome.FAILURE, result_summary="unsat")
        text = e.format_compact()
        assert "unsat" in text

    # -- format_for_context --

    def test_format_for_context_basic(self):
        e = self._make_entry(result_summary="ok")
        text = e.format_for_context()
        assert "ok" in text

    def test_format_for_context_include_args(self):
        e = self._make_entry(arguments={"x": 1, "y": "hello"})
        text = e.format_for_context(include_args=True)
        assert "args:" in text
        assert "x=" in text

    def test_format_for_context_no_args_when_empty(self):
        e = self._make_entry(arguments={})
        text = e.format_for_context(include_args=True)
        assert "args:" not in text

    def test_format_for_context_fix_info(self):
        e = self._make_entry(
            fix_for="call-0000",
            delta_args={"changed": {"x": {"from": 1, "to": 2}}},
        )
        text = e.format_for_context()
        assert "fixed prior failure" in text

    def test_format_for_context_was_fixed(self):
        e = self._make_entry(outcome=ToolOutcome.FAILURE, fixed_by="call-0002")
        text = e.format_for_context()
        assert "later fixed by" in text
        assert "call-0002" in text

    def test_format_for_context_no_delta_args_on_fix(self):
        """is_fix() True but delta_args is None -- no fix line emitted."""
        e = self._make_entry(fix_for="call-0000", delta_args=None)
        text = e.format_for_context()
        assert "fixed prior failure" not in text


class TestErrorPattern:
    """ErrorPattern creation and count increment."""

    def test_creation_defaults(self):
        ep = ErrorPattern(error_type="unsat")
        assert ep.error_type == "unsat"
        assert ep.count == 1
        assert ep.contexts == []
        assert ep.example_args is None
        assert ep.typical_fix is None
        assert ep.fix_delta is None

    def test_count_increment(self):
        ep = ErrorPattern(error_type="timeout", count=3)
        ep.count += 1
        assert ep.count == 4


class TestSuccessPattern:
    """SuccessPattern creation with various fields."""

    def test_creation_defaults(self):
        sp = SuccessPattern()
        assert sp.goal_match is None
        assert sp.arg_hints == {}
        assert sp.notes is None
        assert sp.example_call_id is None
        assert sp.delta_that_fixed is None

    def test_creation_with_all_fields(self):
        sp = SuccessPattern(
            goal_match="find flights",
            arg_hints={"must_include": ["origin"]},
            notes="use IATA codes",
            example_call_id="call-0003",
            delta_that_fixed={"changed": {"x": {"from": 1, "to": 2}}},
        )
        assert sp.goal_match == "find flights"
        assert sp.delta_that_fixed is not None


class TestToolPattern:
    """ToolPattern: record_call, add_error/success_pattern, record_fix, format."""

    @staticmethod
    def _entry(outcome=ToolOutcome.SUCCESS, args=None, time_ms=None) -> ToolLogEntry:
        return ToolLogEntry(
            id="call-0001",
            tool_name="solver",
            outcome=outcome,
            result_summary="ok",
            arguments=args or {},
            execution_time_ms=time_ms,
        )

    # -- record_call --

    def test_record_call_success(self):
        p = ToolPattern(tool_name="solver")
        p.record_call(self._entry(ToolOutcome.SUCCESS))
        assert p.total_calls == 1
        assert p.success_count == 1
        assert p.failure_count == 0

    def test_record_call_failure(self):
        p = ToolPattern(tool_name="solver")
        p.record_call(self._entry(ToolOutcome.FAILURE))
        assert p.failure_count == 1

    def test_record_call_partial_not_counted(self):
        """PARTIAL increments total but not success/failure."""
        p = ToolPattern(tool_name="solver")
        p.record_call(self._entry(ToolOutcome.PARTIAL))
        assert p.total_calls == 1
        assert p.success_count == 0
        assert p.failure_count == 0

    def test_record_call_timing_initial(self):
        p = ToolPattern(tool_name="solver")
        p.record_call(self._entry(time_ms=100))
        assert p.avg_execution_ms == 100.0

    def test_record_call_timing_running_average(self):
        p = ToolPattern(tool_name="solver")
        p.record_call(self._entry(time_ms=100))
        p.record_call(self._entry(time_ms=200))
        # After 2 calls: (100*1 + 200)/2 = 150
        assert p.avg_execution_ms == pytest.approx(150.0)

    def test_record_call_timing_none_ignored(self):
        p = ToolPattern(tool_name="solver")
        p.record_call(self._entry(time_ms=None))
        assert p.avg_execution_ms is None

    def test_record_call_tracks_common_args(self):
        p = ToolPattern(tool_name="solver")
        p.record_call(self._entry(args={"x": 1}))
        assert "x" in p.common_args
        assert 1 in p.common_args["x"]

    def test_record_call_deduplicates_common_args(self):
        p = ToolPattern(tool_name="solver")
        p.record_call(self._entry(args={"x": 1}))
        p.record_call(self._entry(args={"x": 1}))
        assert p.common_args["x"] == [1]

    def test_record_call_common_args_overflow(self):
        """Only keep last 5 unique values per arg."""
        p = ToolPattern(tool_name="solver")
        for i in range(7):
            p.record_call(self._entry(args={"x": i}))
        assert len(p.common_args["x"]) == 5
        # Earliest values dropped
        assert 0 not in p.common_args["x"]
        assert 6 in p.common_args["x"]

    # -- add_error_pattern --

    def test_add_error_pattern_new(self):
        p = ToolPattern(tool_name="solver")
        ep = p.add_error_pattern("unsat", context="goal1", example_args={"x": 1})
        assert ep.error_type == "unsat"
        assert ep.count == 1
        assert "goal1" in ep.contexts

    def test_add_error_pattern_existing_increments(self):
        p = ToolPattern(tool_name="solver")
        p.add_error_pattern("unsat", context="goal1")
        ep = p.add_error_pattern("unsat", context="goal2")
        assert ep.count == 2
        assert "goal1" in ep.contexts
        assert "goal2" in ep.contexts

    def test_add_error_pattern_existing_no_context(self):
        p = ToolPattern(tool_name="solver")
        p.add_error_pattern("unsat", context="a")
        ep = p.add_error_pattern("unsat", context=None)
        assert ep.count == 2

    def test_add_error_pattern_context_overflow(self):
        """Only keep last 5 contexts."""
        p = ToolPattern(tool_name="solver")
        p.add_error_pattern("unsat", context="c0")
        for i in range(1, 7):
            p.add_error_pattern("unsat", context=f"c{i}")
        ep = p.error_patterns[0]
        assert len(ep.contexts) == 5
        assert "c0" not in ep.contexts

    def test_add_error_pattern_no_context(self):
        p = ToolPattern(tool_name="solver")
        ep = p.add_error_pattern("unsat")
        assert ep.contexts == []

    # -- add_success_pattern --

    def test_add_success_pattern_basic(self):
        p = ToolPattern(tool_name="solver")
        sp = p.add_success_pattern(goal_match="find", arg_hints={"x": 1})
        assert sp.goal_match == "find"
        assert sp.arg_hints == {"x": 1}

    def test_add_success_pattern_with_delta(self):
        p = ToolPattern(tool_name="solver")
        sp = p.add_success_pattern(delta_that_fixed={"added": {"y": 2}}, example_call_id="c1")
        assert sp.delta_that_fixed is not None
        assert sp.example_call_id == "c1"

    def test_add_success_pattern_overflow(self):
        """Only keep last 10 patterns."""
        p = ToolPattern(tool_name="solver")
        for i in range(12):
            p.add_success_pattern(goal_match=f"g{i}")
        assert len(p.success_patterns) == 10
        # First two removed
        assert p.success_patterns[0].goal_match == "g2"

    # -- record_fix --

    def test_record_fix_updates_error_pattern(self):
        p = ToolPattern(tool_name="solver")
        p.add_error_pattern("unsat")
        p.record_fix("unsat", {"changed": {"x": {"from": 1, "to": 2}}})
        ep = p.error_patterns[0]
        assert ep.fix_delta is not None
        assert ep.typical_fix is not None

    def test_record_fix_no_match_does_nothing(self):
        p = ToolPattern(tool_name="solver")
        p.add_error_pattern("unsat")
        p.record_fix("timeout", {"added": {"y": 1}})
        ep = p.error_patterns[0]
        assert ep.fix_delta is None

    # -- format_for_context --

    def test_format_for_context_basic(self):
        p = ToolPattern(tool_name="solver")
        p.record_call(self._entry(ToolOutcome.SUCCESS))
        text = p.format_for_context()
        assert "Success rate:" in text

    def test_format_for_context_with_errors(self):
        p = ToolPattern(tool_name="solver")
        p.record_call(self._entry(ToolOutcome.FAILURE))
        p.add_error_pattern("unsat")
        text = p.format_for_context()
        assert "Common errors:" in text
        assert "unsat" in text

    def test_format_for_context_with_fix_hint(self):
        p = ToolPattern(tool_name="solver")
        p.add_error_pattern("unsat")
        p.record_fix("unsat", {"added": {"y": 1}})
        text = p.format_for_context()
        assert "add y" in text

    def test_format_for_context_with_success_delta(self):
        p = ToolPattern(tool_name="solver")
        p.add_success_pattern(delta_that_fixed={"added": {"z": 1}})
        text = p.format_for_context()
        assert "Fixed by:" in text

    def test_format_for_context_with_success_arg_hints(self):
        p = ToolPattern(tool_name="solver")
        p.add_success_pattern(arg_hints={"key": "val"})
        text = p.format_for_context()
        assert "Args:" in text

    # -- success_rate --

    def test_success_rate_zero_calls(self):
        p = ToolPattern(tool_name="solver")
        assert p.success_rate == 0.0

    def test_success_rate_with_calls(self):
        p = ToolPattern(tool_name="solver", total_calls=4, success_count=3)
        assert p.success_rate == pytest.approx(0.75)


class TestProceduralMemory:
    """ProceduralMemory container."""

    def test_get_pattern_new(self):
        pm = ProceduralMemory(session_id="s1")
        pat = pm.get_pattern("tool_x")
        assert pat.tool_name == "tool_x"
        assert "tool_x" in pm.tool_patterns

    def test_get_pattern_existing(self):
        pm = ProceduralMemory(session_id="s1")
        p1 = pm.get_pattern("tool_x")
        p2 = pm.get_pattern("tool_x")
        assert p1 is p2

    def test_allocate_call_id_sequential(self):
        pm = ProceduralMemory(session_id="s1")
        assert pm.allocate_call_id() == "call-0001"
        assert pm.allocate_call_id() == "call-0002"
        assert pm.next_call_id == 3


class TestAbbrev:
    """Module-level _abbrev helper."""

    def test_short_string(self):
        assert _abbrev("hello") == "hello"

    def test_long_string(self):
        result = _abbrev("a" * 30, max_len=20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_exact_boundary(self):
        assert _abbrev("a" * 20, max_len=20) == "a" * 20


class TestDescribeFix:
    """Module-level _describe_fix helper."""

    def test_added_keys(self):
        assert "add" in _describe_fix({"added": {"y": 1, "z": 2}})

    def test_removed_keys(self):
        assert "remove" in _describe_fix({"removed": ["x", "y"]})

    def test_changed_keys(self):
        result = _describe_fix({"changed": {"x": {"from": 1, "to": 2}}})
        assert "change" in result

    def test_empty_delta(self):
        assert _describe_fix({}) == "unknown changes"

    def test_combined(self):
        delta = {
            "added": {"a": 1},
            "removed": ["b"],
            "changed": {"c": {"from": 0, "to": 9}},
        }
        result = _describe_fix(delta)
        assert "add" in result
        assert "remove" in result
        assert "change" in result


# ===========================================================================
# 2. manager.py
# ===========================================================================


class TestToolMemoryManagerCreate:
    """ToolMemoryManager.create() class method."""

    def test_create_basic(self):
        mgr = ToolMemoryManager.create("sess-1")
        assert mgr.session_id == "sess-1"
        assert len(mgr.memory.tool_log) == 0

    def test_create_with_kwargs(self):
        mgr = ToolMemoryManager.create("sess-1", max_log_entries=50)
        assert mgr.max_log_entries == 50


class TestToolMemoryManagerFromSession:
    """ToolMemoryManager.from_session()."""

    async def test_from_session_no_stored_state(self):
        session = MockSession("s1")
        mgr = await ToolMemoryManager.from_session(session)
        assert mgr.session_id == "s1"

    async def test_from_session_with_stored_state(self):
        session = MockSession("s1")
        pm = ProceduralMemory(session_id="s1")
        pm.allocate_call_id()
        await session.set_state("procedural_memory", pm.model_dump(mode="json"))
        mgr = await ToolMemoryManager.from_session(session)
        assert mgr.memory.next_call_id == 2

    async def test_from_session_corrupt_state_falls_back(self):
        session = MockSession("s1")
        await session.set_state("procedural_memory", {"bad": "data"})
        mgr = await ToolMemoryManager.from_session(session)
        # Should fall back to fresh
        assert mgr.session_id == "s1"


class TestToolMemoryManagerSave:
    """ToolMemoryManager.save_to_session()."""

    async def test_save_and_restore(self):
        session = MockSession("s1")
        mgr = ToolMemoryManager.create("s1")
        await mgr.record_call("t", {}, "ok", ToolOutcome.SUCCESS)
        await mgr.save_to_session(session)

        mgr2 = await ToolMemoryManager.from_session(session)
        assert len(mgr2.memory.tool_log) == 1


class TestToolMemoryManagerSessionIdProperty:
    def test_session_id(self):
        mgr = ToolMemoryManager.create("abc")
        assert mgr.session_id == "abc"


class TestToolMemoryManagerRecordCall:
    """ToolMemoryManager.record_call() -- success, failure, timeout, fix detection."""

    async def test_record_success(self):
        mgr = ToolMemoryManager.create("s")
        entry = await mgr.record_call("t", {"x": 1}, "ok", ToolOutcome.SUCCESS)
        assert entry.outcome == ToolOutcome.SUCCESS
        assert entry.tool_name == "t"

    async def test_record_failure(self):
        mgr = ToolMemoryManager.create("s")
        entry = await mgr.record_call(
            "t",
            {},
            None,
            ToolOutcome.FAILURE,
            error_type="unsat",
            error_message="no solution",
        )
        assert entry.is_failure()
        pat = mgr.get_pattern("t")
        assert len(pat.error_patterns) == 1

    async def test_record_timeout(self):
        mgr = ToolMemoryManager.create("s")
        entry = await mgr.record_call("t", {}, None, ToolOutcome.TIMEOUT)
        assert entry.is_failure()
        pat = mgr.get_pattern("t")
        assert pat.error_patterns[0].error_type == "unknown"

    async def test_record_with_context_goal(self):
        mgr = ToolMemoryManager.create("s")
        entry = await mgr.record_call("t", {}, "ok", ToolOutcome.SUCCESS, context_goal="find answer")
        assert entry.context_goal == "find answer"

    async def test_record_with_execution_time(self):
        mgr = ToolMemoryManager.create("s")
        entry = await mgr.record_call("t", {}, "ok", ToolOutcome.SUCCESS, execution_time_ms=42)
        assert entry.execution_time_ms == 42

    async def test_record_with_preceding_call_id(self):
        mgr = ToolMemoryManager.create("s")
        entry = await mgr.record_call("t", {}, "ok", ToolOutcome.SUCCESS, preceding_call_id="call-0000")
        assert entry.preceding_call_id == "call-0000"

    async def test_max_log_enforced(self):
        mgr = ToolMemoryManager.create("s", max_log_entries=3)
        for i in range(5):
            await mgr.record_call("t", {"i": i}, "ok", ToolOutcome.SUCCESS)
        assert len(mgr.memory.tool_log) == 3

    async def test_fix_detection(self):
        """A success after a failure with changed args is detected as a fix."""
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="unsat")
        entry = await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        assert entry.is_fix()
        assert mgr.memory.tool_log[0].was_fixed()
        assert len(mgr.memory.fix_relations) == 1

    async def test_fix_detection_callback(self):
        cb = MagicMock()
        mgr = ToolMemoryManager.create("s", on_fix_detected=cb)
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="e")
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        cb.assert_called_once()

    async def test_fix_adds_success_pattern(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="e")
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS, context_goal="goal")
        pat = mgr.get_pattern("t")
        assert len(pat.success_patterns) >= 1


class TestCheckIfFixesPrior:
    """_check_if_fixes_prior edge cases."""

    async def test_no_fix_different_tool(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("a", {"x": 1}, None, ToolOutcome.FAILURE, error_type="e")
        entry = await mgr.record_call("b", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        assert not entry.is_fix()

    async def test_no_fix_already_fixed(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="e")
        # First fix
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        # Second success should not re-fix the already-fixed call
        entry = await mgr.record_call("t", {"x": 3}, "ok", ToolOutcome.SUCCESS)
        assert not entry.is_fix()

    async def test_no_fix_when_args_identical(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="e")
        entry = await mgr.record_call("t", {"x": 1}, "ok", ToolOutcome.SUCCESS)
        # Same args means no delta -> no fix
        assert not entry.is_fix()

    async def test_fix_records_to_error_pattern(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="unsat")
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        pat = mgr.get_pattern("t")
        ep = pat.error_patterns[0]
        assert ep.fix_delta is not None


class TestComputeArgDelta:
    """_compute_arg_delta: added, removed, changed, none."""

    def test_added_keys(self):
        mgr = ToolMemoryManager.create("s")
        delta = mgr._compute_arg_delta({"a": 1}, {"a": 1, "b": 2})
        assert "added" in delta
        assert "b" in delta["added"]

    def test_removed_keys(self):
        mgr = ToolMemoryManager.create("s")
        delta = mgr._compute_arg_delta({"a": 1, "b": 2}, {"a": 1})
        assert "removed" in delta
        assert "b" in delta["removed"]

    def test_changed_keys(self):
        mgr = ToolMemoryManager.create("s")
        delta = mgr._compute_arg_delta({"a": 1}, {"a": 2})
        assert "changed" in delta
        assert delta["changed"]["a"]["from"] == 1
        assert delta["changed"]["a"]["to"] == 2

    def test_no_changes_returns_none(self):
        mgr = ToolMemoryManager.create("s")
        assert mgr._compute_arg_delta({"a": 1}, {"a": 1}) is None


class TestValuesEqual:
    """_values_equal: equal, not equal, exception fallback."""

    def test_equal(self):
        mgr = ToolMemoryManager.create("s")
        assert mgr._values_equal({"a": 1}, {"a": 1}) is True

    def test_not_equal(self):
        mgr = ToolMemoryManager.create("s")
        assert mgr._values_equal(1, 2) is False

    def test_exception_fallback(self):
        """Non-serialisable objects fall back to ==."""
        mgr = ToolMemoryManager.create("s")

        class Obj:
            def __eq__(self, other):
                return True

        assert mgr._values_equal(Obj(), Obj()) is True

    def test_exception_fallback_not_equal(self):
        mgr = ToolMemoryManager.create("s")

        class Obj:
            def __eq__(self, other):
                return False

        assert mgr._values_equal(Obj(), Obj()) is False


class TestHashArguments:
    """_hash_arguments: normal and exception."""

    def test_normal(self):
        mgr = ToolMemoryManager.create("s")
        h = mgr._hash_arguments({"a": 1})
        assert isinstance(h, str)
        assert len(h) == 12

    def test_deterministic(self):
        mgr = ToolMemoryManager.create("s")
        assert mgr._hash_arguments({"a": 1}) == mgr._hash_arguments({"a": 1})

    def test_exception_path(self):
        """Non-serialisable objects that fail even with default=str."""
        mgr = ToolMemoryManager.create("s")

        class BadObj:
            def __repr__(self):
                raise TypeError("no repr")

            def __str__(self):
                raise TypeError("no str")

        # default=str calls str() which raises TypeError, caught by except
        result = mgr._hash_arguments({"x": BadObj()})
        # Should return "" on exception
        assert result == ""


class TestSummarizeResult:
    """_summarize_result: various types."""

    def _mgr(self):
        return ToolMemoryManager.create("s")

    def test_none(self):
        assert self._mgr()._summarize_result(None) == "null"

    def test_short_string(self):
        assert self._mgr()._summarize_result("hello") == "hello"

    def test_long_string(self):
        s = "x" * 200
        result = self._mgr()._summarize_result(s)
        assert len(result) <= 100
        assert result.endswith("...")

    def test_bool_true(self):
        assert self._mgr()._summarize_result(True) == "true"

    def test_bool_false(self):
        assert self._mgr()._summarize_result(False) == "false"

    def test_int(self):
        assert self._mgr()._summarize_result(42) == "42"

    def test_float(self):
        assert self._mgr()._summarize_result(3.14) == "3.14"

    def test_list(self):
        assert self._mgr()._summarize_result([1, 2, 3]) == "list[3]"

    def test_dict_status(self):
        assert self._mgr()._summarize_result({"status": "ok"}) == "status=ok"

    def test_dict_error(self):
        result = self._mgr()._summarize_result({"error": "boom"})
        assert "boom" in result

    def test_dict_result_nested(self):
        result = self._mgr()._summarize_result({"result": "inner"})
        assert result == "inner"

    def test_dict_plain(self):
        result = self._mgr()._summarize_result({"a": 1, "b": 2})
        assert "2 keys" in result

    def test_other_type(self):
        result = self._mgr()._summarize_result(object())
        assert result == "object"


class TestClassifyResult:
    """_classify_result: all types."""

    def _mgr(self):
        return ToolMemoryManager.create("s")

    def test_none(self):
        assert self._mgr()._classify_result(None) == "null"

    def test_bool(self):
        assert self._mgr()._classify_result(True) == "boolean"

    def test_int(self):
        assert self._mgr()._classify_result(42) == "number"

    def test_float(self):
        assert self._mgr()._classify_result(3.14) == "number"

    def test_string(self):
        assert self._mgr()._classify_result("hi") == "string"

    def test_list(self):
        assert self._mgr()._classify_result([1]) == "list"

    def test_dict_with_status(self):
        assert self._mgr()._classify_result({"status": "ok"}) == "ok"

    def test_dict_plain(self):
        assert self._mgr()._classify_result({"a": 1}) == "object"

    def test_unknown(self):
        assert self._mgr()._classify_result(object()) == "unknown"


class TestGetRecentCalls:
    """get_recent_calls: all, by tool, by outcome."""

    async def test_get_all(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("a", {}, "ok", ToolOutcome.SUCCESS)
        await mgr.record_call("b", {}, "ok", ToolOutcome.SUCCESS)
        calls = mgr.get_recent_calls(limit=10)
        assert len(calls) == 2

    async def test_filter_by_tool(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("a", {}, "ok", ToolOutcome.SUCCESS)
        await mgr.record_call("b", {}, "ok", ToolOutcome.SUCCESS)
        calls = mgr.get_recent_calls(tool_name="a")
        assert len(calls) == 1
        assert calls[0].tool_name == "a"

    async def test_filter_by_outcome(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("a", {}, "ok", ToolOutcome.SUCCESS)
        await mgr.record_call("a", {}, None, ToolOutcome.FAILURE, error_type="e")
        calls = mgr.get_recent_calls(outcome=ToolOutcome.FAILURE)
        assert len(calls) == 1

    async def test_most_recent_first(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("a", {}, "first", ToolOutcome.SUCCESS)
        await mgr.record_call("a", {}, "second", ToolOutcome.SUCCESS)
        calls = mgr.get_recent_calls(limit=10)
        assert calls[0].result_summary == "second"


class TestGetPattern:
    """get_pattern and get_all_patterns."""

    async def test_get_pattern_none(self):
        mgr = ToolMemoryManager.create("s")
        assert mgr.get_pattern("nonexistent") is None

    async def test_get_pattern_existing(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {}, "ok", ToolOutcome.SUCCESS)
        pat = mgr.get_pattern("t")
        assert pat is not None
        assert pat.total_calls == 1

    async def test_get_all_patterns(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("a", {}, "ok", ToolOutcome.SUCCESS)
        await mgr.record_call("b", {}, "ok", ToolOutcome.SUCCESS)
        pats = mgr.get_all_patterns()
        assert "a" in pats
        assert "b" in pats


class TestSearchCalls:
    """search_calls with various filters."""

    async def _setup_mgr(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call(
            "t",
            {"x": 1},
            None,
            ToolOutcome.FAILURE,
            error_type="unsat",
            context_goal="find flights",
        )
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS, context_goal="find flights")
        await mgr.record_call("u", {}, "ok", ToolOutcome.SUCCESS, context_goal="book hotel")
        return mgr

    async def test_no_filters(self):
        mgr = await self._setup_mgr()
        results = mgr.search_calls(limit=100)
        assert len(results) == 3

    async def test_filter_tool_name(self):
        mgr = await self._setup_mgr()
        results = mgr.search_calls(tool_name="t")
        assert all(r.tool_name == "t" for r in results)

    async def test_filter_goal_contains(self):
        mgr = await self._setup_mgr()
        results = mgr.search_calls(goal_contains="flight")
        assert len(results) == 2

    async def test_filter_goal_contains_no_goal(self):
        """Entries with no goal are skipped when goal_contains is set."""
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {}, "ok", ToolOutcome.SUCCESS)
        results = mgr.search_calls(goal_contains="anything")
        assert len(results) == 0

    async def test_filter_outcome(self):
        mgr = await self._setup_mgr()
        results = mgr.search_calls(outcome=ToolOutcome.FAILURE)
        assert len(results) == 1

    async def test_filter_error_type(self):
        mgr = await self._setup_mgr()
        results = mgr.search_calls(error_type="unsat")
        assert len(results) == 1

    async def test_only_fixes(self):
        mgr = await self._setup_mgr()
        results = mgr.search_calls(only_fixes=True)
        # call-0002 fixed call-0001
        assert len(results) == 1
        assert results[0].is_fix()

    async def test_only_fixed(self):
        mgr = await self._setup_mgr()
        results = mgr.search_calls(only_fixed=True)
        assert len(results) == 1
        assert results[0].was_fixed()

    async def test_limit(self):
        mgr = await self._setup_mgr()
        results = mgr.search_calls(limit=1)
        assert len(results) == 1


class TestGetFixForError:
    """get_fix_for_error: found and not found."""

    async def test_found(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="unsat")
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        fix = mgr.get_fix_for_error("t", "unsat")
        assert fix is not None

    async def test_not_found_no_pattern(self):
        mgr = ToolMemoryManager.create("s")
        assert mgr.get_fix_for_error("t", "unsat") is None

    async def test_not_found_no_matching_error(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="unsat")
        assert mgr.get_fix_for_error("t", "timeout") is None


class TestGetSuccessfulArgsForGoal:
    """get_successful_args_for_goal: found and not found."""

    async def test_found(self):
        mgr = ToolMemoryManager.create("s")
        # Create a pattern with a success pattern
        pat = mgr.memory.get_pattern("t")
        pat.add_success_pattern(goal_match="find flights", arg_hints={"origin": "SFO"})
        result = mgr.get_successful_args_for_goal("t", "find flights")
        assert result == {"origin": "SFO"}

    async def test_not_found_no_pattern(self):
        mgr = ToolMemoryManager.create("s")
        assert mgr.get_successful_args_for_goal("t", "goal") is None

    async def test_not_found_no_matching_goal(self):
        mgr = ToolMemoryManager.create("s")
        pat = mgr.memory.get_pattern("t")
        pat.add_success_pattern(goal_match="find hotels", arg_hints={"city": "NYC"})
        assert mgr.get_successful_args_for_goal("t", "find flights") is None

    async def test_partial_match(self):
        """Goal search is substring-based."""
        mgr = ToolMemoryManager.create("s")
        pat = mgr.memory.get_pattern("t")
        pat.add_success_pattern(goal_match="FIND FLIGHTS", arg_hints={"x": 1})
        # "find" is in "FIND FLIGHTS" (case-insensitive)
        result = mgr.get_successful_args_for_goal("t", "find")
        assert result == {"x": 1}


class TestGetStats:
    """get_stats."""

    async def test_stats_empty(self):
        mgr = ToolMemoryManager.create("s")
        stats = mgr.get_stats()
        assert stats["total_calls"] == 0
        assert stats["success_rate"] == 0

    async def test_stats_with_calls(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {}, "ok", ToolOutcome.SUCCESS)
        await mgr.record_call("t", {}, None, ToolOutcome.FAILURE, error_type="e")
        stats = mgr.get_stats()
        assert stats["total_calls"] == 2
        assert stats["success_count"] == 1
        assert stats["failure_count"] == 1
        assert stats["tools_tracked"] == 1
        assert stats["session_id"] == "s"
        assert "created_at" in stats
        assert "updated_at" in stats
        assert stats["total_fixes_detected"] == 0


class TestToDictFromDict:
    """to_dict / from_dict roundtrip."""

    async def test_roundtrip(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"a": 1}, "ok", ToolOutcome.SUCCESS)
        data = mgr.to_dict()
        mgr2 = ToolMemoryManager.from_dict(data)
        assert mgr2.session_id == "s"
        assert len(mgr2.memory.tool_log) == 1

    def test_from_dict_kwargs(self):
        pm = ProceduralMemory(session_id="s")
        data = pm.model_dump(mode="json")
        mgr = ToolMemoryManager.from_dict(data, max_log_entries=50)
        assert mgr.max_log_entries == 50


class TestReset:
    """reset() clears everything."""

    async def test_reset(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="e")
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        mgr.reset()
        assert len(mgr.memory.tool_log) == 0
        assert len(mgr.memory.tool_patterns) == 0
        assert len(mgr.memory.fix_relations) == 0
        assert mgr.memory.next_call_id == 1


# ===========================================================================
# 3. formatter.py
# ===========================================================================


class TestFormatterConfig:
    """FormatterConfig default values."""

    def test_defaults(self):
        cfg = FormatterConfig()
        assert cfg.max_recent_calls == 3
        assert cfg.max_error_patterns == 3
        assert cfg.max_success_patterns == 3
        assert cfg.include_args is False
        assert cfg.include_timing is False
        assert cfg.show_fix_relations is True
        assert cfg.compact is False


class TestFormatForTools:
    """format_for_tools: main entry point."""

    async def test_tools_with_history(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {}, "ok", ToolOutcome.SUCCESS)
        fmt = ProceduralContextFormatter()
        result = fmt.format_for_tools(mgr, ["t"])
        assert "<procedural_memory>" in result
        assert "</procedural_memory>" in result
        assert "tool_memory" in result

    async def test_tools_with_no_history(self):
        mgr = ToolMemoryManager.create("s")
        fmt = ProceduralContextFormatter()
        result = fmt.format_for_tools(mgr, ["unknown_tool"])
        assert result == ""

    async def test_empty_tool_list(self):
        mgr = ToolMemoryManager.create("s")
        fmt = ProceduralContextFormatter()
        result = fmt.format_for_tools(mgr, [])
        assert result == ""

    async def test_multiple_tools(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("a", {}, "ok", ToolOutcome.SUCCESS)
        await mgr.record_call("b", {}, "ok", ToolOutcome.SUCCESS)
        fmt = ProceduralContextFormatter()
        result = fmt.format_for_tools(mgr, ["a", "b"])
        assert result.count("tool_memory") == 4  # open + close for each


class TestFormatToolSection:
    """_format_tool_section: with/without history and patterns."""

    async def test_with_recent_calls_and_pattern(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {}, "ok", ToolOutcome.SUCCESS)
        fmt = ProceduralContextFormatter()
        section = fmt._format_tool_section(mgr, "t")
        assert section is not None
        assert "tool_memory" in section

    async def test_no_history(self):
        mgr = ToolMemoryManager.create("s")
        fmt = ProceduralContextFormatter()
        assert fmt._format_tool_section(mgr, "nope") is None

    async def test_with_pattern_text(self):
        """Pattern section included when there are error/success patterns."""
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="unsat")
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        fmt = ProceduralContextFormatter()
        section = fmt._format_tool_section(mgr, "t")
        assert "patterns" in section

    async def test_pattern_no_errors_no_successes(self):
        """Pattern section omitted when no error/success patterns."""
        mgr = ToolMemoryManager.create("s")
        # PARTIAL does not create error/success patterns
        await mgr.record_call("t", {}, "ok", ToolOutcome.PARTIAL)
        fmt = ProceduralContextFormatter()
        section = fmt._format_tool_section(mgr, "t")
        # Pattern exists (total_calls=1) but no error/success patterns
        # _format_pattern returns None, so no patterns block
        assert section is not None
        assert "<patterns>" not in section


class TestFormatRecentCalls:
    """_format_recent_calls: success, failure, fix relations, args, timing."""

    def _entry(self, **kw) -> ToolLogEntry:
        defaults = {
            "id": "call-0001",
            "tool_name": "t",
            "outcome": ToolOutcome.SUCCESS,
            "result_summary": "ok",
        }
        defaults.update(kw)
        return ToolLogEntry(**defaults)

    def test_success_entry(self):
        fmt = ProceduralContextFormatter()
        text = fmt._format_recent_calls([self._entry()])
        assert "recent_calls" in text

    def test_failure_entry(self):
        fmt = ProceduralContextFormatter()
        e = self._entry(outcome=ToolOutcome.FAILURE, result_summary="fail")
        text = fmt._format_recent_calls([e])
        assert "fail" in text

    def test_fix_relation_shown(self):
        fmt = ProceduralContextFormatter(config=FormatterConfig(show_fix_relations=True))
        e = self._entry(
            fix_for="call-0000",
            delta_args={"changed": {"x": {"from": 1, "to": 2}}},
        )
        text = fmt._format_recent_calls([e])
        assert "fixed prior" in text

    def test_was_fixed_shown(self):
        fmt = ProceduralContextFormatter(config=FormatterConfig(show_fix_relations=True))
        e = self._entry(outcome=ToolOutcome.FAILURE, fixed_by="call-0002")
        text = fmt._format_recent_calls([e])
        assert "fixed by call-0002" in text

    def test_fix_relations_hidden(self):
        fmt = ProceduralContextFormatter(config=FormatterConfig(show_fix_relations=False))
        e = self._entry(
            fix_for="call-0000",
            delta_args={"changed": {"x": {"from": 1, "to": 2}}},
        )
        text = fmt._format_recent_calls([e])
        assert "fixed prior" not in text

    def test_include_args(self):
        fmt = ProceduralContextFormatter(config=FormatterConfig(include_args=True))
        e = self._entry(arguments={"x": 1, "y": 2})
        text = fmt._format_recent_calls([e])
        assert "args:" in text

    def test_include_args_empty(self):
        fmt = ProceduralContextFormatter(config=FormatterConfig(include_args=True))
        e = self._entry(arguments={})
        text = fmt._format_recent_calls([e])
        assert "args:" not in text

    def test_include_timing(self):
        fmt = ProceduralContextFormatter(config=FormatterConfig(include_timing=True))
        e = self._entry(execution_time_ms=123)
        text = fmt._format_recent_calls([e])
        assert "123ms" in text

    def test_timing_none_omitted(self):
        fmt = ProceduralContextFormatter(config=FormatterConfig(include_timing=True))
        e = self._entry(execution_time_ms=None)
        text = fmt._format_recent_calls([e])
        assert "ms]" not in text


class TestFormatPattern:
    """_format_pattern: errors, successes, compact, no patterns."""

    def test_no_patterns(self):
        fmt = ProceduralContextFormatter()
        pat = ToolPattern(tool_name="t", total_calls=1)
        assert fmt._format_pattern(pat) is None

    def test_with_errors(self):
        fmt = ProceduralContextFormatter()
        pat = ToolPattern(tool_name="t", total_calls=1)
        pat.add_error_pattern("unsat")
        text = fmt._format_pattern(pat)
        assert "common_errors" in text
        assert "unsat" in text

    def test_with_error_count_gt_1(self):
        fmt = ProceduralContextFormatter()
        pat = ToolPattern(tool_name="t", total_calls=2)
        pat.add_error_pattern("unsat")
        pat.add_error_pattern("unsat")
        text = fmt._format_pattern(pat)
        assert "x2" in text

    def test_with_error_fix(self):
        fmt = ProceduralContextFormatter()
        pat = ToolPattern(tool_name="t", total_calls=1)
        pat.add_error_pattern("unsat")
        pat.record_fix("unsat", {"added": {"y": 1}})
        text = fmt._format_pattern(pat)
        assert "fix:" in text

    def test_with_successes_delta(self):
        fmt = ProceduralContextFormatter()
        pat = ToolPattern(tool_name="t", total_calls=1)
        pat.add_success_pattern(delta_that_fixed={"added": {"z": 1}})
        text = fmt._format_pattern(pat)
        assert "success_hints" in text
        assert "Fixed by:" in text

    def test_with_successes_arg_hints(self):
        fmt = ProceduralContextFormatter()
        pat = ToolPattern(tool_name="t", total_calls=1)
        pat.add_success_pattern(arg_hints={"key": "val"})
        text = fmt._format_pattern(pat)
        assert "Typical args:" in text

    def test_compact_mode(self):
        fmt = ProceduralContextFormatter(config=FormatterConfig(compact=True))
        pat = ToolPattern(tool_name="t", total_calls=5, success_count=3)
        pat.add_error_pattern("unsat")
        text = fmt._format_pattern(pat)
        # In compact mode, success_rate line is omitted
        assert "success_rate:" not in text

    def test_non_compact_shows_success_rate(self):
        fmt = ProceduralContextFormatter(config=FormatterConfig(compact=False))
        pat = ToolPattern(tool_name="t", total_calls=5, success_count=3)
        pat.add_error_pattern("unsat")
        text = fmt._format_pattern(pat)
        assert "success_rate:" in text

    def test_with_context_goal_filtering(self):
        fmt = ProceduralContextFormatter()
        pat = ToolPattern(tool_name="t", total_calls=1)
        pat.add_success_pattern(goal_match="find flights", arg_hints={"x": 1})
        text = fmt._format_pattern(pat, context_goal="find flights")
        assert text is not None


class TestFilterRelevantSuccesses:
    """_filter_relevant_successes: no goal, matching, fix patterns."""

    def test_no_goal_returns_all(self):
        fmt = ProceduralContextFormatter()
        patterns = [
            SuccessPattern(goal_match="a"),
            SuccessPattern(goal_match="b"),
        ]
        result = fmt._filter_relevant_successes(patterns, None)
        assert len(result) == 2

    def test_matching_goal(self):
        fmt = ProceduralContextFormatter()
        patterns = [
            SuccessPattern(goal_match="find flights"),
            SuccessPattern(goal_match="book hotel"),
        ]
        result = fmt._filter_relevant_successes(patterns, "Find Flights")
        assert len(result) == 1

    def test_no_goal_match_on_pattern(self):
        """Pattern with no goal_match is always included."""
        fmt = ProceduralContextFormatter()
        patterns = [SuccessPattern(goal_match=None)]
        result = fmt._filter_relevant_successes(patterns, "anything")
        assert len(result) == 1

    def test_fix_pattern_always_included(self):
        fmt = ProceduralContextFormatter()
        patterns = [
            SuccessPattern(
                goal_match="unrelated",
                delta_that_fixed={"added": {"y": 1}},
            ),
        ]
        result = fmt._filter_relevant_successes(patterns, "something else")
        assert len(result) == 1

    def test_non_matching_excluded(self):
        fmt = ProceduralContextFormatter()
        patterns = [
            SuccessPattern(goal_match="book hotel"),
        ]
        result = fmt._filter_relevant_successes(patterns, "find flights")
        assert len(result) == 0


class TestFormatDelta:
    """_format_delta: added, removed, changed -- few/many."""

    def _fmt(self):
        return ProceduralContextFormatter()

    def test_added_few(self):
        result = self._fmt()._format_delta({"added": {"x": 1, "y": 2}})
        assert "+x" in result or "+y" in result

    def test_added_many(self):
        added = {f"k{i}": i for i in range(5)}
        result = self._fmt()._format_delta({"added": added})
        assert "+5 args" in result

    def test_removed_few(self):
        result = self._fmt()._format_delta({"removed": ["a", "b"]})
        assert "-a" in result or "-b" in result

    def test_removed_many(self):
        result = self._fmt()._format_delta({"removed": ["a", "b", "c"]})
        assert "-3 args" in result

    def test_changed_few(self):
        result = self._fmt()._format_delta({"changed": {"x": {"from": 1, "to": 2}}})
        assert "x:" in result
        assert "->" in result

    def test_changed_many(self):
        changed = {f"k{i}": {"from": i, "to": i + 1} for i in range(5)}
        result = self._fmt()._format_delta({"changed": changed})
        assert "~5 args" in result

    def test_empty_delta(self):
        result = self._fmt()._format_delta({})
        assert result == "changes"


class TestFormatArgs:
    """_format_args: few items, many items."""

    def _fmt(self):
        return ProceduralContextFormatter()

    def test_few_items(self):
        result = self._fmt()._format_args({"a": 1, "b": 2})
        assert "a=" in result
        assert "b=" in result

    def test_many_items(self):
        args = {f"k{i}": i for i in range(10)}
        result = self._fmt()._format_args(args, max_items=3)
        assert "...+" in result


class TestFormatterAbbrev:
    """ProceduralContextFormatter._abbrev."""

    def test_short(self):
        fmt = ProceduralContextFormatter()
        assert fmt._abbrev("hi") == "hi"

    def test_long(self):
        fmt = ProceduralContextFormatter()
        result = fmt._abbrev("a" * 30, max_len=15)
        assert len(result) == 15
        assert result.endswith("..")


class TestFormatFullSummary:
    """format_full_summary: with stats, tools, no tools."""

    async def test_with_tools(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="e")
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        fmt = ProceduralContextFormatter()
        text = fmt.format_full_summary(mgr)
        assert "procedural_memory_summary" in text
        assert "Most used tools:" in text
        assert "last error:" in text

    async def test_no_tools(self):
        mgr = ToolMemoryManager.create("s")
        fmt = ProceduralContextFormatter()
        text = fmt.format_full_summary(mgr)
        assert "procedural_memory_summary" in text
        assert "Most used tools:" not in text

    async def test_stats_line(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {}, "ok", ToolOutcome.SUCCESS)
        fmt = ProceduralContextFormatter()
        text = fmt.format_full_summary(mgr)
        assert "Total:" in text
        assert "1 calls" in text

    async def test_max_tools_limit(self):
        mgr = ToolMemoryManager.create("s")
        for i in range(10):
            await mgr.record_call(f"tool_{i}", {}, "ok", ToolOutcome.SUCCESS)
        fmt = ProceduralContextFormatter()
        text = fmt.format_full_summary(mgr, max_tools=2)
        # Only 2 tools listed
        count = text.count("tool_")
        assert count <= 4  # tool name appears in line + possibly elsewhere

    async def test_tool_without_errors(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {}, "ok", ToolOutcome.SUCCESS)
        fmt = ProceduralContextFormatter()
        text = fmt.format_full_summary(mgr)
        assert "last error:" not in text


class TestFormatErrorGuidance:
    """format_error_guidance: with fix, similar past fixes, no fixes."""

    async def test_with_fix(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="unsat")
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        fmt = ProceduralContextFormatter()
        text = fmt.format_error_guidance(mgr, "t", "unsat")
        assert "Previous fix:" in text
        assert "error_guidance" in text

    async def test_with_similar_past_fixes(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="unsat")
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        fmt = ProceduralContextFormatter()
        text = fmt.format_error_guidance(mgr, "t", "unsat")
        assert "Past fixes:" in text

    async def test_with_past_fix_delta_args(self):
        """Check that the delta_args on the *failed* entry are shown (they may be None)."""
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="unsat")
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        fmt = ProceduralContextFormatter()
        text = fmt.format_error_guidance(mgr, "t", "unsat")
        # The failed entry has fixed_by set, so it appears in "Past fixes"
        assert "call-0001" in text

    async def test_no_fixes(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {}, None, ToolOutcome.FAILURE, error_type="unsat")
        fmt = ProceduralContextFormatter()
        text = fmt.format_error_guidance(mgr, "t", "timeout")
        assert "No previous fixes found" in text

    async def test_no_pattern_at_all(self):
        mgr = ToolMemoryManager.create("s")
        fmt = ProceduralContextFormatter()
        text = fmt.format_error_guidance(mgr, "t", "unsat")
        assert "No previous fixes found" in text


class TestFormatSuccessTemplate:
    """format_success_template: with hints, without hints."""

    async def test_with_hints(self):
        mgr = ToolMemoryManager.create("s")
        pat = mgr.memory.get_pattern("t")
        pat.add_success_pattern(goal_match="find flights", arg_hints={"origin": "SFO", "dest": "LAX"})
        fmt = ProceduralContextFormatter()
        text = fmt.format_success_template(mgr, "t", "find flights")
        assert text is not None
        assert "success_template" in text
        assert "origin" in text
        assert "dest" in text

    async def test_without_hints(self):
        mgr = ToolMemoryManager.create("s")
        fmt = ProceduralContextFormatter()
        result = fmt.format_success_template(mgr, "t", "find flights")
        assert result is None

    async def test_no_matching_goal(self):
        mgr = ToolMemoryManager.create("s")
        pat = mgr.memory.get_pattern("t")
        pat.add_success_pattern(goal_match="book hotel", arg_hints={"city": "NYC"})
        fmt = ProceduralContextFormatter()
        result = fmt.format_success_template(mgr, "t", "find flights")
        assert result is None


# ===========================================================================
# Additional edge-case and integration tests
# ===========================================================================


class TestManagerEdgeCases:
    """Extra edge-case coverage for manager internals."""

    async def test_record_call_result_types_through_manager(self):
        """Ensure all result types flow through record_call correctly."""
        mgr = ToolMemoryManager.create("s")
        e1 = await mgr.record_call("t", {}, None, ToolOutcome.SUCCESS)
        assert e1.result_summary == "null"
        assert e1.result_type == "null"

        e2 = await mgr.record_call("t", {}, True, ToolOutcome.SUCCESS)
        assert e2.result_summary == "true"
        assert e2.result_type == "boolean"

        e3 = await mgr.record_call("t", {}, [1, 2], ToolOutcome.SUCCESS)
        assert e3.result_summary == "list[2]"
        assert e3.result_type == "list"

    async def test_fix_detection_window(self):
        """Fix detection only looks back fix_detection_window entries."""
        mgr = ToolMemoryManager.create("s", fix_detection_window=2)
        # Record a failure far back
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="e")
        # Add enough successes to push it out of the window
        for _i in range(5):
            await mgr.record_call("other", {}, "ok", ToolOutcome.SUCCESS)
        # Now a success for the same tool should NOT detect a fix
        entry = await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        assert not entry.is_fix()

    async def test_multiple_fix_relations(self):
        """Multiple failures can each be fixed by separate successes."""
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="e")
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        await mgr.record_call("t", {"y": 1}, None, ToolOutcome.FAILURE, error_type="e")
        entry = await mgr.record_call("t", {"y": 2}, "ok", ToolOutcome.SUCCESS)
        assert entry.is_fix()
        assert len(mgr.memory.fix_relations) == 2

    async def test_summarize_dict_result_key(self):
        """Dict with 'result' key recursively summarises."""
        mgr = ToolMemoryManager.create("s")
        entry = await mgr.record_call("t", {}, {"result": {"status": "ok"}}, ToolOutcome.SUCCESS)
        assert entry.result_summary == "status=ok"

    async def test_classify_dict_with_status_value(self):
        mgr = ToolMemoryManager.create("s")
        entry = await mgr.record_call("t", {}, {"status": "completed"}, ToolOutcome.SUCCESS)
        assert entry.result_type == "completed"

    async def test_error_type_defaults_to_unknown_on_failure(self):
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {}, None, ToolOutcome.FAILURE)
        pat = mgr.get_pattern("t")
        assert pat.error_patterns[0].error_type == "unknown"

    async def test_from_session_with_extra_kwargs(self):
        session = MockSession("s1")
        mgr = await ToolMemoryManager.from_session(session, max_log_entries=50)
        assert mgr.max_log_entries == 50


class TestFormatterEdgeCases:
    """Extra edge-case coverage for formatter."""

    async def test_format_recent_calls_fix_without_delta(self):
        """is_fix() True but delta_args is None."""
        fmt = ProceduralContextFormatter(config=FormatterConfig(show_fix_relations=True))
        e = ToolLogEntry(
            id="c1",
            tool_name="t",
            outcome=ToolOutcome.SUCCESS,
            result_summary="ok",
            fix_for="c0",
            delta_args=None,
        )
        text = fmt._format_recent_calls([e])
        assert "fixed prior" not in text

    async def test_format_for_tools_with_context_goal(self):
        mgr = ToolMemoryManager.create("s")
        pat = mgr.memory.get_pattern("t")
        pat.add_success_pattern(goal_match="find flights", arg_hints={"x": 1})
        pat.total_calls = 1
        # Need a recent call for section to be non-empty
        await mgr.record_call("t", {}, "ok", ToolOutcome.SUCCESS)
        fmt = ProceduralContextFormatter()
        result = fmt.format_for_tools(mgr, ["t"], context_goal="find flights")
        assert "<procedural_memory>" in result

    async def test_format_pattern_relevant_successes_empty(self):
        """Success patterns exist but none are relevant."""
        fmt = ProceduralContextFormatter()
        pat = ToolPattern(tool_name="t", total_calls=1)
        pat.add_success_pattern(goal_match="hotel booking", arg_hints={"a": 1})
        text = fmt._format_pattern(pat, context_goal="totally different")
        # Pattern has success, but the filtered list is empty.
        # _format_pattern still includes the patterns block for errors.
        # With only non-relevant successes and no errors, the relevant list is empty.
        # Let's also add an error so the block exists.
        assert text is None or "success_hints" not in text

    async def test_format_delta_combined(self):
        fmt = ProceduralContextFormatter()
        delta = {
            "added": {"a": 1},
            "removed": ["b"],
            "changed": {"c": {"from": 0, "to": 9}},
        }
        result = fmt._format_delta(delta)
        assert ";" in result  # parts joined by ;

    def test_format_args_with_abbreviation(self):
        fmt = ProceduralContextFormatter()
        result = fmt._format_args({"key": "a" * 30})
        assert ".." in result

    async def test_format_error_guidance_fixed_entry_without_delta(self):
        """Fixed entry in Past fixes but no delta_args on the failed entry."""
        mgr = ToolMemoryManager.create("s")
        # Manually create a failure that was fixed without delta_args
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="unsat")
        # Fix it (this sets delta_args on the *success* entry, not on the failure)
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        # The failed entry has fixed_by but delta_args is None on it
        failed = mgr.memory.tool_log[0]
        assert failed.fixed_by is not None
        assert failed.delta_args is None  # delta is on success, not failure
        fmt = ProceduralContextFormatter()
        text = fmt.format_error_guidance(mgr, "t", "unsat")
        # Should still show the fix entry without crashing
        assert "Past fixes:" in text

    async def test_format_error_guidance_fixed_entry_with_delta_on_failed(self):
        """Fixed entry in Past fixes WITH delta_args set on the failed entry."""
        mgr = ToolMemoryManager.create("s")
        await mgr.record_call("t", {"x": 1}, None, ToolOutcome.FAILURE, error_type="unsat")
        await mgr.record_call("t", {"x": 2}, "ok", ToolOutcome.SUCCESS)
        # Manually set delta_args on the failed entry to exercise line 376
        failed = mgr.memory.tool_log[0]
        failed.delta_args = {"changed": {"x": {"from": 1, "to": 2}}}
        fmt = ProceduralContextFormatter()
        text = fmt.format_error_guidance(mgr, "t", "unsat")
        assert "changed:" in text
