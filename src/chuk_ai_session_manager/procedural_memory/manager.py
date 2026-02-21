# chuk_ai_session_manager/procedural_memory/manager.py
"""
Tool Memory Manager - orchestrates procedural memory operations.

Handles:
- Recording tool invocations
- Detecting fix relationships (failure -> success)
- Updating aggregated patterns
- Searching and retrieving history
- Session persistence integration
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from chuk_ai_session_manager.procedural_memory.models import (
    DEFAULT_ERROR_TYPE,
    DeltaChangeKey,
    DeltaKey,
    ProceduralMemory,
    ResultType,
    ToolFixRelation,
    ToolLogEntry,
    ToolMemoryStats,
    ToolOutcome,
    ToolPattern,
)

if TYPE_CHECKING:
    from chuk_ai_session_manager.models.session import Session

log = logging.getLogger(__name__)


class ToolMemoryManager(BaseModel):
    """
    Manages procedural memory - tool traces, patterns, and fixes.

    This is the main interface for:
    - Recording tool calls and their outcomes
    - Detecting when a success fixes a prior failure
    - Building aggregated patterns from traces
    - Retrieving relevant history for context injection
    - Persisting to/from Session state
    """

    memory: ProceduralMemory

    # Configuration
    max_log_entries: int = Field(default=1000)
    max_patterns_per_tool: int = Field(default=10)
    fix_detection_window: int = Field(default=10)  # Look back N calls for fixes

    # Optional callbacks
    on_fix_detected: Callable[[ToolFixRelation], None] | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def create(cls, session_id: str, **kwargs: Any) -> ToolMemoryManager:
        """Create a new manager for a session."""
        memory = ProceduralMemory(session_id=session_id)
        return cls(memory=memory, **kwargs)

    @classmethod
    async def from_session(cls, session: Session, **kwargs: Any) -> ToolMemoryManager:
        """
        Load procedural memory from a session's state.

        Args:
            session: The Session to load from
            **kwargs: Additional configuration

        Returns:
            ToolMemoryManager with restored state
        """
        state_key = "procedural_memory"
        stored = await session.get_state(state_key)

        if stored:
            try:
                memory = ProceduralMemory.model_validate(stored)
                log.info(f"Restored procedural memory for session {session.id}")
                return cls(memory=memory, **kwargs)
            except Exception as e:
                log.warning(f"Failed to restore procedural memory: {e}")

        # Create fresh
        return cls.create(session_id=session.id, **kwargs)

    async def save_to_session(self, session: Session) -> None:
        """
        Persist procedural memory to a session's state.

        Args:
            session: The Session to save to
        """
        state_key = "procedural_memory"
        await session.set_state(state_key, self.memory.model_dump(mode="json"))
        log.debug(f"Saved procedural memory to session {session.id}")

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self.memory.session_id

    # --- Recording ---

    async def record_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        outcome: ToolOutcome,
        context_goal: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        execution_time_ms: int | None = None,
        preceding_call_id: str | None = None,
    ) -> ToolLogEntry:
        """
        Record a tool invocation.

        Args:
            tool_name: Name of the tool called
            arguments: Arguments passed to the tool
            result: The result (will be summarized)
            outcome: Success/failure status
            context_goal: What the user was trying to accomplish
            error_type: Type of error if failure
            error_message: Error details if failure
            execution_time_ms: How long the call took
            preceding_call_id: ID of the call that came before this in a chain

        Returns:
            The created ToolLogEntry
        """
        call_id = self.memory.allocate_call_id()

        entry = ToolLogEntry(
            id=call_id,
            timestamp=datetime.now(UTC),
            tool_name=tool_name,
            arguments=arguments,
            arguments_hash=self._hash_arguments(arguments),
            context_goal=context_goal,
            preceding_call_id=preceding_call_id,
            outcome=outcome,
            result_summary=self._summarize_result(result),
            result_type=self._classify_result(result),
            execution_time_ms=execution_time_ms,
            error_type=error_type,
            error_message=error_message,
        )

        # Add to log
        self.memory.tool_log.append(entry)

        # Enforce max size
        if len(self.memory.tool_log) > self.max_log_entries:
            self.memory.tool_log.pop(0)

        # Update patterns
        pattern = self.memory.get_pattern(tool_name)
        pattern.record_call(entry)

        # Handle success - check if it fixes a prior failure
        if outcome == ToolOutcome.SUCCESS:
            await self._check_if_fixes_prior(entry)

            # Record success pattern if it was a fix
            if entry.is_fix():
                pattern.add_success_pattern(
                    goal_match=context_goal,
                    delta_that_fixed=entry.delta_args,
                    example_call_id=call_id,
                )

        # Handle failure - record error pattern
        elif outcome in (ToolOutcome.FAILURE, ToolOutcome.TIMEOUT):
            pattern.add_error_pattern(
                error_type=error_type or DEFAULT_ERROR_TYPE,
                context=context_goal,
                example_args=arguments,
            )

        self.memory.updated_at = datetime.now(UTC)

        log.debug(f"Recorded tool call: {entry.format_compact()}")
        return entry

    async def _check_if_fixes_prior(self, success_entry: ToolLogEntry) -> None:
        """
        Check if this successful call fixes a recent failure.

        Looks back through recent calls for the same tool that failed,
        and links them if found.
        """
        # Look at recent calls (excluding this one)
        window = self.memory.tool_log[-(self.fix_detection_window + 1) : -1]

        for prior in reversed(window):
            # Must be same tool
            if prior.tool_name != success_entry.tool_name:
                continue

            # Must be a failure
            if not prior.is_failure():
                continue

            # Must not already be fixed
            if prior.was_fixed():
                continue

            # Compute what changed
            delta = self._compute_arg_delta(prior.arguments, success_entry.arguments)

            if delta:
                # Link the fix
                prior.fixed_by = success_entry.id
                success_entry.fix_for = prior.id
                success_entry.delta_args = delta

                # Record the fix relation
                relation = ToolFixRelation(
                    failed_call_id=prior.id,
                    success_call_id=success_entry.id,
                    delta_args=delta,
                )
                self.memory.fix_relations.append(relation)

                # Update error pattern with fix info
                if prior.error_type:
                    pattern = self.memory.get_pattern(success_entry.tool_name)
                    pattern.record_fix(prior.error_type, delta)

                log.info(f"Detected fix: {prior.id} -> {success_entry.id}, delta: {delta}")

                # Callback
                if self.on_fix_detected:
                    self.on_fix_detected(relation)

                # Only link to most recent failure
                break

    def _compute_arg_delta(self, failed_args: dict[str, Any], success_args: dict[str, Any]) -> dict[str, Any] | None:
        """Compute what changed between failed and successful call."""
        delta: dict[str, Any] = {}

        failed_keys = set(failed_args.keys())
        success_keys = set(success_args.keys())

        # Added keys
        added = success_keys - failed_keys
        if added:
            delta[DeltaKey.ADDED] = {k: success_args[k] for k in added}

        # Removed keys
        removed = failed_keys - success_keys
        if removed:
            delta[DeltaKey.REMOVED] = list(removed)

        # Changed values
        changed = {}
        for k in failed_keys & success_keys:
            if not self._values_equal(failed_args[k], success_args[k]):
                changed[k] = {
                    DeltaChangeKey.FROM: failed_args[k],
                    DeltaChangeKey.TO: success_args[k],
                }
        if changed:
            delta[DeltaKey.CHANGED] = changed

        return delta if delta else None

    def _values_equal(self, a: Any, b: Any) -> bool:
        """Check if two values are equal (handling nested structures)."""
        try:
            return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        except (TypeError, ValueError):
            return a == b

    def _hash_arguments(self, arguments: dict[str, Any]) -> str:
        """Create hash of arguments for quick comparison."""
        try:
            args_str = json.dumps(arguments, sort_keys=True, default=str)
            return hashlib.sha256(args_str.encode()).hexdigest()[:12]
        except (TypeError, ValueError):
            return ""

    def _summarize_result(self, result: Any, max_len: int = 100) -> str:
        """Create compact summary of a result."""
        if result is None:
            return "null"

        if isinstance(result, str):
            if len(result) <= max_len:
                return result
            return result[: max_len - 3] + "..."

        if isinstance(result, bool):
            return "true" if result else "false"

        if isinstance(result, (int, float)):
            return str(result)

        if isinstance(result, list):
            return f"list[{len(result)}]"

        if isinstance(result, dict):
            # Check for common patterns
            if "status" in result:
                return f"status={result['status']}"
            if "error" in result:
                return f"error: {result['error']}"
            if "result" in result:
                return self._summarize_result(result["result"], max_len)
            return f"object[{len(result)} keys]"

        return str(type(result).__name__)

    def _classify_result(self, result: Any) -> str:
        """Classify the type of result."""
        if result is None:
            return ResultType.NULL.value
        if isinstance(result, bool):
            return ResultType.BOOLEAN.value
        if isinstance(result, (int, float)):
            return ResultType.NUMBER.value
        if isinstance(result, str):
            return ResultType.STRING.value
        if isinstance(result, list):
            return ResultType.LIST.value
        if isinstance(result, dict):
            if "status" in result:
                return result.get("status", ResultType.OBJECT.value)
            return ResultType.OBJECT.value
        return ResultType.UNKNOWN.value

    # --- Retrieval ---

    def get_recent_calls(
        self,
        tool_name: str | None = None,
        limit: int = 5,
        outcome: ToolOutcome | None = None,
    ) -> list[ToolLogEntry]:
        """
        Get recent tool calls.

        Args:
            tool_name: Filter by tool name (None = all tools)
            limit: Maximum entries to return
            outcome: Filter by outcome (None = all outcomes)

        Returns:
            List of entries, most recent first
        """
        entries = self.memory.tool_log

        if tool_name:
            entries = [e for e in entries if e.tool_name == tool_name]

        if outcome:
            entries = [e for e in entries if e.outcome == outcome]

        return list(reversed(entries[-limit:]))

    def get_pattern(self, tool_name: str) -> ToolPattern | None:
        """Get aggregated pattern for a tool."""
        return self.memory.tool_patterns.get(tool_name)

    def get_all_patterns(self) -> dict[str, ToolPattern]:
        """Get all tool patterns."""
        return self.memory.tool_patterns.copy()

    def search_calls(
        self,
        tool_name: str | None = None,
        goal_contains: str | None = None,
        outcome: ToolOutcome | None = None,
        error_type: str | None = None,
        only_fixes: bool = False,
        only_fixed: bool = False,
        limit: int = 10,
    ) -> list[ToolLogEntry]:
        """
        Search tool log with filters.

        Args:
            tool_name: Filter by tool name
            goal_contains: Filter by goal containing text
            outcome: Filter by outcome
            error_type: Filter by error type
            only_fixes: Only return calls that fixed something
            only_fixed: Only return failures that were fixed
            limit: Maximum results

        Returns:
            Matching entries, most recent first
        """
        results = []

        for entry in reversed(self.memory.tool_log):
            # Apply filters
            if tool_name and entry.tool_name != tool_name:
                continue

            if goal_contains:
                if not entry.context_goal:
                    continue
                if goal_contains.lower() not in entry.context_goal.lower():
                    continue

            if outcome and entry.outcome != outcome:
                continue

            if error_type and entry.error_type != error_type:
                continue

            if only_fixes and not entry.is_fix():
                continue

            if only_fixed and not entry.was_fixed():
                continue

            results.append(entry)

            if len(results) >= limit:
                break

        return results

    def get_fix_for_error(self, tool_name: str, error_type: str) -> dict[str, Any] | None:
        """
        Get the typical fix for an error type.

        Returns the delta_args that typically fixes this error,
        based on observed fix relationships.
        """
        pattern = self.get_pattern(tool_name)
        if not pattern:
            return None

        for ep in pattern.error_patterns:
            if ep.error_type == error_type and ep.fix_delta:
                return ep.fix_delta

        return None

    def get_successful_args_for_goal(self, tool_name: str, goal: str) -> dict[str, Any] | None:
        """
        Get argument hints for a goal based on past successes.

        Searches success patterns for matching goals and returns
        argument hints.
        """
        pattern = self.get_pattern(tool_name)
        if not pattern:
            return None

        goal_lower = goal.lower()
        for sp in pattern.success_patterns:
            if sp.goal_match and goal_lower in sp.goal_match.lower():
                return sp.arg_hints

        return None

    # --- Statistics ---

    def get_stats(self) -> ToolMemoryStats:
        """Get memory statistics."""
        total_calls = len(self.memory.tool_log)
        total_fixes = len(self.memory.fix_relations)
        tools_tracked = len(self.memory.tool_patterns)

        success_count = sum(1 for e in self.memory.tool_log if e.outcome == ToolOutcome.SUCCESS)
        failure_count = sum(1 for e in self.memory.tool_log if e.is_failure())

        return ToolMemoryStats(
            session_id=self.session_id,
            total_calls=total_calls,
            success_count=success_count,
            failure_count=failure_count,
            success_rate=success_count / total_calls if total_calls > 0 else 0,
            total_fixes_detected=total_fixes,
            tools_tracked=tools_tracked,
            created_at=self.memory.created_at.isoformat(),
            updated_at=self.memory.updated_at.isoformat(),
        )

    # --- Persistence hooks ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return self.memory.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> ToolMemoryManager:
        """Restore from dictionary."""
        memory = ProceduralMemory.model_validate(data)
        return cls(memory=memory, **kwargs)

    def reset(self) -> None:
        """Clear all memory."""
        self.memory.tool_log.clear()
        self.memory.tool_patterns.clear()
        self.memory.fix_relations.clear()
        self.memory.next_call_id = 1
        self.memory.updated_at = datetime.now(UTC)
