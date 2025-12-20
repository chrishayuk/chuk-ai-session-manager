# chuk_ai_session_manager/procedural_memory/models.py
"""
Data models for procedural memory.

These models represent:
- Individual tool invocations (traces)
- Aggregated tool patterns (recipes)
- Fix relationships (how failures were resolved)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolOutcome(str, Enum):
    """Outcome of a tool invocation."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # Completed but with warnings/limitations
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ToolFixRelation(BaseModel):
    """Describes how one call fixed a prior failure."""

    failed_call_id: str
    success_call_id: str
    delta_args: dict[str, Any] = Field(default_factory=dict)
    # delta_args structure:
    # {
    #   "added": {"key": value, ...},
    #   "removed": ["key1", "key2"],
    #   "changed": {"key": {"from": old, "to": new}}
    # }


class ToolLogEntry(BaseModel):
    """
    Single tool invocation record (the 'trace').

    This captures everything about a single tool call:
    - What was called and with what arguments
    - The context/goal at the time
    - The outcome and any errors
    - Links to fixes if this was a failure that got resolved
    """

    # Identity
    id: str  # call-001, call-002, ...
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Tool call details
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_hash: str = ""  # For quick comparison

    # Context at call time
    context_goal: Optional[str] = None  # What was user trying to do?
    preceding_call_id: Optional[str] = None  # Chain of calls in this turn

    # Outcome
    outcome: ToolOutcome
    result_summary: str  # Compact summary, not full result
    result_type: Optional[str] = None  # "sat", "list", "object", etc.
    execution_time_ms: Optional[int] = None

    # Error details (if failure)
    error_type: Optional[str] = None  # "unsat", "timeout", "validation", etc.
    error_message: Optional[str] = None

    # Fix tracking
    fixed_by: Optional[str] = None  # ID of call that fixed this failure
    fix_for: Optional[str] = None  # ID of call this fixed
    delta_args: Optional[dict[str, Any]] = None  # What changed to fix it

    model_config = {"frozen": False}

    def is_failure(self) -> bool:
        """Check if this was a failed call."""
        return self.outcome in (ToolOutcome.FAILURE, ToolOutcome.TIMEOUT)

    def is_success(self) -> bool:
        """Check if this was a successful call."""
        return self.outcome == ToolOutcome.SUCCESS

    def was_fixed(self) -> bool:
        """Check if this failure was subsequently fixed."""
        return self.fixed_by is not None

    def is_fix(self) -> bool:
        """Check if this call fixed a prior failure."""
        return self.fix_for is not None

    def format_compact(self) -> str:
        """Format as compact string for logging."""
        status = "✓" if self.is_success() else "✗"
        return f"[{status}] {self.tool_name}: {self.result_summary}"

    def format_for_context(self, include_args: bool = False) -> str:
        """Format for injection into model context."""
        status = "✓" if self.is_success() else "✗"
        lines = [f"[{status}] {self.result_summary}"]

        if include_args and self.arguments:
            # Show abbreviated args
            args_str = ", ".join(f"{k}={_abbrev(v)}" for k, v in self.arguments.items())
            lines.append(f"    args: {args_str}")

        if self.is_fix() and self.delta_args:
            lines.append(f"    (fixed prior failure by: {self.delta_args})")

        if self.is_failure() and self.was_fixed():
            lines.append(f"    (later fixed by {self.fixed_by})")

        return "\n".join(lines)


class ErrorPattern(BaseModel):
    """A pattern of errors for a tool."""

    error_type: str
    count: int = 1
    contexts: list[str] = Field(
        default_factory=list
    )  # Goal contexts where this occurred
    example_args: Optional[dict[str, Any]] = None
    typical_fix: Optional[str] = None  # Description of how it's usually fixed
    fix_delta: Optional[dict[str, Any]] = None  # Actual arg changes that fixed it


class SuccessPattern(BaseModel):
    """A pattern of successful usage for a tool."""

    goal_match: Optional[str] = None  # What kind of goal this pattern works for
    arg_hints: dict[str, Any] = Field(default_factory=dict)
    # arg_hints: {"must_include": [...], "typical_values": {...}}
    notes: Optional[str] = None
    example_call_id: Optional[str] = None
    delta_that_fixed: Optional[dict[str, Any]] = None  # If this was a fix


class ToolPattern(BaseModel):
    """
    Aggregated knowledge about a tool (the 'recipe').

    This accumulates across calls to learn:
    - What arguments typically work
    - What errors occur and how to fix them
    - Success patterns for different goals
    """

    tool_name: str

    # Statistics
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.success_count / self.total_calls

    # Argument statistics
    common_args: dict[str, list[Any]] = Field(default_factory=dict)
    # common_args: {"param_name": [most_common_values]}

    # Success patterns
    success_patterns: list[SuccessPattern] = Field(default_factory=list)

    # Error patterns
    error_patterns: list[ErrorPattern] = Field(default_factory=list)

    # Timing
    avg_execution_ms: Optional[float] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": False}

    def record_call(self, entry: ToolLogEntry) -> None:
        """Update pattern statistics from a call."""
        self.total_calls += 1

        if entry.is_success():
            self.success_count += 1
        elif entry.is_failure():
            self.failure_count += 1

        # Update timing
        if entry.execution_time_ms:
            if self.avg_execution_ms is None:
                self.avg_execution_ms = float(entry.execution_time_ms)
            else:
                # Running average
                self.avg_execution_ms = (
                    self.avg_execution_ms * (self.total_calls - 1)
                    + entry.execution_time_ms
                ) / self.total_calls

        # Track common args
        for key, value in entry.arguments.items():
            if key not in self.common_args:
                self.common_args[key] = []
            # Keep last few values (dedup)
            if value not in self.common_args[key]:
                self.common_args[key].append(value)
                if len(self.common_args[key]) > 5:
                    self.common_args[key].pop(0)

        self.updated_at = datetime.now(timezone.utc)

    def add_error_pattern(
        self,
        error_type: str,
        context: Optional[str] = None,
        example_args: Optional[dict] = None,
    ) -> ErrorPattern:
        """Add or update an error pattern."""
        # Find existing
        for pattern in self.error_patterns:
            if pattern.error_type == error_type:
                pattern.count += 1
                if context and context not in pattern.contexts:
                    pattern.contexts.append(context)
                    if len(pattern.contexts) > 5:
                        pattern.contexts.pop(0)
                return pattern

        # Create new
        pattern = ErrorPattern(
            error_type=error_type,
            contexts=[context] if context else [],
            example_args=example_args,
        )
        self.error_patterns.append(pattern)
        return pattern

    def add_success_pattern(
        self,
        goal_match: Optional[str] = None,
        arg_hints: Optional[dict] = None,
        delta_that_fixed: Optional[dict] = None,
        example_call_id: Optional[str] = None,
    ) -> SuccessPattern:
        """Add a success pattern."""
        pattern = SuccessPattern(
            goal_match=goal_match,
            arg_hints=arg_hints or {},
            delta_that_fixed=delta_that_fixed,
            example_call_id=example_call_id,
        )
        self.success_patterns.append(pattern)

        # Keep limited history
        if len(self.success_patterns) > 10:
            self.success_patterns.pop(0)

        return pattern

    def record_fix(self, error_type: str, fix_delta: dict[str, Any]) -> None:
        """Record that an error was fixed with specific arg changes."""
        for pattern in self.error_patterns:
            if pattern.error_type == error_type:
                pattern.fix_delta = fix_delta
                pattern.typical_fix = _describe_fix(fix_delta)
                return

    def format_for_context(self, max_errors: int = 3, max_successes: int = 3) -> str:
        """Format pattern for injection into model context."""
        lines = []

        # Stats summary
        lines.append(
            f"Success rate: {self.success_rate:.0%} ({self.total_calls} calls)"
        )

        # Error patterns
        if self.error_patterns:
            lines.append("Common errors:")
            for ep in self.error_patterns[-max_errors:]:
                fix_hint = f" -> {ep.typical_fix}" if ep.typical_fix else ""
                lines.append(f"  - {ep.error_type} (x{ep.count}){fix_hint}")

        # Success hints
        if self.success_patterns:
            lines.append("Success hints:")
            for sp in self.success_patterns[-max_successes:]:
                if sp.delta_that_fixed:
                    lines.append(f"  - Fixed by: {sp.delta_that_fixed}")
                elif sp.arg_hints:
                    lines.append(f"  - Args: {sp.arg_hints}")

        return "\n".join(lines)


class ProceduralMemory(BaseModel):
    """
    Container for all procedural memory state.

    This is the "L2/L3" memory that persists beyond the hot cache.
    """

    session_id: str

    # L2: Session tool log (append-only trace)
    tool_log: list[ToolLogEntry] = Field(default_factory=list)
    next_call_id: int = 1

    # L3: Per-tool patterns (aggregated knowledge)
    tool_patterns: dict[str, ToolPattern] = Field(default_factory=dict)

    # Fix relationships
    fix_relations: list[ToolFixRelation] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": False}

    def get_pattern(self, tool_name: str) -> ToolPattern:
        """Get or create pattern for a tool."""
        if tool_name not in self.tool_patterns:
            self.tool_patterns[tool_name] = ToolPattern(tool_name=tool_name)
        return self.tool_patterns[tool_name]

    def allocate_call_id(self) -> str:
        """Allocate next call ID."""
        call_id = f"call-{self.next_call_id:04d}"
        self.next_call_id += 1
        return call_id


# --- Helpers ---


def _abbrev(value: Any, max_len: int = 20) -> str:
    """Abbreviate a value for display."""
    s = str(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _describe_fix(delta: dict[str, Any]) -> str:
    """Generate human-readable description of a fix."""
    parts = []

    if "added" in delta:
        added_keys = list(delta["added"].keys())
        parts.append(f"add {', '.join(added_keys)}")

    if "removed" in delta:
        parts.append(f"remove {', '.join(delta['removed'])}")

    if "changed" in delta:
        changed = []
        for k, v in delta["changed"].items():
            changed.append(f"{k}: {_abbrev(v['from'])} -> {_abbrev(v['to'])}")
        parts.append(f"change {'; '.join(changed)}")

    return "; ".join(parts) if parts else "unknown changes"
