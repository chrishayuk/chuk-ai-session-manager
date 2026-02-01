# chuk_ai_session_manager/procedural_memory/formatter.py
"""
Context Formatter for Procedural Memory.

Handles formatting procedural memory for injection into model context.
This is the "paging" layer - deciding what tool knowledge to include
and how to present it.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from chuk_ai_session_manager.procedural_memory.models import (
    ToolLogEntry,
    ToolPattern,
)
from chuk_ai_session_manager.procedural_memory.manager import ToolMemoryManager


class FormatterConfig(BaseModel):
    """Configuration for procedural memory formatting."""

    # How many recent calls to include per tool
    max_recent_calls: int = 3

    # How many error patterns to show
    max_error_patterns: int = 3

    # How many success patterns to show
    max_success_patterns: int = 3

    # Whether to include argument details
    include_args: bool = False

    # Whether to include timing info
    include_timing: bool = False

    # Whether to show fix relationships
    show_fix_relations: bool = True

    # Compact mode (less verbose)
    compact: bool = False


class ProceduralContextFormatter(BaseModel):
    """
    Formats procedural memory for injection into model context.

    This class handles the "paging" decision - what to include and how.
    It can format:
    - Recent tool history for specific tools
    - Patterns (errors, successes) for tools
    - Full procedural context summary
    """

    config: FormatterConfig = Field(default_factory=FormatterConfig)

    def format_for_tools(
        self,
        manager: ToolMemoryManager,
        tool_names: list[str],
        context_goal: Optional[str] = None,
    ) -> str:
        """
        Format procedural memory for specific tools about to be called.

        This is the main entry point for "just-in-time" paging.

        Args:
            manager: The tool memory manager
            tool_names: Tools that are about to be called
            context_goal: Current goal (for relevance filtering)

        Returns:
            Formatted context string ready for injection
        """
        sections = []

        for tool_name in tool_names:
            tool_section = self._format_tool_section(manager, tool_name, context_goal)
            if tool_section:
                sections.append(tool_section)

        if not sections:
            return ""

        header = "<procedural_memory>\n"
        footer = "\n</procedural_memory>"

        return header + "\n\n".join(sections) + footer

    def _format_tool_section(
        self,
        manager: ToolMemoryManager,
        tool_name: str,
        context_goal: Optional[str] = None,
    ) -> Optional[str]:
        """Format a single tool's procedural memory."""
        lines = []

        # Get recent calls for this tool
        recent = manager.get_recent_calls(
            tool_name=tool_name,
            limit=self.config.max_recent_calls,
        )

        # Get pattern
        pattern = manager.get_pattern(tool_name)

        # Skip if no history
        if not recent and not pattern:
            return None

        lines.append(f'<tool_memory name="{tool_name}">')

        # Recent history
        if recent:
            lines.append(self._format_recent_calls(recent))

        # Patterns
        if pattern:
            pattern_text = self._format_pattern(pattern, context_goal)
            if pattern_text:
                if recent:
                    lines.append("")
                lines.append(pattern_text)

        lines.append("</tool_memory>")

        return "\n".join(lines)

    def _format_recent_calls(self, calls: list[ToolLogEntry]) -> str:
        """Format recent call history."""
        lines = ["<recent_calls>"]

        for entry in calls:
            status = "✓" if entry.is_success() else "✗"
            line = f"  [{status}] {entry.result_summary}"

            # Add fix info
            if self.config.show_fix_relations:
                if entry.is_fix() and entry.delta_args:
                    line += f" (fixed prior by: {self._format_delta(entry.delta_args)})"
                elif entry.is_failure() and entry.was_fixed():
                    line += f" (fixed by {entry.fixed_by})"

            # Add args if configured
            if self.config.include_args and entry.arguments:
                args_str = self._format_args(entry.arguments)
                line += f"\n    args: {args_str}"

            # Add timing if configured
            if self.config.include_timing and entry.execution_time_ms:
                line += f" [{entry.execution_time_ms}ms]"

            lines.append(line)

        lines.append("</recent_calls>")
        return "\n".join(lines)

    def _format_pattern(
        self,
        pattern: ToolPattern,
        context_goal: Optional[str] = None,
    ) -> Optional[str]:
        """Format aggregated patterns."""
        lines = []

        # Only show patterns if we have useful info
        has_errors = bool(pattern.error_patterns)
        has_successes = bool(pattern.success_patterns)

        if not has_errors and not has_successes:
            return None

        lines.append("<patterns>")

        # Stats (if not compact)
        if not self.config.compact:
            lines.append(
                f"  success_rate: {pattern.success_rate:.0%} "
                f"({pattern.total_calls} calls)"
            )

        # Error patterns with fixes
        if has_errors:
            lines.append("  <common_errors>")
            for ep in pattern.error_patterns[-self.config.max_error_patterns :]:
                error_line = f"    - {ep.error_type}"
                if ep.count > 1:
                    error_line += f" (x{ep.count})"
                if ep.typical_fix:
                    error_line += f" -> fix: {ep.typical_fix}"
                lines.append(error_line)
            lines.append("  </common_errors>")

        # Success patterns (especially fixes)
        if has_successes:
            relevant = self._filter_relevant_successes(
                pattern.success_patterns,
                context_goal,
            )
            if relevant:
                lines.append("  <success_hints>")
                for sp in relevant[-self.config.max_success_patterns :]:
                    if sp.delta_that_fixed:
                        lines.append(
                            f"    - Fixed by: {self._format_delta(sp.delta_that_fixed)}"
                        )
                    elif sp.arg_hints:
                        lines.append(f"    - Typical args: {sp.arg_hints}")
                lines.append("  </success_hints>")

        lines.append("</patterns>")
        return "\n".join(lines)

    def _filter_relevant_successes(
        self,
        patterns: list,
        context_goal: Optional[str],
    ) -> list:
        """Filter success patterns to those relevant to current goal."""
        if not context_goal:
            return patterns

        goal_lower = context_goal.lower()
        relevant = []

        for sp in patterns:
            # Include if no goal specified (general pattern)
            if not sp.goal_match:
                relevant.append(sp)
                continue

            # Include if goal matches
            if goal_lower in sp.goal_match.lower():
                relevant.append(sp)
                continue

            # Include if it's a fix (always useful)
            if sp.delta_that_fixed:
                relevant.append(sp)

        return relevant

    def _format_delta(self, delta: dict[str, Any]) -> str:
        """Format arg delta compactly."""
        parts = []

        if "added" in delta:
            added_keys = list(delta["added"].keys())
            if len(added_keys) <= 2:
                parts.append(f"+{', '.join(added_keys)}")
            else:
                parts.append(f"+{len(added_keys)} args")

        if "removed" in delta:
            removed = delta["removed"]
            if len(removed) <= 2:
                parts.append(f"-{', '.join(removed)}")
            else:
                parts.append(f"-{len(removed)} args")

        if "changed" in delta:
            changed_keys = list(delta["changed"].keys())
            if len(changed_keys) <= 2:
                changes = []
                for k in changed_keys:
                    v = delta["changed"][k]
                    changes.append(
                        f"{k}:{self._abbrev(v['from'])}->{self._abbrev(v['to'])}"
                    )
                parts.append(", ".join(changes))
            else:
                parts.append(f"~{len(changed_keys)} args")

        return "; ".join(parts) if parts else "changes"

    def _format_args(self, args: dict[str, Any], max_items: int = 3) -> str:
        """Format arguments compactly."""
        items = list(args.items())[:max_items]
        formatted = [f"{k}={self._abbrev(v)}" for k, v in items]
        if len(args) > max_items:
            formatted.append(f"...+{len(args) - max_items}")
        return ", ".join(formatted)

    def _abbrev(self, value: Any, max_len: int = 15) -> str:
        """Abbreviate a value."""
        s = str(value)
        if len(s) > max_len:
            return s[: max_len - 2] + ".."
        return s

    # --- Full context formatting ---

    def format_full_summary(
        self,
        manager: ToolMemoryManager,
        max_tools: int = 5,
    ) -> str:
        """
        Format a full summary of procedural memory.

        Useful for periodic context refresh or debugging.
        """
        lines = ["<procedural_memory_summary>"]

        stats = manager.get_stats()
        lines.append(
            f"  Total: {stats['total_calls']} calls, "
            f"{stats['success_rate']:.0%} success, "
            f"{stats['total_fixes_detected']} fixes detected"
        )

        # Top tools by usage
        patterns = manager.get_all_patterns()
        sorted_tools = sorted(
            patterns.items(),
            key=lambda x: x[1].total_calls,
            reverse=True,
        )[:max_tools]

        if sorted_tools:
            lines.append("")
            lines.append("  Most used tools:")
            for tool_name, pattern in sorted_tools:
                lines.append(
                    f"    - {tool_name}: {pattern.total_calls} calls, "
                    f"{pattern.success_rate:.0%} success"
                )
                if pattern.error_patterns:
                    top_error = pattern.error_patterns[-1]
                    lines.append(f"      last error: {top_error.error_type}")

        lines.append("</procedural_memory_summary>")
        return "\n".join(lines)

    # --- Specialized formats ---

    def format_error_guidance(
        self,
        manager: ToolMemoryManager,
        tool_name: str,
        error_type: str,
    ) -> str:
        """
        Format guidance for handling a specific error.

        Called when a tool fails - provides context on how
        this error was handled before.
        """
        lines = [f'<error_guidance tool="{tool_name}" error="{error_type}">']

        # Look for fix pattern
        fix_delta = manager.get_fix_for_error(tool_name, error_type)
        if fix_delta:
            lines.append(f"  Previous fix: {self._format_delta(fix_delta)}")

        # Look for similar past failures that were fixed
        similar = manager.search_calls(
            tool_name=tool_name,
            error_type=error_type,
            only_fixed=True,
            limit=3,
        )

        if similar:
            lines.append("  Past fixes:")
            for entry in similar:
                if entry.fixed_by:
                    lines.append(f"    - {entry.id} -> {entry.fixed_by}")
                    if entry.delta_args:
                        lines.append(
                            f"      changed: {self._format_delta(entry.delta_args)}"
                        )

        if len(lines) == 1:
            lines.append("  No previous fixes found for this error type.")

        lines.append("</error_guidance>")
        return "\n".join(lines)

    def format_success_template(
        self,
        manager: ToolMemoryManager,
        tool_name: str,
        goal: str,
    ) -> Optional[str]:
        """
        Format a success template based on past successful calls.

        Returns argument hints for achieving a goal.
        """
        arg_hints = manager.get_successful_args_for_goal(tool_name, goal)
        if not arg_hints:
            return None

        lines = [f'<success_template tool="{tool_name}" goal="{goal}">']
        lines.append("  Recommended args based on past success:")
        for key, value in arg_hints.items():
            lines.append(f"    {key}: {value}")
        lines.append("</success_template>")

        return "\n".join(lines)
