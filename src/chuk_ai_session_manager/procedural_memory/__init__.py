# chuk_ai_session_manager/procedural_memory/__init__.py
"""
Procedural Memory System for Tool Usage Learning.

This module provides:
- Tool invocation tracing (what was called, when, with what args)
- Outcome tracking (success/failure + error types)
- Fix detection (linking failures to subsequent successful fixes)
- Pattern aggregation (learned "recipes" that work)
- Session persistence (via chuk-ai-session-manager)

Memory Hierarchy:
- L1: Hot cache (in ConversationProcessor) - not this module
- L2: ToolLog (session tool traces) - this module
- L3: ToolPatterns (aggregated knowledge) - this module

Usage:
    from chuk_ai_session_manager.procedural_memory import (
        ToolMemoryManager,
        ToolOutcome,
    )

    # Create manager (optionally bound to a session)
    manager = ToolMemoryManager.create(session_id="abc")

    # Record a tool call
    await manager.record_call(
        tool_name="solver_solve_plan",
        arguments={"tasks": 10},
        result={"status": "sat"},
        outcome=ToolOutcome.SUCCESS,
        context_goal="schedule F1 testing"
    )

    # Get context for paging into model
    from chuk_ai_session_manager.procedural_memory import ProceduralContextFormatter
    formatter = ProceduralContextFormatter()
    context = formatter.format_for_tools(manager, ["solver_solve_plan"])
"""

from chuk_ai_session_manager.procedural_memory.models import (
    ToolOutcome,
    ToolLogEntry,
    ToolPattern,
    ToolFixRelation,
    ProceduralMemory,
    ErrorPattern,
    SuccessPattern,
)
from chuk_ai_session_manager.procedural_memory.manager import ToolMemoryManager
from chuk_ai_session_manager.procedural_memory.formatter import (
    ProceduralContextFormatter,
    FormatterConfig,
)

__all__ = [
    # Models
    "ToolOutcome",
    "ToolLogEntry",
    "ToolPattern",
    "ToolFixRelation",
    "ProceduralMemory",
    "ErrorPattern",
    "SuccessPattern",
    # Manager
    "ToolMemoryManager",
    # Formatter
    "ProceduralContextFormatter",
    "FormatterConfig",
]
