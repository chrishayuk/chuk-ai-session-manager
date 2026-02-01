# chuk_ai_session_manager/guards/__init__.py
"""Conversation guards and state management.

Components:
- ToolStateManager: Coordinator for guards, bindings, and cache
- BindingManager: $vN reference system for tool result tracking
- ResultCache: Tool result caching for deduplication
- UngroundedGuard: Detects missing $vN references
- Models: All Pydantic models for state management

Runtime guards (BudgetGuard, RunawayGuard, etc.) are imported from chuk-tool-processor.
"""

# Models
from chuk_ai_session_manager.guards.models import (
    CachedToolResult,
    CacheScope,
    EnforcementLevel,
    NamedVariable,
    PerToolCallStatus,
    ReferenceCheckResult,
    RepairAction,
    RunawayStatus,
    RuntimeLimits,
    RuntimeMode,
    SoftBlock,
    SoftBlockReason,
    ToolClassification,
    UngroundedCallResult,
    UnusedResultAction,
    ValueBinding,
    ValueType,
    classify_value_type,
    compute_args_hash,
)

# Sub-managers
from chuk_ai_session_manager.guards.bindings import BindingManager
from chuk_ai_session_manager.guards.cache import ResultCache

# Chat-specific guard
from chuk_ai_session_manager.guards.ungrounded import (
    UngroundedGuard,
    UngroundedGuardConfig,
)

# Coordinator
from chuk_ai_session_manager.guards.manager import (
    ToolStateManager,
    get_tool_state,
    reset_tool_state,
)

# Re-export runtime guards from chuk-tool-processor
from chuk_tool_processor.guards import (
    BaseGuard,
    BudgetGuard,
    BudgetGuardConfig,
    BudgetState,
    Guard,
    GuardResult,
    GuardVerdict,
    PerToolGuard,
    PerToolGuardConfig,
    PreconditionGuard,
    PreconditionGuardConfig,
    RunawayGuard,
    RunawayGuardConfig,
)

__all__ = [
    # Manager
    "ToolStateManager",
    "get_tool_state",
    "reset_tool_state",
    # Sub-managers
    "BindingManager",
    "ResultCache",
    # Guards (from chuk-tool-processor)
    "BaseGuard",
    "Guard",
    "GuardResult",
    "GuardVerdict",
    "BudgetGuard",
    "BudgetGuardConfig",
    "BudgetState",
    "PerToolGuard",
    "PerToolGuardConfig",
    "PreconditionGuard",
    "PreconditionGuardConfig",
    "RunawayGuard",
    "RunawayGuardConfig",
    # Chat-specific guard
    "UngroundedGuard",
    "UngroundedGuardConfig",
    # Enums and Constants
    "CacheScope",
    "EnforcementLevel",
    "RuntimeMode",
    "UnusedResultAction",
    "ValueType",
    "ToolClassification",
    # Models
    "CachedToolResult",
    "NamedVariable",
    "PerToolCallStatus",
    "ReferenceCheckResult",
    "RepairAction",
    "RunawayStatus",
    "RuntimeLimits",
    "SoftBlock",
    "SoftBlockReason",
    "UngroundedCallResult",
    "ValueBinding",
    # Helpers
    "classify_value_type",
    "compute_args_hash",
]
