# chuk_ai_session_manager/memory/models/__init__.py
"""
Core models for the AI Virtual Memory system.

All public names are re-exported here so existing imports
(``from chuk_ai_session_manager.memory.models import MemoryPage``)
continue to work unchanged.
"""

# --- enums & constants -------------------------------------------------------
# --- ABI & UX metrics ---------------------------------------------------------
from chuk_ai_session_manager.memory.models.abi import (  # noqa: F401
    MemoryABI,
    PageManifestEntry,
    RecallAttempt,
    UserExperienceMetrics,
)

# --- content & tool definitions -----------------------------------------------
from chuk_ai_session_manager.memory.models.content import (  # noqa: F401
    AudioContent,
    FaultEffects,
    FormattedPage,
    ImageContent,
    PageContent,
    PageData,
    PageMeta,
    SearchResultEntry,
    StructuredContent,
    TextContent,
    ToolDefinition,
    ToolFunction,
    ToolParameter,
    ToolParameters,
    VideoContent,
)
from chuk_ai_session_manager.memory.models.enums import (  # noqa: F401
    ALL_COMPRESSION_LEVELS,
    MEMORY_PAGE_MIME_TYPE,
    VM_CHECKPOINT_MIME_TYPE,
    Actor,
    Affinity,
    CompressionLevel,
    ContextPrefix,
    FaultConfidenceThreshold,
    FaultReason,
    MessageRole,
    Modality,
    MutationType,
    PageType,
    StorageTier,
    ToolType,
    VMMode,
)

# --- fault policy & mutation log ----------------------------------------------
from chuk_ai_session_manager.memory.models.fault import (  # noqa: F401
    FaultPolicy,
    FaultRecord,
    PageMutation,
)

# --- page models --------------------------------------------------------------
from chuk_ai_session_manager.memory.models.page import (  # noqa: F401
    MemoryPage,
    PageTableEntry,
)

# --- stats & budgets ----------------------------------------------------------
from chuk_ai_session_manager.memory.models.stats import (  # noqa: F401
    CombinedPageTableStats,
    FaultMetrics,
    PageTableStats,
    StorageStats,
    TLBStats,
    TokenBudget,
    VMMetrics,
    WorkingSetStats,
)

__all__ = [
    # enums
    "CompressionLevel",
    "Modality",
    "StorageTier",
    "Affinity",
    "VMMode",
    "MessageRole",
    "PageType",
    "FaultReason",
    "MutationType",
    "Actor",
    "ContextPrefix",
    "ToolType",
    "FaultConfidenceThreshold",
    # constants
    "MEMORY_PAGE_MIME_TYPE",
    "VM_CHECKPOINT_MIME_TYPE",
    "ALL_COMPRESSION_LEVELS",
    # content
    "TextContent",
    "ImageContent",
    "AudioContent",
    "VideoContent",
    "StructuredContent",
    "PageContent",
    "PageMeta",
    "PageData",
    "FaultEffects",
    "SearchResultEntry",
    "FormattedPage",
    # tool definitions
    "ToolParameter",
    "ToolParameters",
    "ToolFunction",
    "ToolDefinition",
    # stats
    "TLBStats",
    "WorkingSetStats",
    "StorageStats",
    "CombinedPageTableStats",
    "PageTableStats",
    "FaultMetrics",
    "TokenBudget",
    "VMMetrics",
    # fault
    "FaultPolicy",
    "FaultRecord",
    "PageMutation",
    # page
    "MemoryPage",
    "PageTableEntry",
    # abi
    "PageManifestEntry",
    "MemoryABI",
    "RecallAttempt",
    "UserExperienceMetrics",
]
