# chuk_ai_session_manager/memory/__init__.py
"""
AI Virtual Memory subsystem.

This module implements OS-style virtual memory semantics for AI context management:
- Pages: Atomic units of content (text, images, audio, video)
- Working set: Currently mapped pages in context window
- Page faults: Loading content from lower tiers on demand
- Eviction: Moving pages to lower tiers under pressure
- Compression: Multi-resolution representations per page

Design principles:
- Async-native: All I/O operations are async
- Pydantic-native: All models are BaseModel subclasses
- No magic strings: Uses enums and constants throughout
"""

from .models import (
    # Enums
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
    # Constants
    ALL_COMPRESSION_LEVELS,
    MEMORY_PAGE_MIME_TYPE,
    VM_CHECKPOINT_MIME_TYPE,
    # Stats Models
    CombinedPageTableStats,
    FaultMetrics,
    PageTableStats,
    StorageStats,
    TLBStats,
    WorkingSetStats,
    # Content Models
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
    VideoContent,
    # Tool Definition Models
    ToolDefinition,
    ToolFunction,
    ToolParameter,
    ToolParameters,
    # Core Models
    MemoryPage,
    PageTableEntry,
    TokenBudget,
    VMMetrics,
    # Fault Policy Models
    FaultPolicy,
    FaultRecord,
    # Mutation Log Models
    PageMutation,
    # Memory ABI Models
    MemoryABI,
    PageManifestEntry,
    # UX Metrics Models
    RecallAttempt,
    UserExperienceMetrics,
)
from .page_table import PageTable
from .tlb import PageTLB, TLBWithPageTable
from .working_set import (
    AntiThrashPolicy,
    PinnedSet,
    WorkingSetConfig,
    WorkingSetManager,
)
from .pack_cache import ContextPackCache, PackedContext as CachedPackedContext
from .mutation_log import ContextSnapshot, MutationLogLite
from .prefetcher import SimplePrefetcher, ToolUsagePattern
from .context_packer import ContextPacker, ContextPackerConfig, PackedContext
from .manifest import (
    AvailablePageEntry,
    HintType,
    ManifestBuilder,
    ManifestPolicies,
    VMManifest,
    WorkingSetEntry,
    generate_simple_hint,
)
from .fault_handler import (
    FaultResult,
    PageFaultHandler,
    PageSearchHandler,
    SearchResult,
    VMToolError,
    VMToolResult,
)
from .artifacts_bridge import (
    ArtifactsBridge,
    CheckpointEntry,
    CheckpointManifest,
    CheckpointMetadata,
    InMemoryBackend,
    PageMetadata,
)
from .eviction_policy import (
    EvictionCandidate,
    EvictionContext,
    EvictionPolicy,
    ImportanceWeightedLRU,
    ImportanceWeightedLRUConfig,
    LRUEvictionPolicy,
    ModalityAwareLRU,
    ModalityAwareLRUConfig,
)
from .compressor import (
    CompressionResult,
    Compressor,
    CompressorRegistry,
    ImageCompressor,
    PassthroughCompressor,
    TextCompressor,
    TextCompressorConfig,
)
from .manager import MemoryManager, event_to_page
from .demand_paging import DEFAULT_RECALL_SIGNALS, DemandPagingPrePass
from .vm_prompts import (
    PAGE_FAULT_TOOL,
    SEARCH_PAGES_TOOL,
    VM_PASSIVE_PROMPT,
    VM_PROMPTS,
    VM_RELAXED_PROMPT,
    VM_STRICT_PROMPT,
    VM_TOOL_DEFINITIONS,
    VM_TOOLS,
    build_vm_developer_message,
    get_prompt_for_mode,
    get_vm_tools,
    get_vm_tools_as_dicts,
)

__all__ = [
    # Enums
    "Actor",
    "Affinity",
    "CompressionLevel",
    "ContextPrefix",
    "FaultConfidenceThreshold",
    "FaultReason",
    "MessageRole",
    "Modality",
    "MutationType",
    "PageType",
    "StorageTier",
    "ToolType",
    "VMMode",
    # Constants
    "ALL_COMPRESSION_LEVELS",
    "MEMORY_PAGE_MIME_TYPE",
    "VM_CHECKPOINT_MIME_TYPE",
    # Stats Models
    "CombinedPageTableStats",
    "FaultMetrics",
    "PageTableStats",
    "StorageStats",
    "TLBStats",
    "WorkingSetStats",
    # Content Models
    "AudioContent",
    "FaultEffects",
    "FormattedPage",
    "ImageContent",
    "PageContent",
    "PageData",
    "PageMeta",
    "SearchResultEntry",
    "StructuredContent",
    "TextContent",
    "VideoContent",
    # Tool Definition Models
    "ToolDefinition",
    "ToolFunction",
    "ToolParameter",
    "ToolParameters",
    # Core Models
    "MemoryPage",
    "PageTableEntry",
    "TokenBudget",
    "VMMetrics",
    # Fault Policy Models
    "FaultPolicy",
    "FaultRecord",
    # Mutation Log Models
    "PageMutation",
    # Memory ABI Models
    "MemoryABI",
    "PageManifestEntry",
    # UX Metrics Models
    "RecallAttempt",
    "UserExperienceMetrics",
    # Data Structures
    "PageTable",
    "PageTLB",
    "TLBWithPageTable",
    # Working Set
    "AntiThrashPolicy",
    "PinnedSet",
    "WorkingSetConfig",
    "WorkingSetManager",
    # Context Pack Cache
    "CachedPackedContext",
    "ContextPackCache",
    # Mutation Log
    "ContextSnapshot",
    "MutationLogLite",
    # Prefetcher
    "SimplePrefetcher",
    "ToolUsagePattern",
    # Context Packing
    "ContextPacker",
    "ContextPackerConfig",
    "PackedContext",
    # Manifest
    "AvailablePageEntry",
    "HintType",
    "ManifestBuilder",
    "ManifestPolicies",
    "VMManifest",
    "WorkingSetEntry",
    "generate_simple_hint",
    # Fault Handling
    "FaultResult",
    "PageFaultHandler",
    "PageSearchHandler",
    "SearchResult",
    "VMToolError",
    "VMToolResult",
    # Storage
    "ArtifactsBridge",
    "CheckpointEntry",
    "CheckpointManifest",
    "CheckpointMetadata",
    "InMemoryBackend",
    "PageMetadata",
    # Prompts
    "PAGE_FAULT_TOOL",
    "SEARCH_PAGES_TOOL",
    "VM_PASSIVE_PROMPT",
    "VM_PROMPTS",
    "VM_RELAXED_PROMPT",
    "VM_STRICT_PROMPT",
    "VM_TOOL_DEFINITIONS",
    "VM_TOOLS",
    # Eviction Policy
    "EvictionCandidate",
    "EvictionContext",
    "EvictionPolicy",
    "ImportanceWeightedLRU",
    "ImportanceWeightedLRUConfig",
    "LRUEvictionPolicy",
    "ModalityAwareLRU",
    "ModalityAwareLRUConfig",
    # Compressor
    "CompressionResult",
    "Compressor",
    "CompressorRegistry",
    "ImageCompressor",
    "PassthroughCompressor",
    "TextCompressor",
    "TextCompressorConfig",
    # Manager
    "MemoryManager",
    "event_to_page",
    # Builders
    "build_vm_developer_message",
    "get_prompt_for_mode",
    "get_vm_tools",
    "get_vm_tools_as_dicts",
    # Demand Paging
    "DEFAULT_RECALL_SIGNALS",
    "DemandPagingPrePass",
]
