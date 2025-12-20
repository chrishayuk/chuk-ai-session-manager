# chuk_ai_session_manager/memory/__init__.py
"""
AI Virtual Memory subsystem.

This module implements OS-style virtual memory semantics for AI context management:
- Pages: Atomic units of content (text, images, audio, video)
- Working set: Currently mapped pages in context window
- Page faults: Loading content from lower tiers on demand
- Eviction: Moving pages to lower tiers under pressure
- Compression: Multi-resolution representations per page
"""

from .models import (
    Affinity,
    CompressionLevel,
    MemoryPage,
    Modality,
    PageTableEntry,
    StorageTier,
    TokenBudget,
    VMMetrics,
)
from .page_table import PageTable, PageTableStats
from .tlb import PageTLB, TLBWithPageTable
from .vm_prompts import (
    VM_STRICT_PROMPT,
    VM_RELAXED_PROMPT,
    VM_PASSIVE_PROMPT,
    VM_TOOLS,
    build_vm_developer_message,
    get_vm_tools,
)
from .working_set import WorkingSetConfig, WorkingSetManager
from .context_packer import ContextPacker, ContextPackerConfig, PackedContext
from .manifest import (
    AvailablePageEntry,
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
    VMToolResult,
)
from .artifacts_bridge import ArtifactsBridge, InMemoryBackend

__all__ = [
    # Enums
    "Affinity",
    "CompressionLevel",
    "Modality",
    "StorageTier",
    # Core Models
    "MemoryPage",
    "PageTableEntry",
    "TokenBudget",
    "VMMetrics",
    # Data Structures
    "PageTable",
    "PageTableStats",
    "PageTLB",
    "TLBWithPageTable",
    # Working Set
    "WorkingSetConfig",
    "WorkingSetManager",
    # Context Packing
    "ContextPacker",
    "ContextPackerConfig",
    "PackedContext",
    # Manifest
    "AvailablePageEntry",
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
    "VMToolResult",
    # Storage
    "ArtifactsBridge",
    "InMemoryBackend",
    # Prompts
    "VM_STRICT_PROMPT",
    "VM_RELAXED_PROMPT",
    "VM_PASSIVE_PROMPT",
    # Tools
    "VM_TOOLS",
    # Builders
    "build_vm_developer_message",
    "get_vm_tools",
]
