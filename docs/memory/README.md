# AI Virtual Memory System

> Treating AI context like operating system memory: pages, working sets, faults, and swapping.

The AI Virtual Memory subsystem provides OS-style memory semantics for managing AI context windows. Just as operating systems virtualize physical memory to give processes the illusion of infinite address space, this system virtualizes context windows to give AI conversations the illusion of infinite memory.

## Quick Start

```python
from chuk_ai_session_manager.memory import (
    # Core models
    MemoryPage, Modality, StorageTier, CompressionLevel,
    # Data structures
    PageTable, PageTLB, TLBWithPageTable,
    WorkingSetManager, WorkingSetConfig,
    # Context building
    ContextPacker, ManifestBuilder,
    # Fault handling
    PageFaultHandler, PageSearchHandler,
    # Storage
    ArtifactsBridge,
    # Prompts & tools
    VMMode, get_vm_tools, build_vm_developer_message,
)

# Create a memory page
page = MemoryPage(
    page_id="msg_001",
    modality=Modality.TEXT,
    content="What's the weather like?",
    storage_tier=StorageTier.L0,
)

# Register in page table
table = PageTable()
entry = table.register(page)

# Add to working set
working_set = WorkingSetManager()
working_set.add_to_l0(page)

# Pack context for model
packer = ContextPacker()
packed = packer.pack([page])
print(packed.content)  # VM:CONTEXT format
```

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                          │
│         AI Agent / Chat / Tool Orchestration                   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                   VIRTUAL MEMORY LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Page Table   │  │ Working Set  │  │ Fault Handler        │ │
│  │              │  │ Manager      │  │                      │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ TLB Cache    │  │ Context      │  │ Manifest             │ │
│  │              │  │ Packer       │  │ Builder              │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    STORAGE HIERARCHY                           │
│                                                                │
│   L0 (Context)   │  Current prompt context window             │
│   L1 (Cache)     │  Recent pages, hot data (session state)    │
│   L2 (Session)   │  Session storage (chuk-sessions)           │
│   L3 (Disk)      │  Artifact storage (chuk-artifacts)         │
│   L4 (Cold)      │  Archive/S3 (chuk-artifacts vfs-s3)        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Design Principles

1. **Pydantic-native**: All models are `BaseModel` subclasses with proper validation
2. **No magic strings**: Uses enums (`Modality`, `StorageTier`, `CompressionLevel`) throughout
3. **Type-safe**: Full type annotations, returns typed models instead of dicts
4. **Async-ready**: Core operations support async patterns

## Core Concepts

### Storage Tiers

| Tier | Name | Latency | Capacity | Content |
|------|------|---------|----------|---------|
| L0 | Context | 0ms | ~128K tokens | Active prompt window |
| L1 | Cache | <10ms | ~1M tokens | Recent messages, hot pages |
| L2 | Session | <100ms | ~100M tokens | Full session events |
| L3 | Disk | <1s | ~10GB | Artifacts, media, checkpoints |
| L4 | Cold | 1-10s | Unlimited | Archives, old sessions |

### Compression Levels

| Level | Text | Image | Audio | Video |
|-------|------|-------|-------|-------|
| 0 (Full) | Complete messages | Full resolution | Full waveform | All frames |
| 1 (Reduced) | Recent + summaries | Thumbnail | Transcript | Keyframes |
| 2 (Abstract) | Key points only | Caption | Summary | Scene descriptions |
| 3 (Reference) | Topic tags | "image:{id}" | Duration + topic | "video:{id}" |

### Page Types

- **Text**: User messages, assistant responses, summaries
- **Image**: Photos, diagrams, screenshots
- **Audio**: Voice recordings, podcasts
- **Video**: Video content with keyframes
- **Structured**: JSON/tool outputs

## Module Reference

| Module | Purpose |
|--------|---------|
| `models.py` | Core models, enums, and type definitions |
| `page_table.py` | Page metadata tracking and tier management |
| `tlb.py` | Fast cache for page table lookups |
| `working_set.py` | Token budget, L0/L1 capacity, pinning, anti-thrash |
| `context_packer.py` | Pack pages into VM:CONTEXT format |
| `manifest.py` | Generate VM:MANIFEST_JSON for models |
| `fault_handler.py` | Handle page_fault and search_pages tools |
| `artifacts_bridge.py` | Persistent storage integration |
| `vm_prompts.py` | Tool definitions and mode-specific prompts |
| `pack_cache.py` | Cache packed context to avoid re-packing (v0.8) |
| `mutation_log.py` | Append-only log for debugging/replay (v0.8) |
| `prefetcher.py` | Heuristic-based page prefetch (v0.8) |

## v0.8 Components

The v0.8 release adds several components to make the VM system robust and complete:

### Page Type Taxonomy

Not all pages are equal. Different types have different eviction/compression rules:

| Page Type | Description | Eviction Priority |
|-----------|-------------|-------------------|
| `transcript` | Raw turns, tool outputs | Normal |
| `summary` | LLM-generated summaries | Low (rebuildable) |
| `claim` | Decisions, facts, conclusions | Very Low (high-value) |
| `procedure` | Learned patterns | Low |

### Provenance Tracking

Pages can track their lineage via the `provenance` field:

```python
# Summary that derives from messages
summary = MemoryPage(
    page_id="summary_seg_01",
    page_type=PageType.SUMMARY,
    provenance=["msg_001", "msg_002", "msg_003"],  # Source messages
    content="Key points from the discussion...",
)
```

### Pinned Pages

Some pages should never be evicted:

```python
from chuk_ai_session_manager.memory import PinnedSet

pinned = PinnedSet(pin_last_n_turns=3)
pinned.pin("system_prompt")
pinned.pin("claim_auth_decision")
```

### Anti-Thrash Protection

Prevents evicting recently-faulted pages:

```python
from chuk_ai_session_manager.memory import AntiThrashPolicy

policy = AntiThrashPolicy(
    eviction_cooldown_turns=3,
    fault_protection_turns=2,
)
```

### Mutation Log

Debug and replay page operations:

```python
from chuk_ai_session_manager.memory import MutationLogLite

log = MutationLogLite()
# ... operations ...
context = log.get_context_at_turn(turn=5)
```

### UX Metrics

Track user experience metrics:

```python
from chuk_ai_session_manager.memory import UserExperienceMetrics

metrics = UserExperienceMetrics()
print(f"Recall success rate: {metrics.recall_success_rate():.2%}")
print(f"Thrash index: {metrics.thrash_index():.2f}")
```

## Documentation

- [Core Models](./models.md) - Enums, content types, and data models
- [Data Structures](./data-structures.md) - PageTable, TLB, WorkingSet
- [Context & Manifest](./context-manifest.md) - Building model input
- [Fault Handling](./fault-handling.md) - Page faults and search
- [Storage](./storage.md) - ArtifactsBridge and persistence
- [Integration Guide](./integration.md) - Using with SessionManager

## Why Not RAG?

| Capability | RAG | AI Virtual Memory |
|------------|-----|-------------------|
| Page identity & lifecycle | Chunks are anonymous | Pages have IDs, versions, lineage |
| Multi-resolution per page | Single embedding | Full → Summary → Caption → Reference |
| Working set semantics | Top-K retrieval | Explicit hot/warm/cold tracking |
| Deterministic eviction | Implicit | Policy-driven (LRU, importance-weighted) |
| Coherency guarantees | None | Dirty tracking, flush, checkpoint |
| Cross-modal causality | Separate indexes | Unified page model, cross-references |

**Core difference:** RAG optimizes *retrieval*. AI Virtual Memory *virtualizes memory*.

## Example: Full Integration

See [`examples/memory_example.py`](../../examples/memory_example.py) for a comprehensive demonstration of all components working together.

```python
import asyncio
from chuk_ai_session_manager.memory import (
    MemoryPage, Modality, StorageTier, VMMode,
    PageTable, TLBWithPageTable, WorkingSetManager,
    ContextPacker, ManifestBuilder, ManifestPolicies,
    PageFaultHandler, build_vm_developer_message, get_vm_tools,
)

async def build_context_for_model():
    # 1. Create pages from conversation
    pages = [
        MemoryPage(page_id="msg_001", modality=Modality.TEXT,
                   content="User question", storage_tier=StorageTier.L0),
        MemoryPage(page_id="msg_002", modality=Modality.TEXT,
                   content="Assistant response", storage_tier=StorageTier.L0),
    ]

    # 2. Register in page table with TLB
    table = PageTable()
    tlb = TLBWithPageTable(table)
    for page in pages:
        tlb.register(page)

    # 3. Track in working set
    working_set = WorkingSetManager()
    for page in pages:
        working_set.add_to_l0(page)

    # 4. Pack context
    packer = ContextPacker()
    packed = packer.pack(pages)

    # 5. Build manifest
    builder = ManifestBuilder(
        session_id="session_123",
        policies=ManifestPolicies(max_faults_per_turn=3),
    )
    for page in pages:
        builder.add_working_set_page(page)
    manifest = builder.build()

    # 6. Build developer message
    developer_msg = build_vm_developer_message(
        mode=VMMode.STRICT,
        manifest_json=manifest.model_dump_json(),
        context=packed.content,
    )

    # 7. Get tool definitions
    tools = get_vm_tools()

    return developer_msg, tools

asyncio.run(build_context_for_model())
```
