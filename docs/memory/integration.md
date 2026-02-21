# Integration Guide

This guide shows how to integrate the AI Virtual Memory system into your application.

## Overview

The memory system sits between your application and the LLM API, providing:

1. **Context management** - What goes into the prompt
2. **Page tracking** - Where content lives and its state
3. **Tool handling** - Processing page_fault and search_pages calls
4. **Persistence** - Storing pages beyond the current session

```
┌─────────────────────────────────────────────────────────────┐
│                     YOUR APPLICATION                        │
│                                                             │
│   User Input  ─────►  MemoryManager  ─────►  LLM API       │
│                            │                    │           │
│                            ▼                    ▼           │
│                    PageTable/TLB         Tool Calls        │
│                    WorkingSet            (page_fault)      │
│                    ContextPacker         (search_pages)    │
│                    ManifestBuilder                         │
│                            │                    │           │
│                            ▼                    ▼           │
│                    ArtifactsBridge      FaultHandler       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start: SessionManager with VM

The easiest way to use virtual memory is through `SessionManager`:

```python
from chuk_ai_session_manager import SessionManager
from chuk_ai_session_manager.memory import VMMode, WorkingSetConfig

# Zero-config: just enable VM
sm = SessionManager(enable_vm=True, vm_mode=VMMode.STRICT)

# Or with full customization
from chuk_ai_session_manager.memory import (
    ImportanceWeightedLRU, CompressorRegistry,
)

sm = SessionManager(
    enable_vm=True,
    vm_mode=VMMode.STRICT,
    vm_config=WorkingSetConfig(max_l0_tokens=32_000),
    vm_eviction_policy=ImportanceWeightedLRU(),
    vm_compressor_registry=CompressorRegistry.default(),
)

# Use normally — pages are created automatically
await sm.user_says("What's the best auth approach?")
await sm.ai_responds("I recommend JWT tokens...", model="gpt-4o")

# Build context with VM:CONTEXT included
messages = await sm.get_messages_for_llm(include_system=True)
```

The built-in `MemoryManager` (in `memory/manager.py`) orchestrates page table, working set,
fault handling, eviction, and compression automatically. `SessionManager` creates it when
`enable_vm=True` and maps conversation events to memory pages.

## Custom Integration

For full control, you can use the memory components directly.

### Basic Integration

### Step 1: Initialize Components

```python
from chuk_ai_session_manager.memory import (
    # Core
    PageTable, PageTLB, TLBWithPageTable,
    WorkingSetManager, WorkingSetConfig, TokenBudget,
    # Context
    ContextPacker, ManifestBuilder, ManifestPolicies,
    # Fault handling
    PageFaultHandler, PageSearchHandler,
    # Storage
    ArtifactsBridge, InMemoryBackend,
    # Models
    MemoryPage, Modality, StorageTier, VMMode,
    # Prompts
    build_vm_developer_message, get_vm_tools_as_dicts,
)

class MemoryManager:
    def __init__(self, session_id: str):
        self.session_id = session_id

        # Data structures
        self.page_table = PageTable()
        self.tlb = PageTLB(max_entries=512)
        self.combined = TLBWithPageTable(self.page_table, self.tlb)

        # Working set
        config = WorkingSetConfig(
            max_l0_tokens=128_000,
            max_l1_pages=100,
            reserved_tokens=4000,
        )
        budget = TokenBudget(total_limit=128_000, reserved=4000)
        self.working_set = WorkingSetManager(config=config, budget=budget)

        # Context building
        self.packer = ContextPacker()

        # Storage
        backend = InMemoryBackend()
        self.storage = ArtifactsBridge(backend=backend)

        # Page content cache
        self._pages: dict[str, MemoryPage] = {}

        # Fault handler
        self.fault_handler = PageFaultHandler(
            page_table=self.page_table,
            get_page=self._get_page,
            max_faults_per_turn=3,
        )

    async def _get_page(self, page_id: str) -> MemoryPage | None:
        # Check memory cache first
        if page_id in self._pages:
            return self._pages[page_id]

        # Check storage
        entry = self.page_table.lookup(page_id)
        if entry and entry.artifact_id:
            page = await self.storage.load_page(entry.artifact_id)
            if page:
                self._pages[page_id] = page
                return page

        return None
```

### Step 2: Add Messages as Pages

```python
class MemoryManager:
    # ... (previous code)

    async def add_user_message(self, content: str) -> MemoryPage:
        """Add a user message to memory."""
        page_id = f"msg_{len(self._pages):04d}"

        page = MemoryPage(
            page_id=page_id,
            modality=Modality.TEXT,
            content=content,
            storage_tier=StorageTier.L0,
            metadata={"role": "user"},
        )

        # Register and track
        self.combined.register(page)
        self.working_set.add_to_l0(page)
        self._pages[page_id] = page

        return page

    async def add_assistant_message(self, content: str) -> MemoryPage:
        """Add an assistant response to memory."""
        page_id = f"msg_{len(self._pages):04d}"

        page = MemoryPage(
            page_id=page_id,
            modality=Modality.TEXT,
            content=content,
            storage_tier=StorageTier.L0,
            metadata={"role": "assistant"},
        )

        self.combined.register(page)
        self.working_set.add_to_l0(page)
        self._pages[page_id] = page

        return page
```

### Step 3: Build Context for API Call

```python
class MemoryManager:
    # ... (previous code)

    def build_request(self, user_input: str) -> dict:
        """Build a Chat Completions API request."""
        # Get working set pages
        l0_ids = self.working_set.get_l0_page_ids()
        working_pages = [self._pages[pid] for pid in l0_ids if pid in self._pages]

        # Pack context
        packed = self.packer.pack(working_pages)

        # Build manifest
        builder = ManifestBuilder(
            session_id=self.session_id,
            policies=ManifestPolicies(max_faults_per_turn=3),
        )

        # Add working set
        for page in working_pages:
            builder.add_working_set_page(page)

        # Add available pages (from lower tiers)
        for entry in self.page_table.get_by_tier(StorageTier.L2):
            builder.add_available_page_entry(
                AvailablePageEntry(
                    page_id=entry.page_id,
                    modality=entry.modality.value,
                    tier=entry.tier.value,
                    levels=[0, 1, 2],
                    hint=self._generate_hint(entry.page_id),
                )
            )

        manifest = builder.build()

        # Build developer message
        developer_msg = build_vm_developer_message(
            mode=VMMode.STRICT,
            manifest_json=manifest.model_dump_json(),
            context=packed.content,
        )

        return {
            "model": "gpt-4o",
            "messages": [
                {"role": "developer", "content": developer_msg},
                {"role": "user", "content": user_input},
            ],
            "tools": get_vm_tools_as_dicts(),
        }

    def _generate_hint(self, page_id: str) -> str:
        page = self._pages.get(page_id)
        if page and page.content:
            # Simple hint from first few words
            words = page.content.split()[:5]
            return " ".join(words) + "..."
        return f"content from {page_id}"
```

### Step 4: Handle Tool Calls

```python
import json

class MemoryManager:
    # ... (previous code)

    async def process_response(self, response) -> dict | str:
        """Process LLM response, handling any tool calls."""
        message = response.choices[0].message

        if message.tool_calls:
            tool_results = []

            for tool_call in message.tool_calls:
                result = await self._handle_tool_call(tool_call)
                tool_results.append(result)

            return {"tool_results": tool_results}

        return message.content

    async def _handle_tool_call(self, tool_call) -> dict:
        """Handle a single tool call."""
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if name == "page_fault":
            return await self._handle_page_fault(tool_call.id, args)
        elif name == "search_pages":
            return await self._handle_search(tool_call.id, args)
        else:
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"error": f"Unknown tool: {name}"}),
            }

    async def _handle_page_fault(self, call_id: str, args: dict) -> dict:
        """Handle page_fault tool call."""
        page_id = args["page_id"]
        target_level = args.get("target_level", 2)

        if not self.fault_handler.can_fault():
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps({"error": "Fault limit reached"}),
            }

        result = await self.fault_handler.handle_fault(page_id, target_level)

        if result.success:
            # Add to working set
            self.working_set.add_to_l0(result.page)
            self._pages[page_id] = result.page

            # Build tool result
            tool_result = self.fault_handler.build_tool_result(result.page)
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": tool_result.model_dump_json(),
            }
        else:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps({"error": result.error}),
            }

    async def _handle_search(self, call_id: str, args: dict) -> dict:
        """Handle search_pages tool call."""
        # Build available pages list
        available = []
        for entry in self.page_table.get_by_tier(StorageTier.L2):
            available.append(AvailablePageEntry(
                page_id=entry.page_id,
                modality=entry.modality.value,
                tier=entry.tier.value,
                levels=[0, 1, 2],
                hint=self._generate_hint(entry.page_id),
            ))

        handler = PageSearchHandler(available_pages=available)
        result = handler.search(
            query=args["query"],
            modality=args.get("modality"),
            limit=args.get("limit", 5),
        )

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result.model_dump_json(),
        }
```

### Step 5: Manage Memory Pressure

```python
class MemoryManager:
    # ... (previous code)

    async def check_and_evict(self):
        """Check memory pressure and evict if needed."""
        if not self.working_set.needs_eviction():
            return

        # Get eviction candidates
        candidates = self.working_set.get_eviction_candidates(
            from_tier=StorageTier.L0,
        )

        # Calculate how much to evict
        tokens_to_free = self.working_set.calculate_eviction_target()

        freed = 0
        for page_id, score in candidates:
            if freed >= tokens_to_free:
                break

            page = self._pages.get(page_id)
            if not page:
                continue

            # Store to persistent storage
            artifact_id = await self.storage.store_page(page)

            # Update tracking
            self.page_table.update_location(
                page_id, tier=StorageTier.L2, artifact_id=artifact_id
            )
            self.working_set.remove_from_l0(page_id, page)

            freed += page.size_tokens or page.estimate_tokens()

    def new_turn(self):
        """Reset per-turn state."""
        self.fault_handler.new_turn()
```

---

## Complete Usage Example

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI()
    memory = MemoryManager(session_id="session_001")

    # Simulate conversation
    messages = [
        "What's the best way to implement authentication?",
        "Can you show me an example with JWT?",
        "What about refresh tokens?",
    ]

    for user_input in messages:
        # Add user message
        await memory.add_user_message(user_input)

        # Check memory pressure
        await memory.check_and_evict()

        # Build request
        request = memory.build_request(user_input)

        # Call API
        response = await client.chat.completions.create(**request)

        # Process response
        result = await memory.process_response(response)

        if isinstance(result, dict) and "tool_results" in result:
            # Continue conversation with tool results
            request["messages"].extend(result["tool_results"])
            response = await client.chat.completions.create(**request)
            result = await memory.process_response(response)

        # Add assistant response
        await memory.add_assistant_message(result)
        print(f"Assistant: {result[:200]}...")

        # Reset for next turn
        memory.new_turn()

asyncio.run(main())
```

---

## Integration with SessionManager

Since v0.9, `SessionManager` has built-in VM support — no subclassing needed:

```python
from chuk_ai_session_manager import SessionManager
from chuk_ai_session_manager.memory import VMMode

# Enable VM directly
sm = SessionManager(enable_vm=True, vm_mode=VMMode.STRICT)

# Conversation events automatically create memory pages
await sm.user_says("What's the best auth approach?")
await sm.ai_responds("I recommend JWT tokens...", model="gpt-4o")

# get_messages_for_llm includes VM:CONTEXT when include_system=True
messages = await sm.get_messages_for_llm(include_system=True)
```

---

## Infinite Conversation Integration

For conversations that span multiple segments:

```python
class InfiniteVMConversation:
    def __init__(self, session_id: str):
        self.memory = MemoryManager(session_id)
        self.segment_count = 0

    async def on_segment_complete(self):
        """Called when a segment reaches its limit."""
        self.segment_count += 1

        # Get all L0 pages
        l0_ids = self.memory.working_set.get_l0_page_ids()
        pages = [self.memory._pages[pid] for pid in l0_ids]

        # Create checkpoint
        checkpoint_id = await self.memory.storage.checkpoint(
            pages=pages,
            name=f"segment_{self.segment_count}",
        )

        # Create summary page
        summary_content = await self._summarize_segment(pages)
        summary_page = MemoryPage(
            page_id=f"summary_seg_{self.segment_count:02d}",
            modality=Modality.TEXT,
            content=summary_content,
            storage_tier=StorageTier.L0,
            metadata={"type": "summary"},
        )

        # Evict old pages, keep summary
        for page in pages:
            artifact_id = await self.memory.storage.store_page(page)
            self.memory.page_table.update_location(
                page.page_id, tier=StorageTier.L2, artifact_id=artifact_id
            )
            self.memory.working_set.remove_from_l0(page.page_id, page)

        # Add summary to working set
        self.memory.combined.register(summary_page)
        self.memory.working_set.add_to_l0(summary_page)
        self.memory._pages[summary_page.page_id] = summary_page

    async def _summarize_segment(self, pages: list) -> str:
        # Use LLM to summarize
        content = "\n".join(p.content for p in pages if p.content)
        # ... call summarization API
        return f"Summary of segment: {content[:100]}..."
```

---

## Best Practices

### 1. Token Budget Management

```python
# Reserve tokens for system prompt and tools
config = WorkingSetConfig(
    max_l0_tokens=128_000,
    reserved_tokens=8000,  # System + tools + safety margin
)

# Check before adding
if memory.working_set.can_fit(estimated_tokens):
    memory.working_set.add_to_l0(page)
else:
    await memory.check_and_evict()
```

### 2. Efficient Page Lookup

```python
# Use TLBWithPageTable for automatic caching
entry = memory.combined.lookup(page_id)  # TLB hit or auto-populate

# Don't bypass the TLB
# BAD: entry = memory.page_table.lookup(page_id)
# GOOD: entry = memory.combined.lookup(page_id)
```

### 3. Importance-Based Eviction

```python
# Mark important pages
memory.working_set.set_importance("msg_001", importance=0.9)

# System messages are critical
for page in system_pages:
    memory.working_set.set_importance(page.page_id, importance=1.0)
```

### 4. Graceful Degradation

```python
# If fault fails, provide fallback
result = await memory.fault_handler.handle_fault(page_id)
if not result.success:
    # Return hint instead of content
    entry = memory.page_table.lookup(page_id)
    return f"[Content unavailable: {entry.page_id}]"
```

### 5. Checkpoint Regularly

```python
# Checkpoint at natural boundaries
async def on_topic_change():
    await memory.storage.checkpoint(
        pages=get_current_pages(),
        name=f"topic_checkpoint_{datetime.utcnow().isoformat()}",
    )
```

---

## Debugging

### Inspect Memory State

```python
# Page table stats
stats = memory.page_table.get_stats()
print(f"Total pages: {stats.total_pages}")
print(f"By tier: {stats.pages_by_tier}")

# Working set stats
ws_stats = memory.working_set.get_stats()
print(f"L0 pages: {ws_stats.l0_pages}")
print(f"Tokens used: {ws_stats.tokens_used}")
print(f"Utilization: {ws_stats.utilization:.1%}")

# TLB stats
tlb_stats = memory.tlb.get_stats()
print(f"TLB hit rate: {tlb_stats.hit_rate:.1%}")

# Fault metrics
fault_metrics = memory.fault_handler.get_metrics()
print(f"Faults this turn: {fault_metrics.faults_this_turn}")
```

### Trace Page Lifecycle

```python
class TracingMemoryManager(MemoryManager):
    async def add_user_message(self, content: str):
        page = await super().add_user_message(content)
        print(f"[TRACE] Created page {page.page_id} in L0")
        return page

    async def check_and_evict(self):
        before = len(self.working_set.get_l0_page_ids())
        await super().check_and_evict()
        after = len(self.working_set.get_l0_page_ids())
        if before != after:
            print(f"[TRACE] Evicted {before - after} pages from L0")
```
