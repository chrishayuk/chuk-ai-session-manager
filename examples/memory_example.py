#!/usr/bin/env python3
"""
Comprehensive example of the AI Virtual Memory system.

This example demonstrates:
1. Creating and managing memory pages
2. Page table operations
3. TLB caching
4. Working set management
5. Context packing for model input
6. Manifest generation
7. Page fault handling
8. Persistent storage with ArtifactsBridge

Run with: python examples/memory_example.py
"""

import asyncio

from chuk_ai_session_manager.memory import (
    # Enums
    CompressionLevel,
    MessageRole,
    Modality,
    StorageTier,
    MemoryPage,
    TokenBudget,
    # Data Structures
    PageTable,
    PageTLB,
    TLBWithPageTable,
    # Working Set
    WorkingSetConfig,
    WorkingSetManager,
    # Context & Manifest
    ContextPacker,
    ManifestBuilder,
    ManifestPolicies,
    # Fault Handling
    PageFaultHandler,
    PageSearchHandler,
    # Storage
    ArtifactsBridge,
    # Prompts
    VMMode,
    build_vm_developer_message,
    get_vm_tools,
)


async def demo_basic_pages():
    """Demonstrate basic page creation and operations."""
    print("\n" + "=" * 60)
    print("1. BASIC PAGE OPERATIONS")
    print("=" * 60)

    # Create a text page (user message)
    user_page = MemoryPage(
        page_id="msg_001",
        modality=Modality.TEXT,
        content="What's the weather like in San Francisco?",
        storage_tier=StorageTier.L0,  # In context window
        metadata={"role": MessageRole.USER.value},
    )
    print(f"\nCreated user page: {user_page.page_id}")
    print(f"  Modality: {user_page.modality.value}")
    print(f"  Tier: {user_page.storage_tier.value}")
    print(f"  Tokens (est): {user_page.estimate_tokens()}")

    # Create an assistant response
    assistant_page = MemoryPage(
        page_id="msg_002",
        modality=Modality.TEXT,
        content="The weather in San Francisco is currently 65°F with partly cloudy skies.",
        storage_tier=StorageTier.L0,
        metadata={"role": MessageRole.ASSISTANT.value},
    )
    print(f"\nCreated assistant page: {assistant_page.page_id}")

    # Create an image page
    image_page = MemoryPage(
        page_id="img_001",
        modality=Modality.IMAGE,
        caption="Weather map showing San Francisco region",
        dimensions=(1200, 800),
        storage_tier=StorageTier.L1,  # In cache, not context
    )
    print(f"\nCreated image page: {image_page.page_id}")
    print(f"  Caption: {image_page.caption}")
    print(f"  Dimensions: {image_page.dimensions}")

    # Track access
    user_page.mark_accessed()
    print(f"\nAccess count after mark_accessed: {user_page.access_count}")

    return [user_page, assistant_page, image_page]


async def demo_page_table(pages):
    """Demonstrate page table operations."""
    print("\n" + "=" * 60)
    print("2. PAGE TABLE OPERATIONS")
    print("=" * 60)

    # Create page table
    table = PageTable()

    # Register pages
    for page in pages:
        entry = table.register(page)
        print(f"\nRegistered: {entry.page_id}")
        print(f"  Tier: {entry.tier.value}")
        print(f"  Modality: {entry.modality.value}")

    # Lookup
    entry = table.lookup("msg_001")
    print(f"\nLookup msg_001: found={entry is not None}")

    # Get pages by tier
    l0_pages = table.get_by_tier(StorageTier.L0)
    print(f"\nPages in L0: {[e.page_id for e in l0_pages]}")

    l1_pages = table.get_by_tier(StorageTier.L1)
    print(f"Pages in L1: {[e.page_id for e in l1_pages]}")

    # Move a page to a different tier
    table.update_location("msg_001", StorageTier.L2)
    print("\nMoved msg_001 to L2")

    # Get stats
    stats = table.get_stats()
    print("\nPage table stats:")
    print(f"  Total pages: {stats.total_pages}")
    print(f"  Working set size: {stats.working_set_size}")
    print(f"  Pages by tier: {dict(stats.pages_by_tier)}")

    return table


async def demo_tlb(table):
    """Demonstrate TLB caching."""
    print("\n" + "=" * 60)
    print("3. TLB (Translation Lookaside Buffer)")
    print("=" * 60)

    # Create TLB
    tlb = PageTLB(max_entries=100)

    # Create combined wrapper
    wrapper = TLBWithPageTable(table, tlb)

    # First lookup - TLB miss, will populate from page table
    entry = wrapper.lookup("msg_002")
    print(f"\nFirst lookup msg_002: found={entry is not None}")
    print(f"  TLB hits: {tlb.hits}, misses: {tlb.misses}")

    # Second lookup - TLB hit
    entry = wrapper.lookup("msg_002")
    print(f"\nSecond lookup msg_002: found={entry is not None}")
    print(f"  TLB hits: {tlb.hits}, misses: {tlb.misses}")

    # Get TLB stats
    stats = tlb.get_stats()
    print("\nTLB stats:")
    print(f"  Size: {stats.size}/{stats.max_size}")
    print(f"  Hit rate: {stats.hit_rate:.2%}")

    return wrapper


async def demo_working_set(pages):
    """Demonstrate working set management."""
    print("\n" + "=" * 60)
    print("4. WORKING SET MANAGEMENT")
    print("=" * 60)

    # Create working set manager with custom config
    config = WorkingSetConfig(
        max_l0_tokens=8000,
        max_l1_pages=50,
        eviction_threshold=0.85,
        target_utilization=0.70,
        reserved_tokens=1000,
    )
    budget = TokenBudget(total_limit=8000, reserved=1000)
    manager = WorkingSetManager(config=config, budget=budget)

    # Add pages to L0
    for page in pages[:2]:  # Text pages
        page.size_tokens = page.estimate_tokens()
        success = manager.add_to_l0(page)
        print(f"\nAdded {page.page_id} to L0: {success}")
        print(f"  Tokens used: {manager.tokens_used}")

    # Add image to L1 (cache)
    manager.add_to_l1(pages[2])
    print(f"\nAdded {pages[2].page_id} to L1")

    # Check status
    print("\nWorking set status:")
    print(f"  L0 pages: {manager.l0_count}")
    print(f"  L1 pages: {manager.l1_count}")
    print(f"  Utilization: {manager.utilization:.2%}")
    print(f"  Needs eviction: {manager.needs_eviction()}")

    # Get eviction candidates
    candidates = manager.get_eviction_candidates(tokens_needed=1000)
    print(f"\nEviction candidates: {[(c[0], f'{c[1]:.2f}') for c in candidates[:3]]}")

    # Get stats
    stats = manager.get_stats()
    print("\nWorking set stats:")
    print(f"  Total pages: {stats.total_pages}")
    print(f"  Tokens available: {stats.tokens_available}")

    return manager


async def demo_context_packer(pages):
    """Demonstrate context packing for model input."""
    print("\n" + "=" * 60)
    print("5. CONTEXT PACKING")
    print("=" * 60)

    packer = ContextPacker()

    # Pack pages into VM:CONTEXT format
    result = packer.pack(pages)
    print(f"\nPacked {len(result.pages_included)} pages")
    print(f"Estimated tokens: {result.tokens_est}")

    print("\nVM:CONTEXT content:")
    print("-" * 40)
    for line in result.content.split("\n"):
        print(f"  {line}")
    print("-" * 40)

    # Pack with wrapper tags
    wrapped = packer.pack_with_wrapper(pages)
    print("\nWith wrapper (first 200 chars):")
    print(wrapped.content[:200] + "...")

    return result


async def demo_manifest(table, working_set_ids):
    """Demonstrate manifest generation."""
    print("\n" + "=" * 60)
    print("6. MANIFEST GENERATION")
    print("=" * 60)

    # Add some more pages to make manifest interesting
    for i in range(3, 8):
        page = MemoryPage(
            page_id=f"msg_{i:03d}",
            modality=Modality.TEXT,
            content=f"Historical message {i}",
            storage_tier=StorageTier.L2,  # Not in working set
        )
        table.register(page)

    # Build manifest
    builder = ManifestBuilder()
    policies = ManifestPolicies(
        max_faults_per_turn=3,
        upgrade_budget_tokens=2048,
    )

    manifest = builder.build(
        session_id="session_abc123",
        page_table=table,
        working_set_ids=working_set_ids,
        policies=policies,
        hint_generator=lambda e: f"{e.modality.value} from {e.tier.value}",
    )

    print(f"\nManifest for session: {manifest.session_id}")
    print(f"\nWorking set ({len(manifest.working_set)} pages):")
    for ws in manifest.working_set:
        print(
            f"  {ws.page_id}: {ws.modality}, level={ws.level}, tokens={ws.tokens_est}"
        )

    print(f"\nAvailable pages ({len(manifest.available_pages)} pages):")
    for ap in manifest.available_pages[:5]:
        print(f"  {ap.page_id}: {ap.modality}, tier={ap.tier}, hint='{ap.hint}'")

    print("\nPolicies:")
    print(f"  Max faults/turn: {manifest.policies.max_faults_per_turn}")
    print(f"  Upgrade budget: {manifest.policies.upgrade_budget_tokens} tokens")

    # JSON output
    print("\nManifest JSON (first 300 chars):")
    print(manifest.to_json()[:300] + "...")

    return manifest


async def demo_fault_handler(table, pages):
    """Demonstrate page fault handling."""
    print("\n" + "=" * 60)
    print("7. PAGE FAULT HANDLING")
    print("=" * 60)

    # Setup handler
    handler = PageFaultHandler(max_faults_per_turn=2)
    handler.configure(page_table=table)

    # Store pages for retrieval
    for page in pages:
        handler.store_page(page)

    # Simulate a page fault
    print("\nHandling fault for msg_001...")
    result = await handler.handle_fault("msg_001", target_level=CompressionLevel.FULL)
    print(f"  Success: {result.success}")
    print(f"  Source tier: {result.source_tier.value if result.source_tier else 'N/A'}")
    print(f"  Latency: {result.latency_ms:.2f}ms")

    if result.page:
        print(f"  Page content: {result.page.content[:50]}...")

    # Build tool result envelope
    tool_result = handler.build_tool_result(result)
    print("\nTool result envelope:")
    print(f"  Page ID: {tool_result.page.page_id}")
    print(f"  Modality: {tool_result.page.modality}")
    print(f"  Promoted: {tool_result.effects.promoted_to_working_set}")

    # Try another fault
    print("\nHandling fault for img_001...")
    result2 = await handler.handle_fault(
        "img_001", target_level=CompressionLevel.ABSTRACT
    )
    print(f"  Success: {result2.success}")

    # Check fault limit
    print(f"\nFaults this turn: {handler.faults_this_turn}")
    print(f"Can fault again: {handler.can_fault()}")

    # Get metrics
    metrics = handler.get_metrics()
    print("\nFault metrics:")
    print(f"  Total faults: {metrics.total_faults}")
    print(f"  Faults remaining: {metrics.faults_remaining}")

    return handler


async def demo_search_handler(table):
    """Demonstrate page search."""
    print("\n" + "=" * 60)
    print("8. PAGE SEARCH")
    print("=" * 60)

    handler = PageSearchHandler()
    handler.configure(page_table=table)

    # Set hints for pages
    handler.set_hint("msg_001", "weather query san francisco")
    handler.set_hint("msg_002", "weather response temperature")
    handler.set_hint("img_001", "weather map visualization")

    # Search
    result = await handler.search("weather", limit=5)
    print("\nSearch results for 'weather':")
    print(f"  Total available: {result.total_available}")
    print(f"  Matches found: {len(result.results)}")

    for r in result.results:
        print(f"  - {r.page_id}: {r.modality}, relevance={r.relevance:.2f}")

    return handler


async def demo_artifacts_bridge():
    """Demonstrate persistent storage."""
    print("\n" + "=" * 60)
    print("9. PERSISTENT STORAGE (ArtifactsBridge)")
    print("=" * 60)

    bridge = ArtifactsBridge()
    await bridge.configure(session_id="demo_session")

    print(f"\nStorage backend: {bridge.get_stats().backend}")
    print(f"Persistent: {bridge.is_persistent}")

    # Create and store a page
    page = MemoryPage(
        page_id="archived_001",
        modality=Modality.TEXT,
        content="This is archived content that needs to be persisted.",
        storage_tier=StorageTier.L3,
    )

    artifact_id = await bridge.store_page(page, StorageTier.L3)
    print(f"\nStored page, artifact_id: {artifact_id}")

    # Load it back
    loaded = await bridge.load_page(artifact_id)
    print(f"Loaded page: {loaded.page_id}")
    print(f"  Content matches: {loaded.content == page.content}")

    # Store a checkpoint
    pages = [
        MemoryPage(
            page_id=f"cp_{i}", modality=Modality.TEXT, content=f"Checkpoint content {i}"
        )
        for i in range(3)
    ]
    checkpoint_id = await bridge.store_checkpoint(pages, "demo_checkpoint")
    print(f"\nStored checkpoint: {checkpoint_id}")

    # Load checkpoint
    restored = await bridge.load_checkpoint(checkpoint_id)
    print(f"Restored {len(restored)} pages from checkpoint")

    # Stats
    stats = bridge.get_stats()
    print("\nStorage stats:")
    print(f"  Pages stored: {stats.pages_stored}")

    return bridge


async def demo_full_integration():
    """Demonstrate full integration: building developer message."""
    print("\n" + "=" * 60)
    print("10. FULL INTEGRATION: Developer Message")
    print("=" * 60)

    # Create pages
    pages = [
        MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            content="Tell me about quantum computing",
            metadata={"role": MessageRole.USER.value},
        ),
        MemoryPage(
            page_id="msg_002",
            modality=Modality.TEXT,
            content="Quantum computing uses quantum mechanical phenomena...",
            metadata={"role": MessageRole.ASSISTANT.value},
        ),
    ]

    # Setup page table
    table = PageTable()
    for page in pages:
        table.register(page)

    # Add historical pages
    for i in range(3, 10):
        historical = MemoryPage(
            page_id=f"msg_{i:03d}",
            modality=Modality.TEXT,
            storage_tier=StorageTier.L2,
        )
        table.register(historical)

    # Build manifest
    builder = ManifestBuilder()
    manifest = builder.build(
        session_id="sess_integration",
        page_table=table,
        working_set_ids=["msg_001", "msg_002"],
    )

    # Pack context
    packer = ContextPacker()
    context = packer.pack(pages)

    # Build full developer message
    dev_message = build_vm_developer_message(
        mode=VMMode.STRICT,
        manifest_json=manifest.to_json(),
        context=context.content,
        system_prompt="You are a helpful AI assistant.",
        max_faults_per_turn=2,
    )

    print("\nDeveloper message structure:")
    print(f"  Total length: {len(dev_message)} chars")
    print(f"  Contains VM:RULES: {'<VM:RULES>' in dev_message}")
    print(f"  Contains VM:MANIFEST_JSON: {'<VM:MANIFEST_JSON>' in dev_message}")
    print(f"  Contains VM:CONTEXT: {'<VM:CONTEXT>' in dev_message}")

    # Get tools
    tools = get_vm_tools(include_search=True)
    print(f"\nVM Tools ({len(tools)}):")
    for tool in tools:
        print(f"  - {tool.function.name}: {tool.function.description[:50]}...")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("AI VIRTUAL MEMORY SYSTEM DEMO")
    print("=" * 60)

    # 1. Basic pages
    pages = await demo_basic_pages()

    # 2. Page table
    table = await demo_page_table(pages)

    # 3. TLB
    await demo_tlb(table)

    # 4. Working set
    await demo_working_set(pages)

    # 5. Context packing
    await demo_context_packer(pages)

    # 6. Manifest
    await demo_manifest(table, ["msg_001", "msg_002"])

    # 7. Fault handling
    await demo_fault_handler(table, pages)

    # 8. Search
    await demo_search_handler(table)

    # 9. Storage
    await demo_artifacts_bridge()

    # 10. Full integration
    await demo_full_integration()


if __name__ == "__main__":
    asyncio.run(main())
