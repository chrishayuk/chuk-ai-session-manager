#!/usr/bin/env python3
"""
Comprehensive example of the AI Virtual Memory system (v0.8).

This example demonstrates:
1. Creating and managing memory pages with page types
2. Page table operations with provenance tracking
3. TLB caching
4. Working set management with pinning and anti-thrash
5. Context packing for model input
6. Manifest generation
7. Page fault handling with fault reasons and policies
8. Persistent storage with ArtifactsBridge
9. Mutation logging for debugging
10. Prefetcher for heuristic-based page loading
11. UX metrics tracking

Run with: python examples/memory_example.py
"""

import asyncio

from chuk_ai_session_manager.memory import (
    # Enums
    Actor,
    CompressionLevel,
    FaultReason,
    MessageRole,
    Modality,
    MutationType,
    PageType,
    StorageTier,
    # Core Models
    FaultPolicy,
    MemoryABI,
    MemoryPage,
    TokenBudget,
    UserExperienceMetrics,
    # Data Structures
    PageTable,
    PageTLB,
    TLBWithPageTable,
    # Working Set
    AntiThrashPolicy,
    PinnedSet,
    WorkingSetConfig,
    WorkingSetManager,
    # Context & Manifest
    ContextPackCache,
    ContextPacker,
    ManifestBuilder,
    ManifestPolicies,
    # Fault Handling
    PageFaultHandler,
    PageSearchHandler,
    # Storage
    ArtifactsBridge,
    # Mutation Log & Prefetcher
    MutationLogLite,
    SimplePrefetcher,
    # Prompts
    VMMode,
    build_vm_developer_message,
    get_vm_tools,
)


async def demo_basic_pages():
    """Demonstrate basic page creation with v0.8 page types."""
    print("\n" + "=" * 60)
    print("1. BASIC PAGE OPERATIONS (v0.8 with Page Types)")
    print("=" * 60)

    # Create a transcript page (user message)
    user_page = MemoryPage(
        page_id="msg_001",
        modality=Modality.TEXT,
        content="What's the weather like in San Francisco?",
        storage_tier=StorageTier.L0,
        page_type=PageType.TRANSCRIPT,  # v0.8: Page type
        metadata={"role": MessageRole.USER.value},
    )
    print(f"\nCreated user page: {user_page.page_id}")
    print(f"  Page Type: {user_page.page_type.value}")
    print(f"  Modality: {user_page.modality.value}")
    print(f"  Tier: {user_page.storage_tier.value}")
    print(f"  Tokens (est): {user_page.estimate_tokens()}")

    # Create an assistant response
    assistant_page = MemoryPage(
        page_id="msg_002",
        modality=Modality.TEXT,
        content="The weather in San Francisco is currently 65°F with partly cloudy skies.",
        storage_tier=StorageTier.L0,
        page_type=PageType.TRANSCRIPT,
        metadata={"role": MessageRole.ASSISTANT.value},
    )
    print(f"\nCreated assistant page: {assistant_page.page_id}")

    # Create a CLAIM page (v0.8: high-value extracted fact)
    claim_page = MemoryPage(
        page_id="claim_sf_weather",
        modality=Modality.TEXT,
        content="FACT: San Francisco current temperature is 65°F with partly cloudy conditions.",
        storage_tier=StorageTier.L1,
        page_type=PageType.CLAIM,  # Claims have lower eviction priority
        provenance=["msg_002"],  # v0.8: Track where this came from
        pinned=True,  # v0.8: Claims can be pinned
    )
    print(f"\nCreated claim page: {claim_page.page_id}")
    print(f"  Page Type: {claim_page.page_type.value}")
    print(f"  Provenance: {claim_page.provenance}")
    print(f"  Pinned: {claim_page.pinned}")

    # Create a SUMMARY page (v0.8: LLM-generated summary)
    summary_page = MemoryPage(
        page_id="summary_seg_01",
        modality=Modality.TEXT,
        content="Segment 1: User inquired about San Francisco weather. System reported 65°F, partly cloudy.",
        storage_tier=StorageTier.L2,
        page_type=PageType.SUMMARY,
        provenance=["msg_001", "msg_002"],  # Derived from these pages
        represents="msg_001",  # This is a compressed representation
        representation_level=2,  # Summary level
    )
    print(f"\nCreated summary page: {summary_page.page_id}")
    print(f"  Represents: {summary_page.represents}")
    print(f"  Representation Level: {summary_page.representation_level}")

    # Create an ARTIFACT page (tool-created)
    image_page = MemoryPage(
        page_id="img_001",
        modality=Modality.IMAGE,
        caption="Weather map showing San Francisco region",
        dimensions=(1200, 800),
        storage_tier=StorageTier.L1,
        page_type=PageType.ARTIFACT,
    )
    print(f"\nCreated artifact page: {image_page.page_id}")
    print(f"  Caption: {image_page.caption}")
    print(f"  Page Type: {image_page.page_type.value}")

    # Track access
    user_page.mark_accessed()
    print(f"\nAccess count after mark_accessed: {user_page.access_count}")

    return [user_page, assistant_page, claim_page, summary_page, image_page]


async def demo_page_table(pages):
    """Demonstrate page table operations with v0.8 features."""
    print("\n" + "=" * 60)
    print("2. PAGE TABLE OPERATIONS (with Page Types)")
    print("=" * 60)

    # Create page table
    table = PageTable()

    # Register pages
    for page in pages:
        entry = table.register(page)
        print(f"\nRegistered: {entry.page_id}")
        print(f"  Tier: {entry.tier.value}")
        print(f"  Page Type: {entry.page_type.value if entry.page_type else 'N/A'}")
        print(f"  Provenance: {entry.provenance}")

    # Lookup
    entry = table.lookup("claim_sf_weather")
    if entry:
        print("\nLookup claim_sf_weather:")
        print("  Found: True")
        print(f"  Page Type: {entry.page_type.value}")
        print(f"  Pinned: {entry.pinned}")

    # Get pages by tier
    l0_pages = table.get_by_tier(StorageTier.L0)
    print(f"\nPages in L0: {[e.page_id for e in l0_pages]}")

    l1_pages = table.get_by_tier(StorageTier.L1)
    print(f"Pages in L1: {[e.page_id for e in l1_pages]}")

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


async def demo_pinned_set():
    """Demonstrate v0.8 PinnedSet for eviction protection."""
    print("\n" + "=" * 60)
    print("4. PINNED SET (v0.8 - Eviction Protection)")
    print("=" * 60)

    pinned = PinnedSet(
        auto_pin_last_n_turns=2,
        auto_pin_system_prompt=True,
        auto_pin_claims=True,
    )

    # Explicitly pin a page
    pinned.pin("system_prompt")
    print("\nExplicitly pinned: system_prompt")

    # Auto-pin some pages
    pinned.auto_pin("claim_user_prefs")
    pinned.auto_pin("msg_recent_001")
    print("Auto-pinned: claim_user_prefs, msg_recent_001")

    # Check pinning status
    print(f"\nIs 'system_prompt' pinned? {pinned.is_pinned('system_prompt')}")
    print(f"Is 'claim_user_prefs' pinned? {pinned.is_pinned('claim_user_prefs')}")
    print(f"Is 'old_message' pinned? {pinned.is_pinned('old_message')}")

    # Get all pinned
    all_pinned = pinned.get_all_pinned()
    print(f"\nAll pinned pages: {all_pinned}")
    print(f"Total pinned count: {pinned.count()}")

    return pinned


async def demo_anti_thrash():
    """Demonstrate v0.8 AntiThrashPolicy."""
    print("\n" + "=" * 60)
    print("5. ANTI-THRASH POLICY (v0.8 - Prevent Eviction Thrashing)")
    print("=" * 60)

    policy = AntiThrashPolicy(
        eviction_cooldown_turns=3,
        fault_protection_turns=2,
    )

    # Simulate faulting a page
    print("\nTurn 1: Fault in msg_001")
    policy.record_fault("msg_001", turn=1)

    # Try to evict immediately
    can_evict = policy.can_evict("msg_001", current_turn=1)
    print(f"  Can evict msg_001 at turn 1? {can_evict}")

    # Try after protection period
    can_evict = policy.can_evict("msg_001", current_turn=4)
    print(f"  Can evict msg_001 at turn 4? {can_evict}")

    # Check eviction penalty
    penalty = policy.get_eviction_penalty("msg_001", current_turn=2)
    print(f"\n  Eviction penalty at turn 2: {penalty:.2f}")

    penalty = policy.get_eviction_penalty("msg_001", current_turn=5)
    print(f"  Eviction penalty at turn 5: {penalty:.2f}")

    return policy


async def demo_working_set(pages):
    """Demonstrate working set management with v0.8 features."""
    print("\n" + "=" * 60)
    print("6. WORKING SET MANAGEMENT (with Pinning & Anti-Thrash)")
    print("=" * 60)

    # Create working set manager with v0.8 features
    config = WorkingSetConfig(
        max_l0_tokens=8000,
        max_l1_pages=50,
        eviction_threshold=0.85,
        target_utilization=0.70,
        reserved_tokens=1000,
    )
    budget = TokenBudget(total_limit=8000, reserved=1000)
    manager = WorkingSetManager(config=config, budget=budget)

    # v0.8: Pin claim pages
    manager.pinned_set.pin("claim_sf_weather")

    # Add pages to L0
    for page in pages[:3]:  # Text pages
        page.size_tokens = page.estimate_tokens()
        success = manager.add_to_l0(page)
        print(f"\nAdded {page.page_id} to L0: {success}")
        print(f"  Tokens used: {manager.tokens_used}")

    # Add image to L1 (cache)
    manager.add_to_l1(pages[4])
    print(f"\nAdded {pages[4].page_id} to L1")

    # Check status
    print("\nWorking set status:")
    print(f"  L0 pages: {manager.l0_count}")
    print(f"  L1 pages: {manager.l1_count}")
    print(f"  Utilization: {manager.utilization:.2%}")
    print(f"  Pinned pages: {manager.pinned_set.count()}")

    # Get eviction candidates (respects pinning)
    candidates = manager.get_eviction_candidates(tokens_needed=1000)
    print("\nEviction candidates (excluding pinned):")
    for page_id, score in candidates[:3]:
        is_pinned = manager.pinned_set.is_pinned(page_id)
        print(f"  {page_id}: score={score:.2f}, pinned={is_pinned}")

    return manager


async def demo_context_pack_cache():
    """Demonstrate v0.8 ContextPackCache."""
    print("\n" + "=" * 60)
    print("7. CONTEXT PACK CACHE (v0.8 - Avoid Repacking)")
    print("=" * 60)

    cache = ContextPackCache(max_entries=10)

    # Compute working set hash
    working_set_hash = ContextPackCache.compute_working_set_hash(
        page_ids=["msg_001", "msg_002", "claim_001"],
        versions={"msg_001": 1, "msg_002": 1, "claim_001": 1},
    )
    print(f"\nWorking set hash: {working_set_hash}")

    # Simulate caching a packed context
    from chuk_ai_session_manager.memory.pack_cache import PackedContext

    packed = PackedContext(
        vm_context="<VM:CONTEXT>...</VM:CONTEXT>",
        vm_manifest_json='{"session_id": "test"}',
        page_ids=["msg_001", "msg_002", "claim_001"],
        tokens_used=1500,
    )

    cache.put(
        session_id="sess_001",
        model_id="gpt-4",
        token_budget=8000,
        working_set_hash=working_set_hash,
        packed=packed,
    )
    print("\nCached packed context (1500 tokens)")

    # Retrieve from cache
    result = cache.get("sess_001", "gpt-4", 8000, working_set_hash)
    print(f"Cache hit: {result is not None}")
    if result:
        print(f"  Tokens used: {result.tokens_used}")

    # Miss with different hash
    different_hash = ContextPackCache.compute_working_set_hash(["msg_001", "msg_003"])
    result = cache.get("sess_001", "gpt-4", 8000, different_hash)
    print(f"Cache hit with different working set: {result is not None}")

    # Stats
    print("\nCache stats:")
    stats = cache.get_stats()
    print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")
    print(f"  Hit rate: {cache.hit_rate:.2%}")

    return cache


async def demo_mutation_log():
    """Demonstrate v0.8 MutationLogLite for debugging."""
    print("\n" + "=" * 60)
    print("8. MUTATION LOG (v0.8 - Debug & Replay)")
    print("=" * 60)

    log = MutationLogLite(session_id="sess_demo")

    # Record some mutations
    log.record_mutation(
        page_id="msg_001",
        mutation_type=MutationType.CREATE,
        tier_after=StorageTier.L0,
        actor=Actor.USER,
        turn=1,
    )
    print("\nRecorded: CREATE msg_001 at turn 1")

    log.record_mutation(
        page_id="msg_001",
        mutation_type=MutationType.EVICT,
        tier_before=StorageTier.L0,
        tier_after=StorageTier.L2,
        actor=Actor.SYSTEM,
        cause="eviction_pressure",
        turn=5,
    )
    print("Recorded: EVICT msg_001 at turn 5")

    log.record_mutation(
        page_id="msg_001",
        mutation_type=MutationType.FAULT_IN,
        tier_before=StorageTier.L2,
        tier_after=StorageTier.L0,
        actor=Actor.MODEL,
        cause="user_requested_recall",
        turn=8,
    )
    print("Recorded: FAULT_IN msg_001 at turn 8")

    # Record context snapshot
    log.record_context_at_turn(8, ["msg_001", "msg_002", "claim_001"])
    print("\nRecorded context snapshot at turn 8")

    # Query history
    history = log.get_history("msg_001")
    print(f"\nHistory for msg_001 ({len(history)} mutations):")
    for m in history:
        print(f"  Turn {m.turn}: {m.mutation_type.value} by {m.actor.value}")

    # Get context at turn
    context = log.get_context_at_turn(8)
    print(f"\nContext at turn 8: {context}")

    # Summary
    summary = log.get_summary()
    print("\nMutation log summary:")
    print(f"  Total mutations: {summary['total_mutations']}")
    print(f"  Unique pages: {summary['unique_pages']}")
    print(
        f"  Creates: {summary['creates']}, Faults: {summary['faults']}, Evictions: {summary['evictions']}"
    )

    return log


async def demo_prefetcher():
    """Demonstrate v0.8 SimplePrefetcher."""
    print("\n" + "=" * 60)
    print("9. SIMPLE PREFETCHER (v0.8 - Heuristic Prefetch)")
    print("=" * 60)

    prefetcher = SimplePrefetcher(
        max_claims_to_prefetch=3,
        max_recent_tools=3,
    )

    # Record page accesses
    for _ in range(5):
        prefetcher.record_page_access("claim_user_prefs")
    for _ in range(3):
        prefetcher.record_page_access("claim_location")
    prefetcher.record_page_access("msg_001")
    print(
        "\nRecorded page accesses (claim_user_prefs: 5, claim_location: 3, msg_001: 1)"
    )

    # Record tool calls with prereqs
    prefetcher.record_tool_call(
        tool_name="weather_api",
        turn=1,
        pages_accessed_before=["claim_location", "claim_user_prefs"],
    )
    prefetcher.record_tool_call(
        tool_name="weather_api",
        turn=3,
        pages_accessed_before=["claim_location"],
    )
    print("Recorded tool calls: weather_api uses claim_location, claim_user_prefs")

    # Set last segment summary
    prefetcher.set_last_segment_summary("summary_seg_05")
    print("Set last segment summary: summary_seg_05")

    # Get prefetch recommendations
    pages = await prefetcher.prefetch_on_turn_start("sess_demo")
    print(f"\nRecommended prefetch for turn start: {pages}")

    # Get likely tools
    likely_tools = prefetcher.get_likely_tools()
    print(f"Likely tools: {likely_tools}")

    # Get tool prereqs
    prereqs = prefetcher.get_tool_prereq_pages("weather_api")
    print(f"Prereqs for weather_api: {prereqs}")

    # Stats
    stats = prefetcher.get_stats()
    print(f"\nPrefetcher stats: {stats}")

    return prefetcher


async def demo_fault_policy():
    """Demonstrate v0.8 FaultPolicy."""
    print("\n" + "=" * 60)
    print("10. FAULT POLICY (v0.8 - Rate Limiting)")
    print("=" * 60)

    policy = FaultPolicy(
        max_faults_per_turn=3,
        max_fault_tokens_per_turn=8192,
    )

    print("\nFault policy limits:")
    print(f"  Max faults/turn: {policy.max_faults_per_turn}")
    print(f"  Max tokens/turn: {policy.max_fault_tokens_per_turn}")

    # Simulate faults
    print("\nSimulating faults:")
    for i in range(4):
        tokens = 2000
        can_fault = policy.can_fault(tokens)
        print(f"  Fault {i + 1}: can_fault({tokens})={can_fault}")
        if can_fault:
            policy.record_fault(tokens)
            print(
                f"    → Recorded. Faults: {policy.faults_this_turn}, Tokens: {policy.tokens_used_this_turn}"
            )

    # New turn resets
    print("\nNew turn:")
    policy.new_turn()
    print(f"  Faults reset to: {policy.faults_this_turn}")
    print(f"  Can fault now: {policy.can_fault(1000)}")

    return policy


async def demo_ux_metrics():
    """Demonstrate v0.8 UserExperienceMetrics."""
    print("\n" + "=" * 60)
    print("11. UX METRICS (v0.8 - Quality Tracking)")
    print("=" * 60)

    metrics = UserExperienceMetrics()

    # Record some faults
    metrics.record_fault(
        "page_a", FaultReason.USER_REQUESTED_RECALL, turn=1, tokens_loaded=500
    )
    metrics.record_fault(
        "page_b", FaultReason.RESOLVE_REFERENCE, turn=2, tokens_loaded=300
    )
    metrics.record_fault(
        "page_a", FaultReason.RESOLVE_REFERENCE, turn=3, tokens_loaded=500
    )  # Repeat - thrash!
    print("\nRecorded faults: page_a (turn 1), page_b (turn 2), page_a again (turn 3)")

    # Record recall attempts
    metrics.record_recall_attempt(
        turn=1,
        query="What did we discuss about weather?",
        page_ids_cited=["page_a"],
        user_corrected=False,
    )
    metrics.record_recall_attempt(
        turn=2,
        query="What was the temperature?",
        page_ids_cited=["page_b"],
        user_corrected=True,  # User corrected = failed recall
    )
    print("Recorded recall attempts: 1 success, 1 user-corrected")

    # Calculate metrics
    print("\nUX Metrics:")
    print(f"  Recall success rate: {metrics.recall_success_rate():.2%}")
    print(f"  Thrash index (5 turns): {metrics.thrash_index(window_turns=5):.2f}")

    # Fault reason breakdown
    breakdown = metrics.get_fault_reason_breakdown()
    print("\nFault reason breakdown:")
    for reason, count in breakdown.items():
        print(f"  {reason.value}: {count}")

    return metrics


async def demo_memory_abi():
    """Demonstrate v0.8 MemoryABI."""
    print("\n" + "=" * 60)
    print("12. MEMORY ABI (v0.8 - Token Budget Contract)")
    print("=" * 60)

    abi = MemoryABI(
        max_context_tokens=128000,
        reserved_tokens=2000,
        tool_schema_tokens_reserved=500,  # v0.8: Reserve for tool schemas
        faults_allowed=True,
    )

    print("\nMemory ABI configuration:")
    print(f"  Max context tokens: {abi.max_context_tokens}")
    print(f"  Reserved tokens: {abi.reserved_tokens}")
    print(f"  Tool schema reserved: {abi.tool_schema_tokens_reserved}")
    print(f"  Available for pages: {abi.available_tokens}")
    print(f"  Faults allowed: {abi.faults_allowed}")

    return abi


async def demo_context_packer(pages):
    """Demonstrate context packing for model input."""
    print("\n" + "=" * 60)
    print("13. CONTEXT PACKING")
    print("=" * 60)

    packer = ContextPacker()

    # Pack pages into VM:CONTEXT format
    result = packer.pack(pages)
    print(f"\nPacked {len(result.pages_included)} pages")
    print(f"Estimated tokens: {result.tokens_est}")

    print("\nVM:CONTEXT content:")
    print("-" * 40)
    for line in result.content.split("\n")[:10]:  # First 10 lines
        print(f"  {line}")
    print("  ...")
    print("-" * 40)

    return result


async def demo_manifest(table, working_set_ids):
    """Demonstrate manifest generation with v0.8 page types."""
    print("\n" + "=" * 60)
    print("14. MANIFEST GENERATION (with Page Types)")
    print("=" * 60)

    # Add some more pages to make manifest interesting
    for i in range(3, 8):
        page = MemoryPage(
            page_id=f"msg_{i:03d}",
            modality=Modality.TEXT,
            content=f"Historical message {i}",
            storage_tier=StorageTier.L2,
            page_type=PageType.TRANSCRIPT,
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
        hint_generator=lambda e: f"{e.page_type.value if e.page_type else 'unknown'}: {e.modality.value}",
    )

    print(f"\nManifest for session: {manifest.session_id}")
    print(f"\nWorking set ({len(manifest.working_set)} pages):")
    for ws in manifest.working_set:
        print(
            f"  {ws.page_id}: {ws.modality}, level={ws.level}, tokens={ws.tokens_est}"
        )

    print(f"\nAvailable pages ({len(manifest.available_pages)} pages):")
    for ap in manifest.available_pages[:5]:
        print(f"  {ap.page_id}: tier={ap.tier}, hint='{ap.hint}'")

    print("\nPolicies:")
    print(f"  Max faults/turn: {manifest.policies.max_faults_per_turn}")
    print(f"  Upgrade budget: {manifest.policies.upgrade_budget_tokens} tokens")

    return manifest


async def demo_fault_handler(table, pages):
    """Demonstrate page fault handling with v0.8 features."""
    print("\n" + "=" * 60)
    print("15. PAGE FAULT HANDLING (with Fault Reasons)")
    print("=" * 60)

    # Setup handler
    handler = PageFaultHandler(max_faults_per_turn=3)
    handler.configure(page_table=table)

    # Store pages for retrieval
    for page in pages:
        handler.store_page(page)

    # Simulate a page fault with reason
    print("\nHandling fault for msg_001 (user_requested_recall)...")
    result = await handler.handle_fault("msg_001", target_level=CompressionLevel.FULL)
    print(f"  Success: {result.success}")
    print(f"  Source tier: {result.source_tier.value if result.source_tier else 'N/A'}")
    print(f"  Latency: {result.latency_ms:.2f}ms")

    if result.page:
        print(
            f"  Page type: {result.page.page_type.value if result.page.page_type else 'N/A'}"
        )
        print(f"  Content: {result.page.content[:50]}...")

    # Build tool result envelope
    tool_result = handler.build_tool_result(result)
    print("\nTool result envelope:")
    print(f"  Page ID: {tool_result.page.page_id}")
    print(f"  Promoted: {tool_result.effects.promoted_to_working_set}")

    # Check fault limit
    print(f"\nFaults this turn: {handler.faults_this_turn}")
    print(f"Can fault again: {handler.can_fault()}")

    return handler


async def demo_search_handler(table):
    """Demonstrate page search with v0.8 page types."""
    print("\n" + "=" * 60)
    print("16. PAGE SEARCH (with Page Type Filtering)")
    print("=" * 60)

    handler = PageSearchHandler()
    handler.configure(page_table=table)

    # Set hints for pages
    handler.set_hint("msg_001", "weather query san francisco")
    handler.set_hint("msg_002", "weather response temperature")
    handler.set_hint("claim_sf_weather", "fact: san francisco 65F partly cloudy")
    handler.set_hint("img_001", "weather map visualization")

    # Search for weather-related content
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
    print("17. PERSISTENT STORAGE (ArtifactsBridge)")
    print("=" * 60)

    bridge = ArtifactsBridge()
    await bridge.configure(session_id="demo_session")

    print(f"\nStorage backend: {bridge.get_stats().backend}")
    print(f"Persistent: {bridge.is_persistent}")

    # Create and store a page with v0.8 page type
    page = MemoryPage(
        page_id="archived_001",
        modality=Modality.TEXT,
        content="This is archived content that needs to be persisted.",
        storage_tier=StorageTier.L3,
        page_type=PageType.ARTIFACT,
    )

    artifact_id = await bridge.store_page(page, StorageTier.L3)
    print(f"\nStored page, artifact_id: {artifact_id}")

    # Load it back
    loaded = await bridge.load_page(artifact_id)
    print(f"Loaded page: {loaded.page_id}")
    print(f"  Content matches: {loaded.content == page.content}")

    # Stats
    stats = bridge.get_stats()
    print("\nStorage stats:")
    print(f"  Pages stored: {stats.pages_stored}")

    return bridge


async def demo_full_integration():
    """Demonstrate full integration: building developer message."""
    print("\n" + "=" * 60)
    print("18. FULL INTEGRATION: Developer Message")
    print("=" * 60)

    # Create pages with v0.8 page types
    pages = [
        MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            content="Tell me about quantum computing",
            page_type=PageType.TRANSCRIPT,
            metadata={"role": MessageRole.USER.value},
        ),
        MemoryPage(
            page_id="msg_002",
            modality=Modality.TEXT,
            content="Quantum computing uses quantum mechanical phenomena...",
            page_type=PageType.TRANSCRIPT,
            metadata={"role": MessageRole.ASSISTANT.value},
        ),
        MemoryPage(
            page_id="claim_quantum",
            modality=Modality.TEXT,
            content="FACT: Quantum computers use qubits instead of classical bits.",
            page_type=PageType.CLAIM,
            provenance=["msg_002"],
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
            page_type=PageType.TRANSCRIPT,
        )
        table.register(historical)

    # Build manifest
    builder = ManifestBuilder()
    manifest = builder.build(
        session_id="sess_integration",
        page_table=table,
        working_set_ids=["msg_001", "msg_002", "claim_quantum"],
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
        max_faults_per_turn=3,
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
    print("AI VIRTUAL MEMORY SYSTEM DEMO (v0.8)")
    print("=" * 60)

    # 1. Basic pages with page types
    pages = await demo_basic_pages()

    # 2. Page table
    table = await demo_page_table(pages)

    # 3. TLB
    await demo_tlb(table)

    # 4. PinnedSet (v0.8)
    await demo_pinned_set()

    # 5. AntiThrashPolicy (v0.8)
    await demo_anti_thrash()

    # 6. Working set with pinning
    await demo_working_set(pages)

    # 7. Context pack cache (v0.8)
    await demo_context_pack_cache()

    # 8. Mutation log (v0.8)
    await demo_mutation_log()

    # 9. Prefetcher (v0.8)
    await demo_prefetcher()

    # 10. Fault policy (v0.8)
    await demo_fault_policy()

    # 11. UX metrics (v0.8)
    await demo_ux_metrics()

    # 12. Memory ABI (v0.8)
    await demo_memory_abi()

    # 13. Context packing
    await demo_context_packer(pages)

    # 14. Manifest
    await demo_manifest(table, ["msg_001", "msg_002", "claim_sf_weather"])

    # 15. Fault handling
    await demo_fault_handler(table, pages)

    # 16. Search
    await demo_search_handler(table)

    # 17. Storage
    await demo_artifacts_bridge()

    # 18. Full integration
    await demo_full_integration()


if __name__ == "__main__":
    asyncio.run(main())
