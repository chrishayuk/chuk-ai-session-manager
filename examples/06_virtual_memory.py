#!/usr/bin/env python3
"""
Virtual Memory Integration Demo (v0.9)

Demonstrates the MemoryManager orchestrator and SessionManager VM integration.
No API keys required — runs entirely locally.

Shows:
1. SessionManager with enable_vm=True (automatic page creation)
2. MemoryManager standalone usage (create, fault, evict, build context)
3. get_vm_context() for LLM integration
4. Stats and diagnostics across all subsystems
5. Segmentation hook (infinite context + VM rollover)
6. Demand paging pre-pass (recall signal detection + prefetch)

Usage:
    python examples/06_virtual_memory.py
"""

import asyncio

from chuk_ai_session_manager import SessionManager
from chuk_ai_session_manager.memory.manager import MemoryManager
from chuk_ai_session_manager.memory.models import (
    PageType,
    StorageTier,
    VMMode,
)
from chuk_ai_session_manager.memory.working_set import WorkingSetConfig


async def demo_session_manager_integration():
    """SessionManager with enable_vm=True — zero-config VM."""
    print("=" * 60)
    print("1. SessionManager + Virtual Memory (enable_vm=True)")
    print("=" * 60)

    sm = SessionManager(
        system_prompt="You are a helpful coding assistant.",
        enable_vm=True,
        vm_mode=VMMode.STRICT,
    )

    # Normal conversation — VM pages created automatically
    await sm.user_says("I'm building a REST API with FastAPI.")
    await sm.ai_responds(
        "Great choice! FastAPI is async-native and generates OpenAPI docs automatically.",
        model="gpt-4o",
    )
    await sm.user_says("How should I handle authentication?")
    await sm.ai_responds(
        "Use OAuth2 with JWT tokens. FastAPI has built-in OAuth2PasswordBearer.",
        model="gpt-4o",
    )
    await sm.tool_used(
        "code_search",
        {"query": "fastapi auth example"},
        "Found 3 examples in docs/auth/",
    )

    # Check what's in the VM
    vm = sm.vm
    assert vm is not None

    l0_pages = vm.get_l0_pages()
    print(f"\nPages in context (L0): {len(l0_pages)}")
    for page in l0_pages:
        content_preview = str(page.content)[:60]
        print(f"  [{page.page_type.value:10}] {page.page_id}: {content_preview}...")

    # Get VM context for LLM call
    ctx = sm.get_vm_context()
    assert ctx is not None
    print("\nVM context ready for LLM:")
    print(f"  Developer message: {len(ctx['developer_message'])} chars")
    print(f"  Tools: {len(ctx['tools'])} VM tools")
    print(f"  Has manifest: {ctx['manifest'] is not None}")

    # Stats
    stats = vm.get_stats()
    print("\nVM Stats:")
    print(f"  Session: {stats['session_id']}")
    print(f"  Pages in store: {stats['pages_in_store']}")
    print(f"  Turn: {stats['turn']}")
    print(f"  Mode: {stats['mode']}")


async def demo_memory_manager_standalone():
    """MemoryManager standalone — full page lifecycle."""
    print("\n" + "=" * 60)
    print("2. MemoryManager Standalone — Page Lifecycle")
    print("=" * 60)

    vm = MemoryManager(
        session_id="demo-standalone",
        config=WorkingSetConfig(max_l0_tokens=2000),
        mode=VMMode.STRICT,
    )

    # Create pages
    p1 = vm.create_page(
        "User asked about database design for a multi-tenant SaaS app.",
        page_type=PageType.TRANSCRIPT,
        importance=0.6,
        hint="database design multi-tenant",
    )
    p2 = vm.create_page(
        "Recommended PostgreSQL with row-level security and tenant_id column.",
        page_type=PageType.TRANSCRIPT,
        importance=0.5,
        hint="postgresql row-level security",
    )
    p3 = vm.create_page(
        "DECISION: Use PostgreSQL RLS with tenant_id on all tables.",
        page_type=PageType.CLAIM,
        importance=0.9,
        hint="decision: postgres RLS multi-tenant",
    )

    print("\nCreated 3 pages:")
    for p in [p1, p2, p3]:
        print(f"  {p.page_id} [{p.page_type.value}] importance={p.importance}")

    # Add to working set
    for p in [p1, p2, p3]:
        await vm.add_to_working_set(p)

    print(f"\nWorking set: {len(vm.get_l0_pages())} pages")

    # Pin the claim
    vm.pin_page(p3.page_id)
    print(f"Pinned: {p3.page_id}")

    # Advance turn
    vm.new_turn()
    print(f"\nAdvanced to turn {vm.turn}")

    # Evict oldest transcript
    await vm.evict_page(p1.page_id, StorageTier.L2)
    print(f"Evicted {p1.page_id} to L2")

    l0 = vm.get_l0_pages()
    print(f"Working set now: {[p.page_id for p in l0]}")

    # Fault it back in
    result = await vm.handle_fault(p1.page_id)
    print(f"\nPage fault for {p1.page_id}: success={result.success}")
    if result.page:
        print(f"  Loaded from: {result.source_tier}")
        print(f"  Content: {result.page.content[:50]}...")

    l0 = vm.get_l0_pages()
    print(f"Working set after fault: {[p.page_id for p in l0]}")


async def demo_context_building():
    """Build VM context for different modes."""
    print("\n" + "=" * 60)
    print("3. Context Building — Strict vs Passive")
    print("=" * 60)

    vm = MemoryManager(session_id="demo-context", mode=VMMode.STRICT)

    # Add some pages
    pages = [
        vm.create_page("What is quantum computing?", page_type=PageType.TRANSCRIPT),
        vm.create_page(
            "Quantum computing uses qubits that can be in superposition.",
            page_type=PageType.TRANSCRIPT,
        ),
        vm.create_page(
            "FACT: Qubits use superposition and entanglement.",
            page_type=PageType.CLAIM,
            importance=0.9,
        ),
    ]
    for p in pages:
        await vm.add_to_working_set(p)

    # Strict mode — includes VM tools
    ctx_strict = vm.build_context(
        mode=VMMode.STRICT,
        system_prompt="You are a physics tutor.",
    )
    print("\nStrict mode:")
    print(f"  Developer message: {len(ctx_strict['developer_message'])} chars")
    print(f"  VM tools: {len(ctx_strict['tools'])}")
    print(f"  Contains VM:RULES: {'VM:RULES' in ctx_strict['developer_message']}")
    print(f"  Contains VM:CONTEXT: {'VM:CONTEXT' in ctx_strict['developer_message']}")

    # Passive mode — no tools, context only
    ctx_passive = vm.build_context(
        mode=VMMode.PASSIVE,
        system_prompt="You are a physics tutor.",
    )
    print("\nPassive mode:")
    print(f"  Developer message: {len(ctx_passive['developer_message'])} chars")
    print(f"  VM tools: {len(ctx_passive['tools'])} (none in passive)")


async def demo_stats_and_diagnostics():
    """Show comprehensive stats across all subsystems."""
    print("\n" + "=" * 60)
    print("4. Stats & Diagnostics")
    print("=" * 60)

    vm = MemoryManager(session_id="demo-stats")

    # Simulate a conversation
    for i in range(5):
        p = vm.create_page(
            f"Message {i}: discussion about topic {i}",
            page_type=PageType.TRANSCRIPT,
            size_tokens=50,
        )
        await vm.add_to_working_set(p)

    vm.new_turn()

    # Evict some pages
    pages = vm.get_l0_pages()
    if len(pages) >= 2:
        await vm.evict_page(pages[0].page_id, StorageTier.L2)
        await vm.evict_page(pages[1].page_id, StorageTier.L2)

    vm.new_turn()

    # Fault one back
    evicted_id = pages[0].page_id
    await vm.handle_fault(evicted_id)

    # Get stats
    stats = vm.get_stats()
    print(f"\nSession: {stats['session_id']}")
    print(f"Turn: {stats['turn']}")
    print(f"Mode: {stats['mode']}")
    print(f"Pages in store: {stats['pages_in_store']}")

    print("\nPage Table:")
    pt = stats["page_table"]
    print(f"  Total pages: {pt['total_pages']}")
    print(f"  Pages by tier: {dict(pt['pages_by_tier'])}")

    print("\nWorking Set:")
    ws = stats["working_set"]
    print(f"  L0 pages: {ws['l0_pages']}")
    print(f"  Tokens used: {ws['tokens_used']}")

    print("\nFault Handler:")
    fh = stats["fault_handler"]
    print(f"  Total faults: {fh['total_faults']}")
    print(f"  Faults remaining: {fh['faults_remaining']}")

    print("\nMutation Log:")
    ml = stats["mutation_log"]
    print(f"  Total mutations: {ml['total_mutations']}")
    print(
        f"  Creates: {ml['creates']}, Faults: {ml['faults']}, Evictions: {ml['evictions']}"
    )

    print("\nMetrics:")
    m = stats["metrics"]
    print(f"  Faults total: {m['faults_total']}")
    print(f"  Evictions total: {m['evictions_total']}")


async def demo_segmentation_hook():
    """Infinite context + VM: segment rollover creates summary, evicts old pages."""
    print("\n" + "=" * 60)
    print("5. Segmentation Hook — Infinite Context + VM")
    print("=" * 60)

    sm = SessionManager(
        system_prompt="You are a helpful assistant.",
        enable_vm=True,
        infinite_context=True,
        max_turns_per_segment=2,  # Force rollover after 2 turns
        vm_mode=VMMode.STRICT,
    )

    vm = sm.vm
    assert vm is not None

    # Fill first segment (2 turns = 4 messages)
    await sm.user_says("I'm designing a microservices architecture.")
    await sm.ai_responds("Microservices split your app into independent services.")
    await sm.user_says("What about service communication?")
    await sm.ai_responds("Use REST for sync, message queues for async.")

    l0_before = [p.page_id for p in vm.get_l0_pages()]
    print(f"\nBefore segmentation: {len(l0_before)} pages in L0")

    # This triggers segment rollover (3rd turn in a 2-turn segment)
    await sm.user_says("Now tell me about deployment.")

    l0_after = vm.get_l0_pages()
    print(f"After segmentation: {len(l0_after)} pages in L0")

    # Show what's in L0 now
    for page in l0_after:
        pinned = vm.working_set.is_pinned(page.page_id)
        pin_marker = " [PINNED]" if pinned else ""
        content_preview = str(page.content)[:50]
        print(
            f"  [{page.page_type.value:10}] {page.page_id}{pin_marker}: "
            f"{content_preview}..."
        )

    # Summary page should be pinned
    summary_pages = [p for p in l0_after if p.page_type == PageType.SUMMARY]
    print(f"\nSummary pages: {len(summary_pages)}")
    if summary_pages:
        print(f"  Pinned: {vm.working_set.is_pinned(summary_pages[0].page_id)}")

    # Old transcript pages evicted to L2
    pt_stats = vm.page_table.get_stats()
    print(f"\nPage table: {pt_stats.total_pages} total pages")
    print(f"  By tier: {dict(pt_stats.pages_by_tier)}")

    # VM session_id should match new segment
    print(f"\nVM session_id matches SessionManager: {vm.session_id == sm.session_id}")


async def demo_demand_paging():
    """Demand paging pre-pass: recall detection + automatic prefetch."""
    print("\n" + "=" * 60)
    print("6. Demand Paging — Recall Detection + Prefetch")
    print("=" * 60)

    vm = MemoryManager(
        session_id="demo-demand-paging",
        config=WorkingSetConfig(max_l0_tokens=4000),
        mode=VMMode.STRICT,
    )

    # Create pages with hints (simulating a conversation)
    p1 = vm.create_page(
        "User wants to build a REST API with FastAPI.",
        page_type=PageType.TRANSCRIPT,
        hint="fastapi rest api",
    )
    p2 = vm.create_page(
        "DECISION: Use PostgreSQL for the database.",
        page_type=PageType.CLAIM,
        importance=0.9,
        hint="decision: postgresql database",
    )
    p3 = vm.create_page(
        "Discussed Redis for caching layer.",
        page_type=PageType.TRANSCRIPT,
        hint="redis caching",
    )
    p4 = vm.create_page(
        "Summary: API design with FastAPI + PostgreSQL + Redis.",
        page_type=PageType.SUMMARY,
        importance=0.8,
        hint="summary: api design stack",
    )

    # Add all to working set, then evict to simulate old context
    for p in [p1, p2, p3, p4]:
        await vm.add_to_working_set(p)

    vm.new_turn()

    # Evict everything to L2 (simulating context pressure)
    for p in [p1, p2, p3, p4]:
        await vm.evict_page(p.page_id, StorageTier.L2)

    print(f"\nAfter eviction: {len(vm.get_l0_pages())} pages in L0")
    print("All pages evicted to L2 (simulating old conversation)")

    vm.new_turn()

    # Now simulate a user message with recall signal
    message = "What did we decide about the database?"
    print(f'\nUser says: "{message}"')

    # Run demand paging pre-pass
    faulted = await vm.demand_pre_pass(message)
    print(f"Pre-pass faulted {len(faulted)} pages back into L0:")
    for pid in faulted:
        page = vm._page_store.get(pid)
        if page:
            print(f"  [{page.page_type.value:10}] {pid}: {str(page.content)[:50]}...")

    # Show final L0 state
    l0 = vm.get_l0_pages()
    print(f"\nL0 after demand paging: {len(l0)} pages")

    # Try a message WITHOUT recall signal
    vm.new_turn()
    normal_message = "How do I create a new endpoint?"
    print(f'\nUser says: "{normal_message}"')
    faulted2 = await vm.demand_pre_pass(normal_message)
    print(f"Pre-pass faulted {len(faulted2)} pages (no recall signal)")

    # Topic-based match should still work
    vm.new_turn()
    topic_message = "Tell me more about redis configuration"
    print(f'\nUser says: "{topic_message}"')
    faulted3 = await vm.demand_pre_pass(topic_message)
    print(f"Pre-pass faulted {len(faulted3)} pages (topic match on 'redis')")


async def main():
    print("AI Virtual Memory — v0.9 MemoryManager Demo")
    print("No API keys required.\n")

    await demo_session_manager_integration()
    await demo_memory_manager_standalone()
    await demo_context_building()
    await demo_stats_and_diagnostics()
    await demo_segmentation_hook()
    await demo_demand_paging()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
