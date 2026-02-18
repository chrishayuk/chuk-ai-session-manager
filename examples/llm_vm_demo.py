#!/usr/bin/env python3
"""
LLM Virtual Memory Demo (v0.8 — low-level components)

NOTE: For the recommended v0.9 approach using MemoryManager and
SessionManager(enable_vm=True), see examples/06_virtual_memory.py instead.

A practical demonstration of the AI Virtual Memory system with a real LLM.
Uses the low-level VM components directly to show the full page lifecycle.
Simulates a multi-turn conversation where the model must use page faults
to recall information from earlier in the conversation.

Features demonstrated:
- Page type taxonomy (transcript, claim, summary)
- Provenance tracking
- PinnedSet for claims
- FaultPolicy rate limiting
- MutationLog for debugging
- UserExperienceMetrics tracking
- Manifest and context packing

Usage:
    # With OpenAI
    export OPENAI_API_KEY=your-key
    python examples/llm_vm_demo.py

    # With Anthropic
    export ANTHROPIC_API_KEY=your-key
    python examples/llm_vm_demo.py --provider anthropic

    # Dry run (no API calls, shows what would happen)
    python examples/llm_vm_demo.py --dry-run
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from chuk_ai_session_manager.memory import (
    # Enums
    Actor,
    FaultReason,
    Modality,
    MutationType,
    PageType,
    StorageTier,
    VMMode,
    # Models
    FaultPolicy,
    MemoryABI,
    MemoryPage,
    TokenBudget,
    UserExperienceMetrics,
    # Components
    AntiThrashPolicy,
    ContextPacker,
    ManifestBuilder,
    ManifestPolicies,
    MutationLogLite,
    PageFaultHandler,
    PageSearchHandler,
    PageTable,
    PinnedSet,
    SimplePrefetcher,
    WorkingSetConfig,
    WorkingSetManager,
    # Prompts
    build_vm_developer_message,
    get_vm_tools,
)


def load_env():
    """Load environment variables from .env file."""
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent / ".env",
        Path.home() / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key not in os.environ:
                            os.environ[key] = value
            return True
    return False


load_env()


class VirtualMemorySystem:
    """
    Complete Virtual Memory system for managing LLM context.

    Integrates all v0.8 components into a cohesive system.
    """

    def __init__(self, session_id: str = "demo_session"):
        self.session_id = session_id
        self.current_turn = 0

        # Core data structures
        self.page_table = PageTable()
        self.page_store: dict[str, MemoryPage] = {}

        # Working set management
        self.config = WorkingSetConfig(
            max_l0_tokens=4000,  # Small for demo
            max_l1_pages=20,
            eviction_threshold=0.85,
        )
        self.budget = TokenBudget(total_limit=4000, reserved=500)
        self.working_set = WorkingSetManager(config=self.config, budget=self.budget)

        # v0.8 components
        self.pinned_set = PinnedSet(
            auto_pin_last_n_turns=2,
            auto_pin_claims=True,
        )
        self.anti_thrash = AntiThrashPolicy(
            eviction_cooldown_turns=3,
            fault_protection_turns=2,
        )
        self.mutation_log = MutationLogLite(session_id=session_id)
        self.prefetcher = SimplePrefetcher(max_claims_to_prefetch=3)
        self.fault_policy = FaultPolicy(
            max_faults_per_turn=3,
            max_fault_tokens_per_turn=2000,
        )
        self.ux_metrics = UserExperienceMetrics()
        self.memory_abi = MemoryABI(
            max_context_tokens=8000,
            reserved_tokens=1000,
            tool_schema_tokens_reserved=300,
        )

        # Fault/search handlers
        self.fault_handler = PageFaultHandler(max_faults_per_turn=3)
        self.fault_handler.configure(page_table=self.page_table)
        self.search_handler = PageSearchHandler()
        self.search_handler.configure(page_table=self.page_table)

        # Context packing
        self.packer = ContextPacker()
        self.manifest_builder = ManifestBuilder()

        # Track working set page IDs
        self.working_set_ids: list[str] = []

    def new_turn(self):
        """Start a new turn."""
        self.current_turn += 1
        self.fault_policy.new_turn()
        self.mutation_log.record_context_at_turn(
            self.current_turn, self.working_set_ids
        )

    def add_message(
        self,
        role: str,
        content: str,
        extract_claims: bool = True,
    ) -> MemoryPage:
        """Add a user or assistant message as a page."""
        page_id = f"msg_{self.current_turn:03d}_{role}"

        page = MemoryPage(
            page_id=page_id,
            modality=Modality.TEXT,
            content=content,
            storage_tier=StorageTier.L0,
            page_type=PageType.TRANSCRIPT,
            metadata={"role": role, "turn": self.current_turn},
        )

        self._register_page(page)
        self._add_to_working_set(page)

        # Log mutation
        self.mutation_log.record_mutation(
            page_id=page_id,
            mutation_type=MutationType.CREATE,
            tier_after=StorageTier.L0,
            actor=Actor.USER if role == "user" else Actor.MODEL,
            turn=self.current_turn,
        )

        return page

    def add_claim(
        self,
        claim_id: str,
        content: str,
        provenance: list[str],
    ) -> MemoryPage:
        """Add a claim (high-value extracted fact)."""
        page = MemoryPage(
            page_id=claim_id,
            modality=Modality.TEXT,
            content=content,
            storage_tier=StorageTier.L1,  # Claims stay hot
            page_type=PageType.CLAIM,
            provenance=provenance,
            pinned=True,
        )

        self._register_page(page)
        self.pinned_set.pin(claim_id)

        # Log mutation
        self.mutation_log.record_mutation(
            page_id=claim_id,
            mutation_type=MutationType.CREATE,
            tier_after=StorageTier.L1,
            actor=Actor.SYSTEM,
            cause="claim_extraction",
            turn=self.current_turn,
        )

        return page

    def add_summary(
        self,
        summary_id: str,
        content: str,
        represents: list[str],
    ) -> MemoryPage:
        """Add a summary of previous messages."""
        page = MemoryPage(
            page_id=summary_id,
            modality=Modality.TEXT,
            content=content,
            storage_tier=StorageTier.L2,
            page_type=PageType.SUMMARY,
            provenance=represents,
        )

        self._register_page(page)

        # Log mutation
        self.mutation_log.record_mutation(
            page_id=summary_id,
            mutation_type=MutationType.CREATE,
            tier_after=StorageTier.L2,
            actor=Actor.SYSTEM,
            cause="summarization",
            turn=self.current_turn,
        )

        return page

    def _register_page(self, page: MemoryPage):
        """Register a page in the system."""
        self.page_store[page.page_id] = page
        self.page_table.register(page)
        self.fault_handler.store_page(page)

        # Set search hint
        hint = page.content[:100] if page.content else ""
        self.search_handler.set_hint(page.page_id, hint)

    def _add_to_working_set(self, page: MemoryPage):
        """Add a page to the working set."""
        page.size_tokens = page.estimate_tokens()
        self.working_set.add_to_l0(page)
        if page.page_id not in self.working_set_ids:
            self.working_set_ids.append(page.page_id)

    def evict_old_messages(self, keep_last_n: int = 2):
        """Evict old transcript pages to free up space."""
        # Get transcript pages sorted by turn
        transcripts = [
            (pid, self.page_store[pid])
            for pid in self.working_set_ids
            if pid in self.page_store
            and self.page_store[pid].page_type == PageType.TRANSCRIPT
        ]

        # Sort by turn (from metadata)
        transcripts.sort(
            key=lambda x: x[1].metadata.get("turn", 0) if x[1].metadata else 0
        )

        # Evict all but the last N
        to_evict = transcripts[:-keep_last_n] if len(transcripts) > keep_last_n else []

        for page_id, page in to_evict:
            if self.pinned_set.is_pinned(page_id):
                continue
            if not self.anti_thrash.can_evict(page_id, self.current_turn):
                continue

            # Evict
            self.working_set_ids.remove(page_id)
            self.page_table.update_location(page_id, StorageTier.L2)
            self.anti_thrash.record_eviction(page_id, self.current_turn)

            self.mutation_log.record_mutation(
                page_id=page_id,
                mutation_type=MutationType.EVICT,
                tier_before=StorageTier.L0,
                tier_after=StorageTier.L2,
                actor=Actor.SYSTEM,
                cause="context_pressure",
                turn=self.current_turn,
            )

    async def handle_page_fault(
        self,
        page_id: str,
        target_level: int = 0,
        reason: str = "resolve_reference",
    ) -> dict:
        """Handle a page fault request from the model."""
        # Check fault policy
        page = self.page_store.get(page_id)
        if not page:
            return {"error": f"Page '{page_id}' not found"}

        tokens_est = page.estimate_tokens()
        if not self.fault_policy.can_fault(tokens_est):
            return {"error": "Fault limit reached for this turn"}

        if page_id in self.working_set_ids:
            return {"error": "Page already in working set"}

        # Map reason to enum
        fault_reason = {
            "user_requested": FaultReason.USER_REQUESTED_RECALL,
            "resolve_reference": FaultReason.RESOLVE_REFERENCE,
            "tool_prereq": FaultReason.TOOL_PREREQUISITE,
            "speculative": FaultReason.SPECULATIVE,
        }.get(reason, FaultReason.RESOLVE_REFERENCE)

        # Record the fault
        self.fault_policy.record_fault(tokens_est)
        self.anti_thrash.record_fault(page_id, self.current_turn)
        self.prefetcher.record_page_access(page_id)

        # Update UX metrics
        self.ux_metrics.record_fault(
            page_id=page_id,
            reason=fault_reason,
            turn=self.current_turn,
            tokens_loaded=tokens_est,
        )

        # Log mutation
        self.mutation_log.record_mutation(
            page_id=page_id,
            mutation_type=MutationType.FAULT_IN,
            tier_before=StorageTier.L2,
            tier_after=StorageTier.L0,
            actor=Actor.MODEL,
            cause=reason,
            turn=self.current_turn,
        )

        # Add to working set
        self._add_to_working_set(page)
        self.page_table.update_location(page_id, StorageTier.L0)

        return {
            "page": {
                "page_id": page_id,
                "modality": page.modality.value,
                "page_type": page.page_type.value if page.page_type else "unknown",
                "content": {"text": page.content},
                "provenance": page.provenance,
            },
            "effects": {
                "promoted_to_working_set": True,
                "tokens_est": tokens_est,
                "fault_reason": fault_reason.value,
            },
        }

    async def handle_search(
        self,
        query: str,
        page_type: Optional[str] = None,
        limit: int = 5,
    ) -> dict:
        """Handle a search request from the model."""
        result = await self.search_handler.search(query, limit=limit)

        # Filter by page type if specified
        results = []
        for r in result.results:
            page = self.page_store.get(r.page_id)
            if page and page_type:
                if page.page_type and page.page_type.value != page_type:
                    continue
            results.append(
                {
                    "page_id": r.page_id,
                    "modality": r.modality,
                    "page_type": page.page_type.value
                    if page and page.page_type
                    else "unknown",
                    "tier": r.tier,
                    "hint": r.hint,
                    "relevance": r.relevance,
                }
            )

        return {
            "results": results[:limit],
            "total_available": result.total_available,
        }

    def build_context(self) -> tuple[str, str]:
        """Build VM:CONTEXT and VM:MANIFEST_JSON for the model."""
        # Get pages in working set
        pages = [
            self.page_store[pid]
            for pid in self.working_set_ids
            if pid in self.page_store
        ]

        # Pack context
        packed = self.packer.pack(pages)

        # Build manifest
        policies = ManifestPolicies(
            max_faults_per_turn=self.fault_policy.max_faults_per_turn
            - self.fault_policy.faults_this_turn,
            upgrade_budget_tokens=self.fault_policy.max_fault_tokens_per_turn
            - self.fault_policy.tokens_used_this_turn,
        )

        manifest = self.manifest_builder.build(
            session_id=self.session_id,
            page_table=self.page_table,
            working_set_ids=self.working_set_ids,
            policies=policies,
            hint_generator=lambda e: self._generate_hint(e.page_id),
        )

        return packed.content, manifest.to_json()

    def _generate_hint(self, page_id: str) -> str:
        """Generate a hint for a page."""
        page = self.page_store.get(page_id)
        if not page:
            return ""

        prefix = f"[{page.page_type.value}]" if page.page_type else ""
        content = (
            page.content[:60] + "..."
            if page.content and len(page.content) > 60
            else (page.content or "")
        )
        return f"{prefix} {content}"

    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "session_id": self.session_id,
            "current_turn": self.current_turn,
            "pages_total": len(self.page_store),
            "working_set_size": len(self.working_set_ids),
            "pinned_count": self.pinned_set.count(),
            "fault_policy": {
                "faults_this_turn": self.fault_policy.faults_this_turn,
                "tokens_this_turn": self.fault_policy.tokens_used_this_turn,
            },
            "ux_metrics": {
                "recall_success_rate": self.ux_metrics.recall_success_rate(),
                "thrash_index": self.ux_metrics.thrash_index(),
            },
            "mutation_log": self.mutation_log.get_summary(),
        }

    def print_state(self):
        """Print current system state."""
        print("\n" + "=" * 60)
        print(f"VM STATE - Turn {self.current_turn}")
        print("=" * 60)

        print(f"\nWorking Set ({len(self.working_set_ids)} pages):")
        for pid in self.working_set_ids:
            page = self.page_store.get(pid)
            if page:
                pinned = "📌" if self.pinned_set.is_pinned(pid) else "  "
                ptype = page.page_type.value if page.page_type else "?"
                content = (
                    (page.content[:40] + "...")
                    if page.content and len(page.content) > 40
                    else page.content
                )
                print(f"  {pinned} [{ptype:10}] {pid}: {content}")

        print("\nAvailable (not in working set):")
        for pid, page in self.page_store.items():
            if pid not in self.working_set_ids:
                ptype = page.page_type.value if page.page_type else "?"
                tier = (
                    self.page_table.lookup(pid).tier.value
                    if self.page_table.lookup(pid)
                    else "?"
                )
                content = (
                    (page.content[:40] + "...")
                    if page.content and len(page.content) > 40
                    else page.content
                )
                print(f"     [{ptype:10}] {pid} ({tier}): {content}")

        print(
            f"\nFault Budget: {self.fault_policy.max_faults_per_turn - self.fault_policy.faults_this_turn} remaining"
        )


# =============================================================================
# Demo Scenarios
# =============================================================================


async def run_demo(provider: str, model: str, dry_run: bool = False):
    """Run the VM demo with simulated conversation."""
    print("\n" + "=" * 60)
    print("AI VIRTUAL MEMORY - LLM DEMO (v0.8)")
    print("=" * 60)

    vm = VirtualMemorySystem(session_id="llm_demo")

    # ==========================================================================
    # PHASE 1: Build up conversation history
    # ==========================================================================
    print("\n>>> PHASE 1: Building conversation history")

    # Turn 1: User asks about project
    vm.new_turn()
    vm.add_message(
        "user",
        "I'm working on a Python web API. I want to use FastAPI with PostgreSQL.",
    )
    vm.add_message(
        "assistant",
        "Great choice! FastAPI is excellent for building APIs. For PostgreSQL, I recommend using SQLAlchemy as your ORM with async support via asyncpg.",
    )

    # Extract claim
    vm.add_claim(
        "claim_stack",
        "DECISION: Tech stack is FastAPI + PostgreSQL + SQLAlchemy (async)",
        provenance=["msg_001_user", "msg_001_assistant"],
    )

    # Turn 2: Authentication discussion
    vm.new_turn()
    vm.add_message(
        "user",
        "How should I handle authentication? I need both API keys for services and JWT for users.",
    )
    vm.add_message(
        "assistant",
        "For dual auth, use FastAPI's dependency injection. API keys via header check, JWTs via OAuth2PasswordBearer. Store refresh tokens in HTTP-only cookies, rotate on use.",
    )

    # Extract claims
    vm.add_claim(
        "claim_auth_approach",
        "DECISION: Dual auth - API keys (header) + JWT (OAuth2PasswordBearer)",
        provenance=["msg_002_user", "msg_002_assistant"],
    )
    vm.add_claim(
        "claim_refresh_tokens",
        "DECISION: Refresh tokens in HTTP-only cookies, rotate on each use",
        provenance=["msg_002_assistant"],
    )

    # Turn 3: Rate limiting
    vm.new_turn()
    vm.add_message("user", "What about rate limiting?")
    vm.add_message(
        "assistant",
        "Use slowapi for FastAPI rate limiting. Set 100 req/min for authenticated users, 20 req/min for API keys. Store counts in Redis for distributed limiting.",
    )

    vm.add_claim(
        "claim_rate_limits",
        "DECISION: Rate limits - 100/min (users), 20/min (API keys), Redis backend",
        provenance=["msg_003_assistant"],
    )

    # Turn 4: Database schema
    vm.new_turn()
    vm.add_message(
        "user",
        "Let's discuss the database schema. I need users, organizations, and projects.",
    )
    vm.add_message(
        "assistant",
        "Use a multi-tenant schema: users belong to organizations, projects belong to organizations. Add a user_organizations junction table for many-to-many. Use UUIDs for all primary keys.",
    )

    vm.add_claim(
        "claim_schema",
        "DECISION: Multi-tenant schema with users, organizations, projects. UUIDs for PKs.",
        provenance=["msg_004_assistant"],
    )

    # Create a summary of early turns
    vm.add_summary(
        "summary_turns_1_2",
        "Early discussion covered: 1) Tech stack (FastAPI+PostgreSQL+SQLAlchemy), 2) Dual authentication (API keys + JWT), 3) Refresh token handling (HTTP-only cookies)",
        represents=[
            "msg_001_user",
            "msg_001_assistant",
            "msg_002_user",
            "msg_002_assistant",
        ],
    )

    # Turn 5: More discussion
    vm.new_turn()
    vm.add_message("user", "Should I use Alembic for migrations?")
    vm.add_message(
        "assistant",
        "Yes, Alembic is the standard for SQLAlchemy migrations. Set up with --async flag. Use revision --autogenerate for schema changes. Keep migrations in version control.",
    )

    # Now simulate context pressure - evict old transcript pages
    print("\n>>> Simulating context pressure - evicting old transcripts...")
    vm.evict_old_messages(keep_last_n=2)

    vm.print_state()

    # ==========================================================================
    # PHASE 2: User asks about something from earlier
    # ==========================================================================
    print("\n>>> PHASE 2: User asks about earlier decision (requires recall)")

    vm.new_turn()
    user_query = "Wait, what did we decide about the rate limits again? And remind me of the auth approach."

    vm.add_message("user", user_query)

    # Build context for model
    context_content, manifest_json = vm.build_context()

    print("\n--- VM:MANIFEST_JSON (available pages) ---")
    manifest = json.loads(manifest_json)
    print(f"Working set: {[p['page_id'] for p in manifest['working_set']]}")
    print(f"Available pages: {[p['page_id'] for p in manifest['available_pages']]}")
    print(f"Faults remaining: {manifest['policies']['max_faults_per_turn']}")

    if dry_run:
        print("\n--- DRY RUN: Simulating model behavior ---")
        print("Model would search for 'rate limit' and 'auth'...")
        print("Model would fault in claim_rate_limits and claim_auth_approach...")

        # Simulate the faults
        result1 = await vm.handle_page_fault(
            "claim_rate_limits", reason="user_requested"
        )
        print(f"\nFault 1: {result1.get('page', {}).get('page_id', 'error')}")

        result2 = await vm.handle_page_fault(
            "claim_auth_approach", reason="user_requested"
        )
        print(f"Fault 2: {result2.get('page', {}).get('page_id', 'error')}")

        # Simulated response
        response = """Based on our earlier discussion [ref: claim_rate_limits], we decided on:
- **Rate Limits**: 100 requests/minute for authenticated users, 20 requests/minute for API keys, using Redis as the backend

And for authentication [ref: claim_auth_approach]:
- **Dual Auth**: API keys checked via headers, JWTs via OAuth2PasswordBearer
- Refresh tokens stored in HTTP-only cookies and rotated on each use [ref: claim_refresh_tokens]"""

        print(f"\n--- Simulated Model Response ---\n{response}")

    else:
        # Actually call the LLM
        response = await call_llm(
            provider=provider,
            model=model,
            vm=vm,
            user_query=user_query,
            context_content=context_content,
            manifest_json=manifest_json,
        )
        print(f"\n--- Model Response ---\n{response}")

    vm.print_state()

    # ==========================================================================
    # PHASE 3: Show final stats
    # ==========================================================================
    print("\n>>> PHASE 3: Final Statistics")
    print("\n" + "=" * 60)
    print("VM SYSTEM STATS")
    print("=" * 60)

    stats = vm.get_stats()
    print(f"\nSession: {stats['session_id']}")
    print(f"Total turns: {stats['current_turn']}")
    print(f"Total pages: {stats['pages_total']}")
    print(f"Working set size: {stats['working_set_size']}")
    print(f"Pinned pages: {stats['pinned_count']}")

    print("\nFault activity this turn:")
    print(f"  Faults: {stats['fault_policy']['faults_this_turn']}")
    print(f"  Tokens loaded: {stats['fault_policy']['tokens_this_turn']}")

    print("\nUX Metrics:")
    print(f"  Recall success rate: {stats['ux_metrics']['recall_success_rate']:.2%}")
    print(f"  Thrash index: {stats['ux_metrics']['thrash_index']:.2f}")

    print("\nMutation Log:")
    ml = stats["mutation_log"]
    print(f"  Total mutations: {ml['total_mutations']}")
    print(
        f"  Creates: {ml['creates']}, Faults: {ml['faults']}, Evictions: {ml['evictions']}"
    )

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


async def call_llm(
    provider: str,
    model: str,
    vm: VirtualMemorySystem,
    user_query: str,
    context_content: str,
    manifest_json: str,
) -> str:
    """Call the LLM with VM context and handle tool calls."""

    # Build developer message
    dev_message = build_vm_developer_message(
        mode=VMMode.STRICT,
        manifest_json=manifest_json,
        context=context_content,
        system_prompt="You are a helpful AI assistant helping with a software project. Use the VM tools to recall information not in your current context.",
        max_faults_per_turn=3,
    )

    # Get VM tools
    tools = get_vm_tools(include_search=True)

    if provider == "openai":
        return await _call_openai(model, dev_message, user_query, tools, vm)
    elif provider == "anthropic":
        return await _call_anthropic(model, dev_message, user_query, tools, vm)
    else:
        return f"Unknown provider: {provider}"


async def _call_openai(
    model: str,
    dev_message: str,
    user_query: str,
    tools: list,
    vm: VirtualMemorySystem,
) -> str:
    """Call OpenAI with VM tools."""
    try:
        from openai import OpenAI
    except ImportError:
        return "OpenAI not installed. Run: pip install openai"

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY not set"

    client = OpenAI(api_key=api_key)

    # Convert tools to OpenAI format
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            },
        }
        for t in tools
    ]

    messages = [
        {"role": "developer", "content": dev_message},
        {"role": "user", "content": user_query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=openai_tools,
    )

    message = response.choices[0].message
    iterations = 0

    while message.tool_calls and iterations < 5:
        iterations += 1
        print(f"\n--- Model requesting {len(message.tool_calls)} tool call(s) ---")

        # Process tool calls
        tool_results = []
        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  → {tc.function.name}({args})")

            if tc.function.name == "page_fault":
                result = await vm.handle_page_fault(
                    args["page_id"],
                    args.get("target_level", 0),
                    args.get("reason", "resolve_reference"),
                )
            elif tc.function.name == "search_pages":
                result = await vm.handle_search(
                    args["query"],
                    args.get("page_type"),
                    args.get("limit", 5),
                )
            else:
                result = {"error": f"Unknown tool: {tc.function.name}"}

            if "page" in result:
                print(f"  ← Loaded {result['page']['page_id']}")
            elif "results" in result:
                print(f"  ← Found {len(result['results'])} results")

            tool_results.append(
                {
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

        # Add to messages
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        for result in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"],
                }
            )

        # Get next response
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
        )
        message = response.choices[0].message

    return message.content or "(no response)"


async def _call_anthropic(
    model: str,
    dev_message: str,
    user_query: str,
    tools: list,
    vm: VirtualMemorySystem,
) -> str:
    """Call Anthropic with VM tools."""
    try:
        import anthropic
    except ImportError:
        return "Anthropic not installed. Run: pip install anthropic"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "ANTHROPIC_API_KEY not set"

    client = anthropic.Anthropic(api_key=api_key)

    # Convert tools to Anthropic format
    anthropic_tools = [
        {
            "name": t.function.name,
            "description": t.function.description,
            "input_schema": t.function.parameters,
        }
        for t in tools
    ]

    messages = [{"role": "user", "content": user_query}]

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=dev_message,
        tools=anthropic_tools,
        messages=messages,
    )

    iterations = 0

    while response.stop_reason == "tool_use" and iterations < 5:
        iterations += 1
        print("\n--- Model requesting tool calls ---")

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            print(f"  → {tu.name}({tu.input})")

            if tu.name == "page_fault":
                result = await vm.handle_page_fault(
                    tu.input["page_id"],
                    tu.input.get("target_level", 0),
                    tu.input.get("reason", "resolve_reference"),
                )
            elif tu.name == "search_pages":
                result = await vm.handle_search(
                    tu.input["query"],
                    tu.input.get("page_type"),
                    tu.input.get("limit", 5),
                )
            else:
                result = {"error": f"Unknown tool: {tu.name}"}

            if "page" in result:
                print(f"  ← Loaded {result['page']['page_id']}")

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result),
                }
            )

        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=dev_message,
            tools=anthropic_tools,
            messages=messages,
        )

    # Extract text
    return "".join(b.text for b in response.content if hasattr(b, "text"))


def main():
    parser = argparse.ArgumentParser(description="LLM Virtual Memory Demo (v0.8)")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use (default: gpt-4o or claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without API calls (simulates model behavior)",
    )

    args = parser.parse_args()

    if args.model is None:
        args.model = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
        }[args.provider]

    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Dry run: {args.dry_run}")

    asyncio.run(run_demo(args.provider, args.model, args.dry_run))


if __name__ == "__main__":
    main()
