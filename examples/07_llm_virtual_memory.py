#!/usr/bin/env python3
"""
LLM Virtual Memory Demo (v0.11)

Comprehensive demonstration of the MemoryManager orchestrator with a real
OpenAI LLM.  The model uses VM tools (page_fault, search_pages) to recall
evicted context across eight demo phases.

Features shown:
- MemoryManager with ImportanceWeightedLRU + CompressorRegistry
- TRANSCRIPT, CLAIM, and SUMMARY page types with provenance tracking
- Full conversation -> eviction -> model recall via page_fault AND search_pages
- Demand paging pre-pass for proactive recall
- Search-driven discovery (search_pages -> page_fault chain)
- Compress-before-evict and compression pipeline
- VM:CONTEXT, VM:MANIFEST_JSON, and VM:RULES in developer message
- --dry-run mode to see the VM flow without API calls

Usage:
    # Set key in .env or environment
    echo "OPENAI_API_KEY=your-key" > .env
    python examples/07_llm_virtual_memory.py

    # Dry run (no API calls, shows VM flow only)
    python examples/07_llm_virtual_memory.py --dry-run
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

from chuk_ai_session_manager.memory.compressor import CompressorRegistry
from chuk_ai_session_manager.memory.eviction_policy import ImportanceWeightedLRU
from chuk_ai_session_manager.memory.fault_handler import VMToolError
from chuk_ai_session_manager.memory.manager import MemoryManager
from chuk_ai_session_manager.memory.models import (
    CompressionLevel,
    PageType,
    StorageTier,
    VMMode,
)
from chuk_ai_session_manager.memory.working_set import WorkingSetConfig

DEFAULT_MODEL = "gpt-4o"
MAX_TOOL_ITERATIONS = 5


def _load_dotenv() -> None:
    """Load variables from .env file if present."""
    for candidate in (
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ):
        if candidate.is_file():
            with open(candidate) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    if key not in os.environ:
                        os.environ[key] = value
            return


_load_dotenv()

SYSTEM_PROMPT = (
    "You are an AI assistant helping design a REST API. "
    "Use the VM tools to recall earlier conversation context when needed."
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_vm() -> MemoryManager:
    """Create a MemoryManager with eviction policy and compression."""
    return MemoryManager(
        session_id="llm-vm-demo",
        config=WorkingSetConfig(max_l0_tokens=5000),
        mode=VMMode.STRICT,
        eviction_policy=ImportanceWeightedLRU(),
        compressor_registry=CompressorRegistry.default(),
    )


# ---------------------------------------------------------------------------
# Phase 1: Build conversation history  (7 turns, 17 pages)
# ---------------------------------------------------------------------------

# Conversation data: (page_id, page_type, importance, hint, content)
CONVERSATION = [
    # Turn 1 -- Tech stack
    (
        "turn1_user",
        PageType.TRANSCRIPT,
        0.5,
        "python web api fastapi",
        "I'm building a Python web API. I want to use FastAPI with PostgreSQL. "
        "What ORM should I use?",
    ),
    (
        "turn1_asst",
        PageType.TRANSCRIPT,
        0.5,
        "fastapi sqlalchemy postgresql asyncpg",
        "Great choice! FastAPI is excellent for APIs. For PostgreSQL, I recommend "
        "SQLAlchemy with async support via asyncpg. It gives you full ORM "
        "capabilities with native async.",
    ),
    (
        "claim_stack",
        PageType.CLAIM,
        0.9,
        "decision: tech stack fastapi postgresql sqlalchemy",
        "DECISION: Tech stack is FastAPI + PostgreSQL + SQLAlchemy (asyncpg driver)",
    ),
    # Turn 2 -- Authentication
    (
        "turn2_user",
        PageType.TRANSCRIPT,
        0.5,
        "authentication api keys jwt",
        "How should I handle authentication? I need both API keys for external "
        "services and JWT tokens for end users.",
    ),
    (
        "turn2_asst",
        PageType.TRANSCRIPT,
        0.5,
        "dual auth oauth2 api keys jwt cookies",
        "For dual auth, use FastAPI's dependency injection. API keys via "
        "X-API-Key header check, JWTs via OAuth2PasswordBearer. Store refresh "
        "tokens in HTTP-only cookies and rotate them on each use.",
    ),
    (
        "claim_auth",
        PageType.CLAIM,
        0.9,
        "decision: dual auth api keys jwt oauth2",
        "DECISION: Dual authentication -- API keys via header for services, "
        "JWT via OAuth2PasswordBearer for users. Refresh tokens in HTTP-only "
        "cookies, rotated on use.",
    ),
    # Turn 3 -- Rate limiting
    (
        "turn3_user",
        PageType.TRANSCRIPT,
        0.5,
        "rate limiting api",
        "What about rate limiting for the API?",
    ),
    (
        "turn3_asst",
        PageType.TRANSCRIPT,
        0.5,
        "slowapi redis rate limits 100 20",
        "Use slowapi for FastAPI rate limiting. Set 100 requests/minute for "
        "authenticated users, 20 requests/minute for API key access. Use Redis "
        "as the backend for distributed rate limit counters.",
    ),
    (
        "claim_rates",
        PageType.CLAIM,
        0.9,
        "decision: rate limits 100/min users 20/min api keys redis",
        "DECISION: Rate limits -- 100 req/min for authenticated users, "
        "20 req/min for API keys. Backend: Redis via slowapi.",
    ),
    # Turn 4 -- Database schema
    (
        "turn4_user",
        PageType.TRANSCRIPT,
        0.5,
        "database schema users organizations projects",
        "Let's design the database schema. I need users, organizations, and "
        "projects with proper relationships.",
    ),
    (
        "turn4_asst",
        PageType.TRANSCRIPT,
        0.5,
        "multi-tenant uuid postgresql rls organizations",
        "Use a multi-tenant schema: users belong to organizations via a "
        "junction table (many-to-many), projects belong to organizations "
        "(one-to-many). Use UUIDs for all primary keys. Add row-level "
        "security with tenant_id on all tables.",
    ),
    # Turn 5 -- Error handling
    (
        "turn5_user",
        PageType.TRANSCRIPT,
        0.5,
        "error handling http status codes validation",
        "How should we handle errors? I want consistent error responses "
        "across all endpoints.",
    ),
    (
        "turn5_asst",
        PageType.TRANSCRIPT,
        0.5,
        "rfc7807 problem details validation pydantic 422 500 correlation",
        "Use RFC 7807 Problem Details format for all errors. Return a JSON "
        "body with type, title, status, detail, and instance fields. "
        "Map Pydantic validation errors to 422 with field-level details. "
        "Use 500 with a correlation ID for unexpected errors, logged to "
        "structured logging with the same correlation ID for tracing.",
    ),
    (
        "claim_errors",
        PageType.CLAIM,
        0.9,
        "decision: rfc7807 problem details error format correlation ids",
        "DECISION: Error format is RFC 7807 Problem Details. Pydantic "
        "validation -> 422. Unexpected errors -> 500 with correlation ID "
        "for log tracing.",
    ),
    # Turn 6 -- Deployment (NO claim -- forces search_pages later)
    (
        "turn6_user",
        PageType.TRANSCRIPT,
        0.5,
        "deployment docker kubernetes ci cd",
        "What's the deployment plan? We need containerization and CI/CD.",
    ),
    (
        "turn6_asst",
        PageType.TRANSCRIPT,
        0.5,
        "docker multi-stage alpine kubernetes helm github actions ecr",
        "Use multi-stage Docker builds with python:3.12-slim as the final "
        "stage. Deploy to Kubernetes via Helm charts. CI/CD with GitHub "
        "Actions: lint, test, build image, push to ECR, deploy to staging "
        "on PR merge, production on release tag. Include database migrations "
        "in a Kubernetes Job that runs before the main deployment.",
    ),
    # Turn 7 -- Monitoring
    (
        "turn7_user",
        PageType.TRANSCRIPT,
        0.5,
        "monitoring observability metrics logging alerting",
        "What monitoring and observability setup should we use?",
    ),
    (
        "turn7_asst",
        PageType.TRANSCRIPT,
        0.5,
        "opentelemetry prometheus grafana structured logging elk pagerduty correlation",
        "Use OpenTelemetry for distributed tracing and metrics collection. "
        "Export metrics to Prometheus, visualize in Grafana dashboards. "
        "Structured JSON logging with correlation IDs (matching the error "
        "handling correlation IDs). Ship logs to ELK stack. Set up PagerDuty "
        "alerts for P99 latency > 500ms and error rate > 5%.",
    ),
    (
        "claim_monitoring",
        PageType.CLAIM,
        0.9,
        "decision: opentelemetry prometheus grafana elk pagerduty alerts",
        "DECISION: Observability stack is OpenTelemetry + Prometheus + Grafana "
        "for metrics, ELK for logs. PagerDuty alerts on P99 > 500ms and "
        "error rate > 5%.",
    ),
]


async def build_conversation(vm: MemoryManager) -> None:
    """Populate the VM with a simulated API design conversation."""
    turn = 0
    for page_id, page_type, importance, hint, content in CONVERSATION:
        page = vm.create_page(
            content,
            page_type=page_type,
            importance=importance,
            hint=hint,
            page_id=page_id,
        )
        await vm.add_to_working_set(page)

        # Pin claims
        if page_type == PageType.CLAIM:
            vm.pin_page(page_id)

        # Advance turn after each assistant response
        if page_id.endswith("_asst"):
            vm.new_turn()
            turn += 1


# ---------------------------------------------------------------------------
# Phase 2: Create summary pages with provenance
# ---------------------------------------------------------------------------

# (page_id, importance, hint, content, provenance_page_ids)
SUMMARIES = [
    (
        "summary_turns_1_3",
        0.8,
        "summary: tech stack auth rate limits fastapi postgresql jwt",
        "SUMMARY of turns 1-3: Tech stack decided as FastAPI + PostgreSQL "
        "via SQLAlchemy/asyncpg. Dual auth: API keys via X-API-Key header, "
        "JWTs via OAuth2PasswordBearer with refresh tokens in HTTP-only "
        "cookies. Rate limits: 100 req/min users, 20 req/min API keys "
        "via slowapi + Redis.",
        [
            "turn1_user",
            "turn1_asst",
            "turn2_user",
            "turn2_asst",
            "turn3_user",
            "turn3_asst",
        ],
    ),
    (
        "summary_turns_4_5",
        0.8,
        "summary: database schema multi-tenant error handling rfc7807",
        "SUMMARY of turns 4-5: Database uses multi-tenant schema with "
        "users, organizations (many-to-many via junction), and projects "
        "(one-to-many). UUIDs for PKs, row-level security. Errors use "
        "RFC 7807 Problem Details format with correlation IDs for tracing.",
        [
            "turn4_user",
            "turn4_asst",
            "turn5_user",
            "turn5_asst",
        ],
    ),
]


async def build_summaries(vm: MemoryManager) -> None:
    """Create summary pages with provenance tracking."""
    for page_id, importance, hint, content, provenance in SUMMARIES:
        page = vm.create_page(
            content,
            page_type=PageType.SUMMARY,
            importance=importance,
            hint=hint,
            page_id=page_id,
            provenance=provenance,
        )
        await vm.add_to_working_set(page)
        vm.pin_page(page_id)


# ---------------------------------------------------------------------------
# Phase 3: Ask with full context  (API call 1)
# ---------------------------------------------------------------------------


async def ask_with_full_context(vm: MemoryManager, client: Any, model: str) -> str:
    """LLM call -- model sees full context and summarizes."""
    vm.new_turn()
    ctx = vm.build_context(system_prompt=SYSTEM_PROMPT)

    messages = [
        {"role": "developer", "content": ctx["developer_message"]},
        {
            "role": "user",
            "content": "Summarize the key technical decisions we've made so far.",
        },
    ]

    response = await client.chat.completions.create(
        model=model, messages=messages, tools=ctx["tools"]
    )
    return response.choices[0].message.content or "(no response)"


# ---------------------------------------------------------------------------
# Phase 4: Evict old context
# ---------------------------------------------------------------------------


async def evict_old_context(vm: MemoryManager) -> list:
    """Evict non-pinned transcript pages to L2."""
    evicted = await vm.evict_segment_pages(target_tier=StorageTier.L2)
    vm.new_turn()
    return evicted


# ---------------------------------------------------------------------------
# Phase 5: Search-driven recall  (API call 2)
# ---------------------------------------------------------------------------


async def ask_with_search(vm: MemoryManager, client: Any, model: str) -> str:
    """
    LLM call -- model must use search_pages to discover deployment pages.

    Turn 6 (deployment/CI/CD) has no claim and is not covered by any
    summary, so the model must search to find the right page_ids.
    No demand_pre_pass is called -- let the model discover via tools.
    """
    vm.new_turn()
    ctx = vm.build_context(system_prompt=SYSTEM_PROMPT)

    user_question = (
        "What CI/CD pipeline and deployment setup did we plan? "
        "Include the specific steps and tools."
    )

    messages: list[dict[str, Any]] = [
        {"role": "developer", "content": ctx["developer_message"]},
        {"role": "user", "content": user_question},
    ]

    return await _run_tool_loop(vm, client, model, ctx, messages)


# ---------------------------------------------------------------------------
# Phase 6: Demand paging + cross-topic recall  (API call 3)
# ---------------------------------------------------------------------------


async def ask_with_recall(vm: MemoryManager, client: Any, model: str) -> str:
    """
    LLM call -- demand paging pre-faults relevant pages, model may
    still need additional page_fault calls for cross-topic details.
    """
    vm.new_turn()

    user_question = (
        "What are the exact monitoring alert thresholds we decided on, "
        "and how do the correlation IDs from error handling connect to "
        "the observability setup?"
    )
    faulted = await vm.demand_pre_pass(user_question)
    if faulted:
        print(f"  Demand paging pre-faulted {len(faulted)} pages: {faulted}")

    ctx = vm.build_context(system_prompt=SYSTEM_PROMPT)

    messages: list[dict[str, Any]] = [
        {"role": "developer", "content": ctx["developer_message"]},
        {"role": "user", "content": user_question},
    ]

    return await _run_tool_loop(vm, client, model, ctx, messages)


# ---------------------------------------------------------------------------
# Shared tool-call loop
# ---------------------------------------------------------------------------


async def _run_tool_loop(
    vm: MemoryManager,
    client: Any,
    model: str,
    ctx: dict,
    messages: list[dict[str, Any]],
) -> str:
    """Run the model in a tool-call loop until it produces a final answer."""
    for iteration in range(MAX_TOOL_ITERATIONS):
        response = await client.chat.completions.create(
            model=model, messages=messages, tools=ctx["tools"]
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or "(no response)"

        print(f"  Model requesting {len(message.tool_calls)} tool call(s):")

        # Append assistant message with tool_calls
        messages.append(_serialize_assistant_message(message))

        for tc in message.tool_calls:
            result_json = await _handle_tool_call(vm, tc)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result_json}
            )

    return message.content or "(max tool iterations reached)"


async def _handle_tool_call(vm: MemoryManager, tool_call: Any) -> str:
    """Dispatch a VM tool call to MemoryManager and return JSON."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if name == "page_fault":
        page_id = args["page_id"]
        target_level = args.get("target_level", 2)
        print(f"    -> page_fault({page_id!r}, level={target_level})")

        result = await vm.handle_fault(page_id, target_level)

        if result.success and result.page:
            tokens = result.page.size_tokens or result.page.estimate_tokens()
            print(f"    <- Loaded from {result.source_tier}, {tokens} tokens")
            tool_result = vm._fault_handler.build_tool_result(result)
            return tool_result.to_json()

        error = VMToolError(error=result.error or "Page not found", page_id=page_id)
        return error.model_dump_json()

    if name == "search_pages":
        query = args["query"]
        limit = args.get("limit", 5)
        print(f"    -> search_pages({query!r}, limit={limit})")

        search_result = await vm.search_pages(query, limit=limit)
        print(f"    <- Found {len(search_result.results)} results")
        return search_result.to_json()

    return json.dumps({"error": f"Unknown tool: {name}"})


def _serialize_assistant_message(message: Any) -> dict:
    """Convert OpenAI assistant message with tool_calls to a dict."""
    return {
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


# ---------------------------------------------------------------------------
# Phase 7: Compression demo
# ---------------------------------------------------------------------------


async def show_compression(vm: MemoryManager) -> None:
    """Demonstrate compression through the levels on a transcript page."""
    # Find the longest transcript page for a visible compression demo
    best_id: Optional[str] = None
    best_tokens = 0
    for pid, page in vm._page_store.items():
        if (
            page.page_type == PageType.TRANSCRIPT
            and page.compression_level == CompressionLevel.FULL
        ):
            tokens = page.size_tokens or page.estimate_tokens()
            if tokens > best_tokens:
                best_id = pid
                best_tokens = tokens

    if not best_id:
        print("  No transcript pages available for compression.")
        return

    page = vm._page_store[best_id]
    content_preview = str(page.content)[:60]
    print(f"  Page: {best_id}")
    print(f"    FULL:      {best_tokens:3d} tokens -- {content_preview!r}...")

    total_saved = 0
    for level in [
        CompressionLevel.REDUCED,
        CompressionLevel.ABSTRACT,
        CompressionLevel.REFERENCE,
    ]:
        result = await vm.compress_page(best_id, level)
        if result and result.tokens_saved > 0:
            compressed_content = str(vm._page_store[best_id].content)[:50]
            print(
                f"    {level.name:10s} {result.compressed_tokens:3d} tokens "
                f"(saved {result.tokens_saved}) -- {compressed_content!r}..."
            )
            total_saved += result.tokens_saved

    final_level = vm._page_store[best_id].compression_level
    print(f"    Total saved: {total_saved} tokens  (final level: {final_level.name})")


# ---------------------------------------------------------------------------
# Phase 8: Stats
# ---------------------------------------------------------------------------


def print_stats(vm: MemoryManager) -> None:
    """Print VM metrics summary."""
    stats = vm.get_stats()
    m = stats.metrics
    ml = stats.mutation_log
    ws = stats.working_set

    print(f"  Session:      {stats.session_id}")
    print(f"  Turns:        {stats.turn}")
    print(f"  Pages total:  {stats.pages_in_store}")
    print(f"  L0 pages:     {ws.l0_pages}")
    print(f"  L0 tokens:    {ws.tokens_used}")
    print(f"  Faults:       {m.faults_total}")
    print(f"  Evictions:    {m.evictions_total}")
    print(f"  Compressions: {m.compressions_total}")
    print(f"  Tokens saved: {m.tokens_saved_by_compression}")
    print(
        f"  Mutations:    {ml.total_mutations} "
        f"(creates={ml.creates}, faults={ml.faults}, "
        f"evictions={ml.evictions})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Virtual Memory Demo (v0.11)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run VM flow without API calls",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        print("OPENAI_API_KEY not set.")
        print("  echo 'OPENAI_API_KEY=your-key' > .env")
        print("  python examples/07_llm_virtual_memory.py")
        print()
        print("Or run with --dry-run to see the VM flow without API calls.")
        return

    print("=" * 60)
    print("AI Virtual Memory -- LLM Demo (v0.11)")
    print("=" * 60)

    vm = setup_vm()
    client = None
    if api_key and not args.dry_run:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)

    # Phase 1 -- Build conversation
    print("\n--- Phase 1: Building conversation history (7 turns) ---")
    await build_conversation(vm)
    l0 = vm.get_l0_pages()
    print(f"  Created {len(vm._page_store)} pages, {len(l0)} in L0")
    for p in l0:
        pinned = "PINNED" if vm.working_set.is_pinned(p.page_id) else ""
        print(f"    [{p.page_type.value:10}] {p.page_id} {pinned}")

    # Phase 2 -- Create summaries
    print("\n--- Phase 2: Creating summary pages with provenance ---")
    await build_summaries(vm)
    l0 = vm.get_l0_pages()
    for page_id, _, _, _, provenance in SUMMARIES:
        print(f"  [{page_id}] provenance: {len(provenance)} pages  PINNED")
    print(f"  Total pages now: {len(vm._page_store)}, {len(l0)} in L0")

    # Phase 3 -- Full context summary
    print("\n--- Phase 3: Model summarizes with full context ---")
    if client:
        summary = await ask_with_full_context(vm, client, args.model)
        print(f"\n  Model:\n  {summary}")
    else:
        n = len(vm._page_store)
        print(f"  [dry-run] Model would see all {n} pages and summarize decisions.")

    # Phase 4 -- Evict
    print("\n--- Phase 4: Evicting old transcript pages ---")
    evicted = await evict_old_context(vm)
    l0 = vm.get_l0_pages()
    print(f"  Evicted {len(evicted)} transcript pages to L2")
    print(f"  Remaining in L0 ({len(l0)} pages):")
    for p in l0:
        print(f"    [{p.page_type.value:10}] {p.page_id}: {str(p.content)[:60]}...")

    # Phase 5 -- Search-driven recall
    print("\n--- Phase 5: Search-driven recall (search_pages demo) ---")
    if client:
        answer = await ask_with_search(vm, client, args.model)
        print(f"\n  Model:\n  {answer}")
    else:
        print(
            "  [dry-run] User asks about deployment/CI/CD (not in any claim/summary).\n"
            "  Model would call search_pages('deployment') -> find turn6_asst\n"
            "  Model would call page_fault('turn6_asst') -> load content\n"
            "  Model would answer with CI/CD pipeline details."
        )

    # Phase 6 -- Demand paging + cross-topic recall
    print("\n--- Phase 6: Demand paging + cross-topic recall ---")
    if client:
        answer = await ask_with_recall(vm, client, args.model)
        print(f"\n  Model:\n  {answer}")
    else:
        print(
            "  [dry-run] demand_pre_pass detects recall signals, pre-faults pages.\n"
            "  Model uses claim_monitoring + faulted transcripts to answer\n"
            "  about monitoring thresholds and correlation ID connections."
        )

    # Phase 7 -- Compression
    print("\n--- Phase 7: Compression pipeline ---")
    await show_compression(vm)

    # Phase 8 -- Stats
    print("\n\n--- VM Statistics ---")
    print_stats(vm)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
