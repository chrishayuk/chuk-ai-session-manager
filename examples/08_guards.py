#!/usr/bin/env python3
"""
08_guards.py — Conversation Guards & Tool State Management

Demonstrates:
1. ToolStateManager — the coordinator for guards, bindings, and cache
2. Value Bindings ($vN) — tracking tool results as referenceable values
3. Result Cache — deduplicating identical tool calls
4. Ungrounded Call Detection — flagging calls that don't reference prior results
5. Budget & Runaway Detection — preventing infinite tool loops
6. Guard Integration — checking all guards before executing a tool

No API keys required.
"""

import asyncio
from chuk_ai_session_manager.guards import (
    get_tool_state,
    reset_tool_state,
    RuntimeLimits,
    RuntimeMode,
)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print("=" * 60)


async def main() -> None:
    print("Conversation Guards & Tool State Management Demo")
    print("No API keys required.\n")

    # ------------------------------------------------------------------ #
    section("1. ToolStateManager — Basic Setup")
    # ------------------------------------------------------------------ #

    reset_tool_state()
    tsm = get_tool_state()

    # Configure with strict mode
    tsm.set_mode(RuntimeMode.STRICT)
    print("Mode: STRICT")
    print(f"Budget status: {tsm.get_budget_status()}")

    # Or configure custom limits
    tsm.configure(
        RuntimeLimits(
            tool_budget=20,
            max_tool_per_name=5,
            max_degenerate_streak=3,
        )
    )
    print("Custom limits applied: budget=20, per-name=5, degenerate-streak=3")

    # ------------------------------------------------------------------ #
    section("2. Value Bindings ($vN References)")
    # ------------------------------------------------------------------ #

    # Simulate: LLM calls math_add(a=5, b=3) -> 8
    b1 = tsm.bind_value("math_add", {"a": 5, "b": 3}, value=8)
    print(f"Binding: ${b1.id} = math_add(a=5, b=3) -> {b1.raw_value}")
    print(f"  Type: {b1.value_type}")

    # Simulate: LLM calls math_sqrt(x=64) -> 8.0
    b2 = tsm.bind_value("math_sqrt", {"x": 64}, value=8.0, aliases=["sqrt_64"])
    print(f"Binding: ${b2.id} = math_sqrt(x=64) -> {b2.raw_value}")
    print(f"  Aliases: {b2.aliases}")

    # Look up by ID (without $) or alias
    found = tsm.get_binding("v1")
    print(f"\nLookup 'v1': {found.raw_value if found else 'not found'}")
    found = tsm.get_binding("sqrt_64")
    print(f"Lookup 'sqrt_64': {found.raw_value if found else 'not found'}")

    # Resolve references in tool arguments
    args = {"x": "$v1", "y": "$v2"}
    resolved = tsm.resolve_references(args)
    print(f"\nResolve {args} -> {resolved}")

    # Check references (validation)
    bad_args = {"x": "$v1", "y": "$v99"}
    check = tsm.check_references(bad_args)
    print(f"Check {bad_args}: valid={check.valid}, missing={check.missing_refs}")

    # Format for model context
    print(f"\nBindings for model:\n{tsm.format_bindings_for_model()}")

    # ------------------------------------------------------------------ #
    section("3. Result Cache — Deduplication")
    # ------------------------------------------------------------------ #

    # Cache a tool result
    cached = tsm.cache_result(
        "weather_lookup", {"city": "Tokyo"}, {"temp": 22, "condition": "Sunny"}
    )
    print(f"Cached: weather_lookup(city='Tokyo') -> {cached.result}")
    print(f"  Signature: {cached.signature}")

    # Try to get the same call again — cache hit
    hit = tsm.get_cached_result("weather_lookup", {"city": "Tokyo"})
    print(f"\nCache hit: {hit is not None}")
    if hit:
        print(f"  Result: {hit.result}")
        print(f"  Call count: {hit.call_count}")

    # Different arguments — cache miss
    miss = tsm.get_cached_result("weather_lookup", {"city": "London"})
    print(f"Cache miss (London): {miss is None}")

    # Named variables
    tsm.store_variable("pi", 3.14159, units=None, source_tool="math_constant")
    pi = tsm.get_variable("pi")
    print(f"\nNamed variable 'pi': {pi.value if pi else 'not found'}")

    # Cache stats
    stats = tsm.get_cache_stats()
    print(
        f"Cache stats: {stats.total_cached} cached, {stats.duplicate_calls} duplicates, {stats.total_variables} variables"
    )

    # ------------------------------------------------------------------ #
    section("4. Ungrounded Call Detection")
    # ------------------------------------------------------------------ #

    # Register user's numeric literals from their prompt
    user_prompt = "Calculate the area of a circle with radius 5"
    count = tsm.register_user_literals(user_prompt)
    print(f"User prompt: '{user_prompt}'")
    print(f"  Extracted {count} numeric literals")

    # This call uses a user-provided literal (5) — grounded
    result = tsm.check_ungrounded_call("math_area", {"radius": 5})
    print(f"\nmath_area(radius=5): ungrounded={result.is_ungrounded}")
    print("  (5 is from user prompt, so it's grounded)")

    # This call uses 42 which wasn't in user prompt or bindings — ungrounded
    result = tsm.check_ungrounded_call("math_area", {"radius": 42})
    print(f"\nmath_area(radius=42): ungrounded={result.is_ungrounded}")
    print(f"  numeric_args={result.numeric_args}")
    print(f"  has_bindings={result.has_bindings}")

    # This call references a binding — grounded
    result = tsm.check_ungrounded_call("math_multiply", {"a": "$v1", "b": 5})
    print(f"\nmath_multiply(a=$v1, b=5): ungrounded={result.is_ungrounded}")
    print("  ($v1 is a binding reference, 5 is user literal)")

    # ------------------------------------------------------------------ #
    section("5. Budget & Runaway Detection")
    # ------------------------------------------------------------------ #

    reset_tool_state()
    tsm = get_tool_state()
    tsm.set_budget(5)
    print("Budget set to 5 tool calls")

    for i in range(6):
        tsm.record_tool_call(f"tool_{i % 2}")
        status = tsm.check_runaway(f"tool_{i % 2}")
        budget = tsm.get_budget_status()
        remaining = budget.total.limit - budget.total.used
        print(
            f"  Call {i + 1}: tool_{i % 2} — remaining={remaining}, "
            f"should_stop={status.should_stop}"
        )
        if status.should_stop:
            print(f"    Reason: {status.message}")
            break

    print(f"\nDiscovery exhausted: {tsm.is_discovery_exhausted()}")
    print(f"Execution exhausted: {tsm.is_execution_exhausted()}")

    # ------------------------------------------------------------------ #
    section("6. Guard Integration — check_all_guards()")
    # ------------------------------------------------------------------ #

    reset_tool_state()
    tsm = get_tool_state()
    tsm.set_budget(10)

    # Simulate a conversation flow
    print("Simulating a multi-tool conversation:\n")

    # Register user's numeric literals so guards know 144 is grounded
    tsm.register_user_literals("What is the square root of 144?")

    # Step 1: LLM calls math_sqrt(x=144) — grounded (144 in user prompt)
    ug1 = tsm.check_ungrounded_call("math_sqrt", {"x": 144})
    print(f"1. math_sqrt(x=144): ungrounded={ug1.is_ungrounded}")
    tsm.record_tool_call("math_sqrt")
    b = tsm.bind_value("math_sqrt", {"x": 144}, value=12.0)
    tsm.cache_result("math_sqrt", {"x": 144}, 12.0)
    print(f"   Bound: ${b.id} = 12.0")

    # Step 2: LLM calls math_multiply using $v1 bindings
    resolved = tsm.resolve_references({"a": "$v1", "b": "$v1"})
    ug2 = tsm.check_ungrounded_call("math_multiply", {"a": "$v1", "b": "$v1"})
    print(f"2. math_multiply(a=$v1, b=$v1): ungrounded={ug2.is_ungrounded}")
    tsm.record_tool_call("math_multiply")
    b2 = tsm.bind_value("math_multiply", resolved, value=144.0)
    tsm.cache_result("math_multiply", resolved, 144.0)
    print(f"   Resolved: {resolved}")
    print(f"   Bound: ${b2.id} = 144.0")

    # Step 3: Same call again — cache hit (deduplication)
    cached = tsm.get_cached_result("math_multiply", resolved)
    print(f"3. Duplicate math_multiply: cached={cached is not None}")

    # Step 4: Runaway check
    status = tsm.check_runaway()
    budget = tsm.get_budget_status()
    print(
        f"4. Budget: {budget.total.limit - budget.total.used} remaining, runaway={status.should_stop}"
    )

    # Final state summary for model
    print(f"\nState for model:\n{tsm.format_state_for_model()}")

    # ------------------------------------------------------------------ #
    section("7. Tool Classification")
    # ------------------------------------------------------------------ #

    tools = [
        "search_tools",
        "list_tools",  # discovery
        "math_add",
        "math_sqrt",  # idempotent math
        "web_search",
        "file_write",  # parameterized / execution
    ]
    for name in tools:
        disc = tsm.is_discovery_tool(name)
        math = tsm.is_idempotent_math_tool(name)
        param = tsm.is_parameterized_tool(name)
        labels = []
        if disc:
            labels.append("discovery")
        if math:
            labels.append("idempotent-math")
        if param:
            labels.append("parameterized")
        if not labels:
            labels.append("general")
        print(f"  {name:20s} -> {', '.join(labels)}")

    # ------------------------------------------------------------------ #
    section("DEMO COMPLETE")
    # ------------------------------------------------------------------ #


if __name__ == "__main__":
    asyncio.run(main())
