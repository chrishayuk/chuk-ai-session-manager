#!/usr/bin/env python3
"""
09_procedural_memory.py — Procedural Memory (Tool Usage Learning)

Demonstrates:
1. ToolMemoryManager — recording tool calls and outcomes
2. Failure-to-Fix detection — auto-detecting when a retry fixes a prior failure
3. ToolPattern aggregation — success/error patterns per tool
4. ProceduralContextFormatter — injecting learned patterns into model context
5. Session persistence — saving/loading procedural memory from sessions

No API keys required.
"""

import asyncio

from chuk_ai_session_manager.procedural_memory import (
    ToolMemoryManager,
    ToolOutcome,
    ProceduralContextFormatter,
    FormatterConfig,
)
from chuk_ai_session_manager.procedural_memory.models import ToolFixRelation
from chuk_ai_session_manager.models.session import Session


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print("=" * 60)


async def main() -> None:
    print("Procedural Memory — Tool Usage Learning Demo")
    print("No API keys required.\n")

    # ------------------------------------------------------------------ #
    section("1. Recording Tool Calls")
    # ------------------------------------------------------------------ #

    manager = ToolMemoryManager.create(session_id="demo-procedural")
    print(f"Created manager for session: {manager.session_id}")

    # Record a successful call
    entry1 = await manager.record_call(
        tool_name="weather_lookup",
        arguments={"city": "Tokyo", "units": "celsius"},
        result={"temp": 22, "condition": "Sunny"},
        outcome=ToolOutcome.SUCCESS,
        context_goal="Check weather for trip planning",
        execution_time_ms=150,
    )
    print(f"\nRecorded: {entry1.format_compact()}")

    # Record another successful call
    entry2 = await manager.record_call(
        tool_name="weather_lookup",
        arguments={"city": "London", "units": "celsius"},
        result={"temp": 12, "condition": "Rainy"},
        outcome=ToolOutcome.SUCCESS,
        context_goal="Compare weather",
        execution_time_ms=200,
    )
    print(f"Recorded: {entry2.format_compact()}")

    # Record a failed call
    entry3 = await manager.record_call(
        tool_name="flight_search",
        arguments={"from": "TYO", "to": "LHR", "date": "2025-13-01"},
        result=None,
        outcome=ToolOutcome.FAILURE,
        context_goal="Find flights",
        error_type="ValidationError",
        error_message="Invalid date format: month 13 out of range",
        execution_time_ms=50,
    )
    print(f"Recorded: {entry3.format_compact()}")

    # Record a timeout
    entry4 = await manager.record_call(
        tool_name="hotel_search",
        arguments={"city": "Tokyo", "checkin": "2025-06-01", "nights": 3},
        result=None,
        outcome=ToolOutcome.TIMEOUT,
        context_goal="Find accommodation",
        error_type="TimeoutError",
        error_message="Request timed out after 30s",
        execution_time_ms=30000,
    )
    print(f"Recorded: {entry4.format_compact()}")

    print(f"\nTotal entries: {len(manager.memory.tool_log)}")

    # ------------------------------------------------------------------ #
    section("2. Failure-to-Fix Detection")
    # ------------------------------------------------------------------ #

    # Set up a callback to see fix detections
    fixes_detected: list[ToolFixRelation] = []
    manager.on_fix_detected = lambda fix: fixes_detected.append(fix)

    # Now retry the flight search with corrected date
    entry5 = await manager.record_call(
        tool_name="flight_search",
        arguments={"from": "TYO", "to": "LHR", "date": "2025-06-01"},
        result={"flights": [{"price": 850, "airline": "JAL"}]},
        outcome=ToolOutcome.SUCCESS,
        context_goal="Find flights",
        execution_time_ms=300,
    )
    print(f"Retry recorded: {entry5.format_compact()}")

    if fixes_detected:
        fix = fixes_detected[0]
        print("\nFix detected!")
        print(f"  Failed call: {fix.failed_call_id}")
        print(f"  Fixed by:    {fix.success_call_id}")
        print(f"  Arg delta:   {fix.delta_args}")
    else:
        print("\nNo fix relationship detected (calls may have been too different)")

    # Check if the failed entry was marked as fixed
    failed = next((e for e in manager.memory.tool_log if e.id == entry3.id), None)
    if failed and failed.was_fixed():
        print(f"  Entry {failed.id} marked as fixed_by={failed.fixed_by}")

    # ------------------------------------------------------------------ #
    section("3. Tool Patterns")
    # ------------------------------------------------------------------ #

    # Add more calls to build patterns
    for city in ["Paris", "Berlin", "Rome"]:
        await manager.record_call(
            tool_name="weather_lookup",
            arguments={"city": city, "units": "celsius"},
            result={"temp": 18, "condition": "Cloudy"},
            outcome=ToolOutcome.SUCCESS,
            execution_time_ms=180,
        )

    # Also record a weather failure
    await manager.record_call(
        tool_name="weather_lookup",
        arguments={"city": "", "units": "celsius"},
        result=None,
        outcome=ToolOutcome.FAILURE,
        error_type="ValidationError",
        error_message="City name cannot be empty",
    )

    # View aggregated patterns
    pattern = manager.memory.get_pattern("weather_lookup")
    if pattern:
        print("weather_lookup pattern:")
        print(f"  Total calls: {pattern.total_calls}")
        print(f"  Successes:   {pattern.success_count}")
        print(f"  Failures:    {pattern.failure_count}")
        print(f"  Success rate: {pattern.success_rate:.0%}")
        if pattern.error_patterns:
            print(f"  Error patterns: {len(pattern.error_patterns)}")
            for ep in pattern.error_patterns:
                fix_info = f" (fix: {ep.typical_fix})" if ep.typical_fix else ""
                print(f"    - {ep.error_type} (x{ep.count}){fix_info}")

    pattern2 = manager.memory.get_pattern("flight_search")
    if pattern2:
        print("\nflight_search pattern:")
        print(f"  Total calls: {pattern2.total_calls}")
        print(f"  Successes:   {pattern2.success_count}")
        print(f"  Failures:    {pattern2.failure_count}")
        if pattern2.error_patterns:
            for ep in pattern2.error_patterns:
                fix_info = f" (fix: {ep.typical_fix})" if ep.typical_fix else ""
                print(f"  Error: {ep.error_type} (x{ep.count}){fix_info}")
    print(f"\nFix relations: {len(manager.memory.fix_relations)}")

    # ------------------------------------------------------------------ #
    section("4. ProceduralContextFormatter")
    # ------------------------------------------------------------------ #

    # Default formatting
    formatter = ProceduralContextFormatter()
    context = formatter.format_for_tools(
        manager,
        tool_names=["weather_lookup", "flight_search"],
        context_goal="Planning a trip",
    )
    print("Context for model (default):")
    print(context)

    # Compact formatting
    compact_formatter = ProceduralContextFormatter(
        config=FormatterConfig(
            compact=True,
            max_recent_calls=2,
            max_error_patterns=2,
            include_timing=True,
        )
    )
    compact_context = compact_formatter.format_for_tools(
        manager,
        tool_names=["weather_lookup"],
    )
    print("\nContext for model (compact + timing):")
    print(compact_context)

    # ------------------------------------------------------------------ #
    section("5. Session Persistence")
    # ------------------------------------------------------------------ #

    # Save to a session
    session = await Session.create()
    await manager.save_to_session(session)
    print(f"Saved procedural memory to session {session.id[:8]}...")

    # Load from session
    loaded = await ToolMemoryManager.from_session(session)
    print(f"Loaded manager: session={loaded.session_id}")
    print(f"  Entries: {len(loaded.memory.tool_log)}")
    print(f"  Patterns: {list(loaded.memory.tool_patterns.keys())}")
    print(f"  Fix relations: {len(loaded.memory.fix_relations)}")

    # Verify data survived round-trip
    loaded_pattern = loaded.memory.get_pattern("weather_lookup")
    if loaded_pattern:
        print(f"  weather_lookup success rate: {loaded_pattern.success_rate:.0%}")

    # ------------------------------------------------------------------ #
    section("6. Full Workflow — Tool Loop with Learning")
    # ------------------------------------------------------------------ #

    mgr = ToolMemoryManager.create(session_id="workflow-demo")

    print("Simulating a solver workflow:\n")

    # Step 1: Discovery — list available solvers
    e1 = await mgr.record_call(
        tool_name="list_solvers",
        arguments={},
        result=["cpsat", "glpk", "scip"],
        outcome=ToolOutcome.SUCCESS,
        context_goal="Find a solver for scheduling",
    )
    print(f"  1. {e1.format_compact()}")

    # Step 2: First attempt — wrong solver config
    e2 = await mgr.record_call(
        tool_name="solver_solve",
        arguments={"solver": "cpsat", "constraints": [], "objective": "minimize"},
        result=None,
        outcome=ToolOutcome.FAILURE,
        error_type="InvalidInput",
        error_message="constraints cannot be empty",
    )
    print(f"  2. {e2.format_compact()}")

    # Step 3: Fix — add constraints
    e3 = await mgr.record_call(
        tool_name="solver_solve",
        arguments={
            "solver": "cpsat",
            "constraints": [{"type": "capacity", "max": 10}],
            "objective": "minimize",
        },
        result={"status": "optimal", "value": 42},
        outcome=ToolOutcome.SUCCESS,
        context_goal="Solve the scheduling problem",
    )
    print(f"  3. {e3.format_compact()}")

    # Step 4: Another call using the result
    e4 = await mgr.record_call(
        tool_name="format_report",
        arguments={"solution_value": 42, "solver": "cpsat"},
        result="Schedule optimized: 42 units",
        outcome=ToolOutcome.SUCCESS,
        preceding_call_id=e3.id,
    )
    print(f"  4. {e4.format_compact()}")

    # Generate context for the model
    fmt = ProceduralContextFormatter(config=FormatterConfig(show_fix_relations=True))
    ctx = fmt.format_for_tools(mgr, ["solver_solve", "format_report"])
    print("\nLearned context for model:")
    print(ctx)

    # ------------------------------------------------------------------ #
    section("DEMO COMPLETE")
    # ------------------------------------------------------------------ #


if __name__ == "__main__":
    asyncio.run(main())
