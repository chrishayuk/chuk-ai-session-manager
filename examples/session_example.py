#!/usr/bin/env python3
# examples/session_example.py
"""
Async Session-manager demo

Demonstrates:
• CHUK Sessions storage backend
• Simple session with events and token tracking
• Parent → children → grand-child hierarchy
• Runs + run-level events
• Prompt-builder showcase with multiple strategies
• Complete session lifecycle management
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List

from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.session_run import SessionRun, RunStatus
from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore, setup_chuk_sessions_storage
from chuk_ai_session_manager.session_prompt_builder import (
    build_prompt_from_session,
    PromptStrategy,
)

# ─────────────────────────── logging ────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Quiet down noisy loggers
logging.getLogger("chuk_sessions").setLevel(logging.WARNING)
logging.getLogger("chuk_ai_session_manager").setLevel(logging.WARNING)

# ─────────────────────────── helpers ────────────────────────────────
async def describe_session(sess: Session) -> None:
    """Pretty print session information."""
    log.info("\n=== Session %s ===", sess.id[:8] + "...")
    log.info("events=%d | runs=%d | children=%d | tokens=%d | cost=$%.6f", 
             len(sess.events), len(sess.runs), len(sess.child_ids), 
             sess.total_tokens, sess.total_cost)

    # Show events
    for i, evt in enumerate(sess.events, 1):
        content = str(evt.message)
        if len(content) > 60:
            content = content[:57] + "..."
        source_emoji = {"user": "👤", "llm": "🤖", "system": "🔧"}.get(evt.source.value, "❓")
        log.info("  %d. %s [%s/%s] %s", i, source_emoji, evt.source.value, evt.type.value, content)

    # Show runs
    for i, run in enumerate(sess.runs, 1):
        status_emoji = {
            "pending": "⏳",
            "running": "🔄", 
            "completed": "✅",
            "failed": "❌",
            "cancelled": "⏹️"
        }.get(run.status.value, "❓")
        log.info("  Run %d: %s %s (%s)", i, status_emoji, run.id[:8] + "...", run.status.value)

    # Show hierarchy relationships
    ancestors = await sess.ancestors()
    if ancestors:
        log.info("  🔗 ancestors: %s", [a.id[:8] + "..." for a in ancestors])
    
    descendants = await sess.descendants()
    if descendants:
        log.info("  🔗 descendants: %s", [d.id[:8] + "..." for d in descendants])

async def show_prompt_strategies(session: Session) -> None:
    """Demonstrate different prompt building strategies."""
    log.info("\n=== Prompt Building Strategies ===")
    
    strategies = [
        (PromptStrategy.MINIMAL, "Minimal"),
        (PromptStrategy.TASK_FOCUSED, "Task Focused"),
        (PromptStrategy.TOOL_FOCUSED, "Tool Focused"),
        (PromptStrategy.CONVERSATION, "Conversation"),
        (PromptStrategy.HIERARCHICAL, "Hierarchical")
    ]
    
    for strategy, name in strategies:
        try:
            prompt = await build_prompt_from_session(session, strategy)
            log.info("\n%s Strategy (%d messages):", name, len(prompt))
            for i, msg in enumerate(prompt, 1):
                role = msg.get("role", "unknown")
                content = msg.get("content")
                if content:
                    content_preview = content[:50] + "..." if len(content) > 50 else content
                    log.info("  %d. [%s]: %s", i, role, content_preview)
                else:
                    log.info("  %d. [%s]: <tool calls>", i, role)
        except Exception as e:
            log.warning("%s strategy failed: %s", name, e)

# ─────────────────────────── main demo ──────────────────────────────
async def main() -> None:
    """Run the comprehensive session management demo."""
    log.info("🚀 Starting CHUK AI Session Manager Demo")
    
    # Setup CHUK Sessions storage backend
    setup_chuk_sessions_storage(sandbox_id="session-example-demo", default_ttl_hours=2)
    backend = get_backend()
    store = ChukSessionsStore(backend)
    log.info("✅ CHUK Sessions storage initialized")

    # 1. Simple session with token tracking ----------------------------
    log.info("\n📝 1. Creating simple session with token tracking...")
    
    simple = await Session.create()
    await simple.metadata.set_property("demo", "simple_conversation")
    await simple.metadata.set_property("topic", "quantum_computing")
    
    # Add events with token tracking
    user_event1 = await SessionEvent.create_with_tokens(
        message="Hello! I have a question about quantum computing.",
        prompt="Hello! I have a question about quantum computing.",
        model="gpt-4o-mini",
        source=EventSource.USER,
    )
    await simple.add_event_and_save(user_event1)
    
    llm_event1 = await SessionEvent.create_with_tokens(
        message="Sure! I'd be happy to help with quantum computing questions. What specifically would you like to know?",
        prompt="",
        completion="Sure! I'd be happy to help with quantum computing questions. What specifically would you like to know?",
        model="gpt-4o-mini",
        source=EventSource.LLM,
    )
    await simple.add_event_and_save(llm_event1)
    
    user_event2 = await SessionEvent.create_with_tokens(
        message="Can you explain quantum entanglement in simple terms?",
        prompt="Can you explain quantum entanglement in simple terms?",
        model="gpt-4o-mini",
        source=EventSource.USER,
    )
    await simple.add_event_and_save(user_event2)
    
    llm_event2 = await SessionEvent.create_with_tokens(
        message="Quantum entanglement is when two particles become connected in such a way that measuring one instantly affects the other, no matter how far apart they are. Think of it like having two magical coins that always land on opposite sides.",
        prompt="",
        completion="Quantum entanglement is when two particles become connected in such a way that measuring one instantly affects the other, no matter how far apart they are. Think of it like having two magical coins that always land on opposite sides.",
        model="gpt-4o-mini",
        source=EventSource.LLM,
    )
    await simple.add_event_and_save(llm_event2)
    
    # Add a summary
    summary_event = SessionEvent(
        message="User asked about quantum computing fundamentals, specifically quantum entanglement. Provided simple explanation using coin analogy.",
        source=EventSource.SYSTEM,
        type=EventType.SUMMARY,
    )
    await simple.add_event_and_save(summary_event)
    
    await describe_session(simple)

    # 2. Hierarchical session structure --------------------------------
    log.info("\n🌳 2. Building hierarchical session structure...")

    parent = await Session.create()
    await parent.metadata.set_property("demo", "ai_capabilities_discussion")
    await parent.metadata.set_property("level", "parent")
    
    parent_event = await SessionEvent.create_with_tokens(
        message="Let's discuss different AI capabilities and how they work.",
        prompt="Let's discuss different AI capabilities and how they work.",
        model="gpt-4o-mini",
        source=EventSource.USER,
    )
    await parent.add_event_and_save(parent_event)

    # Child A: Language Models
    child_a = await Session.create(parent_id=parent.id)
    await child_a.metadata.set_property("demo", "language_models")
    await child_a.metadata.set_property("level", "child")
    
    child_a_event = await SessionEvent.create_with_tokens(
        message="Tell me about how language models like GPT work.",
        prompt="Tell me about how language models like GPT work.",
        model="gpt-4o-mini",
        source=EventSource.USER,
    )
    await child_a.add_event_and_save(child_a_event)

    # Child B: Computer Vision
    child_b = await Session.create(parent_id=parent.id)
    await child_b.metadata.set_property("demo", "computer_vision")
    await child_b.metadata.set_property("level", "child")
    
    child_b_event = await SessionEvent.create_with_tokens(
        message="How does computer vision work in AI systems?",
        prompt="How does computer vision work in AI systems?",
        model="gpt-4o-mini",
        source=EventSource.USER,
    )
    await child_b.add_event_and_save(child_b_event)

    # Grand-child: Deep dive into transformers
    grand = await Session.create(parent_id=child_a.id)
    await grand.metadata.set_property("demo", "transformer_architecture")
    await grand.metadata.set_property("level", "grandchild")
    
    grand_event = await SessionEvent.create_with_tokens(
        message="How do transformer architectures actually process text?",
        prompt="How do transformer architectures actually process text?",
        model="gpt-4o-mini", 
        source=EventSource.USER,
    )
    await grand.add_event_and_save(grand_event)

    # Refresh sessions to get updated relationships
    parent = await store.get(parent.id)
    child_a = await store.get(child_a.id)
    grand = await store.get(grand.id)

    log.info("🔗 Hierarchy relationships:")
    log.info("   Grand-child ancestors: %s", [a.id[:8] + "..." for a in await grand.ancestors()])
    log.info("   Parent descendants: %s", [d.id[:8] + "..." for d in await parent.descendants()])

    await describe_session(grand)

    # 3. Session runs demo ---------------------------------------------
    log.info("\n⚙️ 3. Session with runs and task tracking...")
    
    run_sess = await Session.create()
    await run_sess.metadata.set_property("demo", "data_analysis_workflow")
    
    # Initial user request
    user_request = await SessionEvent.create_with_tokens(
        message="Analyze these three datasets for patterns and generate a report.",
        prompt="Analyze these three datasets for patterns and generate a report.",
        model="gpt-4o-mini",
        source=EventSource.USER,
    )
    await run_sess.add_event_and_save(user_request)

    # Create runs for different datasets
    run1 = await SessionRun.create(metadata={"dataset": "sales_data.csv"})
    run2 = await SessionRun.create(metadata={"dataset": "customer_data.csv"})
    run3 = await SessionRun.create(metadata={"dataset": "product_data.csv"})

    # Simulate run lifecycle
    await run1.mark_running()
    await run1.mark_completed()
    
    await run2.mark_running()
    await run2.mark_failed("Data format validation error")
    
    await run3.mark_running()
    # Run 3 stays running

    run_sess.runs.extend([run1, run2, run3])

    # Add run-specific events
    run1_event = SessionEvent(
        message="Successfully processed sales_data.csv: Found seasonal patterns in Q4",
        source=EventSource.SYSTEM,
        task_id=run1.id,
    )
    await run_sess.add_event_and_save(run1_event)
    
    run2_event = SessionEvent(
        message="Error processing customer_data.csv: Invalid date format in column 'signup_date'",
        source=EventSource.SYSTEM,
        task_id=run2.id,
    )
    await run_sess.add_event_and_save(run2_event)
    
    run3_event = SessionEvent(
        message="Processing product_data.csv: Analyzing inventory patterns...",
        source=EventSource.SYSTEM,
        task_id=run3.id,
    )
    await run_sess.add_event_and_save(run3_event)
    
    await describe_session(run_sess)

    # 4. Tool execution demo -------------------------------------------
    log.info("\n🔧 4. Tool execution and prompt building demo...")
    
    tool_sess = await Session.create()
    await tool_sess.metadata.set_property("demo", "weather_tool_demo")
    
    # User asks for weather
    weather_request = await SessionEvent.create_with_tokens(
        message="What's the weather like in New York today?",
        prompt="What's the weather like in New York today?",
        model="gpt-4o-mini",
        source=EventSource.USER,
    )
    await tool_sess.add_event_and_save(weather_request)

    # Assistant responds with tool call
    assistant_evt = await SessionEvent.create_with_tokens(
        message="I'll check the current weather in New York for you.",
        prompt="",
        completion="I'll check the current weather in New York for you.",
        model="gpt-4o-mini",
        source=EventSource.LLM,
    )
    await tool_sess.add_event_and_save(assistant_evt)

    # Tool execution result
    tool_result = SessionEvent(
        message={
            "tool": "get_weather",
            "arguments": {"location": "New York"},
            "result": {
                "temperature": 72,
                "condition": "Sunny",
                "location": "New York",
                "humidity": 45,
                "wind_speed": 8.2
            },
            "success": True
        },
        source=EventSource.SYSTEM,
        type=EventType.TOOL_CALL,
    )
    await tool_result.set_metadata("parent_event_id", assistant_evt.id)
    await tool_sess.add_event_and_save(tool_result)

    # Final assistant response
    final_response = await SessionEvent.create_with_tokens(
        message="The weather in New York today is sunny with a temperature of 72°F. Humidity is at 45% with winds at 8.2 mph. It's a beautiful day!",
        prompt="",
        completion="The weather in New York today is sunny with a temperature of 72°F. Humidity is at 45% with winds at 8.2 mph. It's a beautiful day!",
        model="gpt-4o-mini",
        source=EventSource.LLM,
    )
    await tool_sess.add_event_and_save(final_response)

    await describe_session(tool_sess)

    # 5. Prompt building strategies showcase ---------------------------
    await show_prompt_strategies(tool_sess)

    # 6. Session statistics and summary --------------------------------
    log.info("\n📊 6. Session statistics and summary...")
    
    all_sessions = [simple, parent, child_a, child_b, grand, run_sess, tool_sess]
    
    total_sessions = len(all_sessions)
    total_events = sum(len(s.events) for s in all_sessions)
    total_tokens = sum(s.total_tokens for s in all_sessions)
    total_cost = sum(s.total_cost for s in all_sessions)
    
    log.info("📈 Demo Summary:")
    log.info("   Sessions created: %d", total_sessions)
    log.info("   Total events: %d", total_events)
    log.info("   Total tokens: %d", total_tokens)
    log.info("   Total estimated cost: $%.6f", total_cost)
    
    # Show event type breakdown
    event_types = {}
    for session in all_sessions:
        for event in session.events:
            event_type = f"{event.source.value}:{event.type.value}"
            event_types[event_type] = event_types.get(event_type, 0) + 1
    
    log.info("   Event breakdown:")
    for event_type, count in sorted(event_types.items()):
        log.info("     %s: %d", event_type, count)

    # 7. Storage backend info ------------------------------------------
    session_ids = await store.list_sessions()
    log.info("\n💾 Storage backend contains %d sessions", len(session_ids))
    for sid in session_ids[:5]:  # Show first 5
        log.info("   • %s", sid[:8] + "...")
    if len(session_ids) > 5:
        log.info("   ... and %d more", len(session_ids) - 5)

    log.info("\n🎉 All done - comprehensive async demo complete!")
    log.info("=" * 60)
    log.info("✨ Key Features Demonstrated:")
    log.info("   • Session creation and management")
    log.info("   • Event tracking with token usage")
    log.info("   • Hierarchical session relationships")
    log.info("   • Run-based task tracking")
    log.info("   • Tool execution logging")
    log.info("   • Multiple prompt building strategies")
    log.info("   • Complete observability and analytics")

# Helper function for pretty JSON printing
def json_pretty(obj):
    """Pretty print JSON with nice formatting."""
    return json.dumps(obj, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())