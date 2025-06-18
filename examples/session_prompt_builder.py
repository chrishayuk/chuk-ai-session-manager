#!/usr/bin/env python3
# examples/session_prompt_builder_demo_fixed.py
"""
Async demo for the Session Prompt Builder - FIXED VERSION

Highlights
----------
* Creates sessions and events entirely with the async API
* Shows Minimal / Conversation / Hierarchical / Tool-Focused strategies
* Demonstrates prompt-length management with truncate_prompt_to_token_limit()
* Uses a stubbed LLM + tool executor so it runs completely offline
* Fixed imports for current architecture with CHUK Sessions backend
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

# ── session imports - FIXED for current architecture ───────────────────
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore, setup_chuk_sessions_storage
from chuk_ai_session_manager.session_prompt_builder import (
    build_prompt_from_session,
    PromptStrategy,
    truncate_prompt_to_token_limit,
)

# ── logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# Quiet down noisy loggers
logging.getLogger("chuk_sessions").setLevel(logging.WARNING)
logging.getLogger("chuk_ai_session_manager").setLevel(logging.WARNING)

# ── stub LLM & tool helpers ─────────────────────────────────────────
async def fake_llm(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tiny stand-in for an LLM call that triggers a weather tool
    whenever the word "weather" appears in the prompt.
    """
    prompt_txt = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)
    log.info("🤖 LLM received %d messages (%d chars)", len(messages), len(prompt_txt))

    if "weather" in prompt_txt.lower():
        return {
            "role": "assistant",
            "content": "Let me check the weather for you.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York"}),
                    },
                }
            ],
        }

    return {
        "role": "assistant",
        "content": "This is a stubbed response from the fake LLM. I can help with various tasks!",
    }


async def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Fake tool execution so the demo stays offline."""
    log.info("🔧 Executing tool %s → %s", name, args)
    if name == "get_weather":
        city = args.get("location", "Unknown")
        return {
            "temperature": 72, 
            "condition": "Sunny", 
            "location": city,
            "humidity": 45,
            "wind_speed": 5.2
        }
    elif name == "get_time":
        return {"current_time": "2025-06-18T01:15:00Z", "timezone": "UTC"}
    return {"error": f"unknown tool: {name}"}


# ── individual strategy demos ───────────────────────────────────────
async def demo_minimal() -> None:
    """Demonstrate the MINIMAL prompt strategy."""
    log.info("\n🎯 === MINIMAL Strategy Demo ===")
    session = await Session.create()
    await session.metadata.set_property("demo", "minimal_strategy")

    # Add user message with token tracking
    user_event = await SessionEvent.create_with_tokens(
        message="What's the weather like in New York today?",
        prompt="What's the weather like in New York today?",
        model="gpt-4o-mini",
        source=EventSource.USER,
        type=EventType.MESSAGE,
    )
    await session.add_event_and_save(user_event)

    # Build initial prompt
    prompt = await build_prompt_from_session(session, PromptStrategy.MINIMAL)
    log.info("📤 Initial prompt sent to LLM:")
    for i, msg in enumerate(prompt, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content") or "<no content>"  # Handle None content
        log.info("   %d. [%s]: %s", i, role, content[:50] + "..." if len(content) > 50 else content)

    # Get LLM response
    llm_resp = await fake_llm(prompt)
    log.info("🤖 LLM Response: %s", llm_resp.get("content", ""))
    if llm_resp.get("tool_calls"):
        log.info("🔧 LLM wants to call %d tools", len(llm_resp["tool_calls"]))

    # Add assistant message with token tracking
    assistant_evt = await SessionEvent.create_with_tokens(
        message=llm_resp,
        prompt="",
        completion=llm_resp.get("content", ""),
        model="gpt-4o-mini",
        source=EventSource.LLM,
        type=EventType.MESSAGE,
    )
    await session.add_event_and_save(assistant_evt)

    # Execute tools if present
    for call in llm_resp.get("tool_calls", []):
        fn = call.get("function", {})
        tool_name = fn.get("name", "unknown")
        
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        
        result = await execute_tool(tool_name, args)
        
        # Log tool execution
        tool_event = SessionEvent(
            message={
                "tool": tool_name,
                "arguments": args,
                "result": result,
                "success": "error" not in result
            },
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL,
        )
        await tool_event.set_metadata("parent_event_id", assistant_evt.id)
        await tool_event.set_metadata("call_id", call.get("id", "unknown"))
        await session.add_event_and_save(tool_event)

    # Build final prompt after tool execution
    final_prompt = await build_prompt_from_session(session, PromptStrategy.MINIMAL)
    log.info("📤 Final prompt after tool execution (%d messages):", len(final_prompt))
    for i, msg in enumerate(final_prompt, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content")
        if content:
            preview = content[:50] + "..." if len(content) > 50 else content
            log.info("   %d. [%s]: %s", i, role, preview)
        else:
            log.info("   %d. [%s]: <tool calls>", i, role)

    log.info("✅ Minimal strategy demo complete - %d events, %d tokens", 
             len(session.events), session.total_tokens)


async def demo_conversation() -> None:
    """Demonstrate the CONVERSATION prompt strategy."""
    log.info("\n💬 === CONVERSATION Strategy Demo ===")
    session = await Session.create()
    await session.metadata.set_property("demo", "conversation_strategy")

    # Build a realistic conversation
    conversation = [
        ("user", "Tell me about quantum computing in simple terms."),
        ("assistant", "Quantum computers use quantum bits (qubits) that can exist in superpositions, allowing them to process information differently than classical computers."),
        ("user", "How is that different from regular computer bits?"),
        ("assistant", "Classical bits are strictly 0 or 1, like light switches that are either on or off. Qubits can be both 0 AND 1 simultaneously, like a spinning coin that's both heads and tails until it lands."),
        ("user", "What practical applications does quantum computing have?"),
        ("assistant", "Quantum computing shows promise for cryptography, drug discovery, financial modeling, and optimization problems. However, most applications are still theoretical as we're in the early stages of the technology."),
        ("user", "When will quantum computers become mainstream?"),
    ]

    for role, msg in conversation:
        if role == "user":
            event = await SessionEvent.create_with_tokens(
                message=msg,
                prompt=msg,
                model="gpt-4o-mini",
                source=EventSource.USER,
                type=EventType.MESSAGE,
            )
        else:
            event = await SessionEvent.create_with_tokens(
                message=msg,
                prompt="",
                completion=msg,
                model="gpt-4o-mini",
                source=EventSource.LLM,
                type=EventType.MESSAGE,
            )
        await session.add_event_and_save(event)

    # Build conversation prompt
    prompt = await build_prompt_from_session(session, PromptStrategy.CONVERSATION)
    log.info("📤 Conversation prompt (%d messages):", len(prompt))
    for i, msg in enumerate(prompt, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""  # Handle None content
        preview = content[:60] + "..." if len(content) > 60 else content
        log.info("   %d. [%s]: %s", i, role, preview)

    log.info("✅ Conversation strategy demo complete - %d events, %d tokens", 
             len(session.events), session.total_tokens)


async def demo_hierarchical() -> None:
    """Demonstrate the HIERARCHICAL prompt strategy."""
    log.info("\n🌳 === HIERARCHICAL Strategy Demo ===")
    
    # Create parent session
    parent = await Session.create()
    await parent.metadata.set_property("demo", "hierarchical_parent")
    await parent.metadata.set_property("topic", "travel_planning")
    
    parent_event = await SessionEvent.create_with_tokens(
        message="I'm planning a trip to Japan and want to explore both historical sites and natural beauty.",
        prompt="I'm planning a trip to Japan and want to explore both historical sites and natural beauty.",
        model="gpt-4o-mini",
        source=EventSource.USER,
        type=EventType.MESSAGE,
    )
    await parent.add_event_and_save(parent_event)
    
    # Add summary to parent
    summary_event = SessionEvent(
        message="User wants Japan trip combining historical sites (temples, castles) with nature (mountains, gardens). Interested in cultural experiences and scenic beauty.",
        source=EventSource.SYSTEM,
        type=EventType.SUMMARY,
    )
    await parent.add_event_and_save(summary_event)

    # Create child session
    child = await Session.create(parent_id=parent.id)
    await child.metadata.set_property("demo", "hierarchical_child")
    await child.metadata.set_property("topic", "itinerary_planning")
    
    child_event = await SessionEvent.create_with_tokens(
        message="Can you suggest a detailed 7-day itinerary that balances temples, nature, and local food experiences?",
        prompt="Can you suggest a detailed 7-day itinerary that balances temples, nature, and local food experiences?",
        model="gpt-4o-mini",
        source=EventSource.USER,
        type=EventType.MESSAGE,
    )
    await child.add_event_and_save(child_event)

    # Build hierarchical prompt (includes parent context)
    prompt = await build_prompt_from_session(
        child,
        PromptStrategy.HIERARCHICAL,
        include_parent_context=True,
    )
    log.info("📤 Hierarchical prompt with parent context (%d messages):", len(prompt))
    for i, msg in enumerate(prompt, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""  # Handle None content
        preview = content[:70] + "..." if len(content) > 70 else content
        log.info("   %d. [%s]: %s", i, role, preview)

    log.info("✅ Hierarchical strategy demo complete")
    log.info("   Parent session: %d events, %d tokens", len(parent.events), parent.total_tokens)
    log.info("   Child session: %d events, %d tokens", len(child.events), child.total_tokens)


async def demo_tool_focused() -> None:
    """Demonstrate the TOOL_FOCUSED prompt strategy."""
    log.info("\n🔧 === TOOL_FOCUSED Strategy Demo ===")
    session = await Session.create()
    await session.metadata.set_property("demo", "tool_focused_strategy")

    # User asks for weather in multiple cities
    user_event = await SessionEvent.create_with_tokens(
        message="What's the weather like in New York, Tokyo, and London right now?",
        prompt="What's the weather like in New York, Tokyo, and London right now?",
        model="gpt-4o-mini",
        source=EventSource.USER,
        type=EventType.MESSAGE,
    )
    await session.add_event_and_save(user_event)

    # Assistant responds
    assistant_evt = await SessionEvent.create_with_tokens(
        message="I'll check the current weather in all three cities for you.",
        prompt="",
        completion="I'll check the current weather in all three cities for you.",
        model="gpt-4o-mini",
        source=EventSource.LLM,
        type=EventType.MESSAGE,
    )
    await session.add_event_and_save(assistant_evt)

    # Add multiple tool calls
    cities_weather = [
        ("New York", "Sunny", 72),
        ("Tokyo", "Rainy", 18),
        ("London", "Cloudy", 15)
    ]

    for city, condition, temp in cities_weather:
        tool_event = SessionEvent(
            message={
                "tool": "get_weather",
                "arguments": {"location": city},
                "result": {
                    "location": city,
                    "condition": condition, 
                    "temperature": temp,
                    "humidity": 50 + (temp % 30),
                    "timestamp": "2025-06-18T01:15:00Z"
                },
                "success": True
            },
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL,
        )
        await tool_event.set_metadata("parent_event_id", assistant_evt.id)
        await tool_event.set_metadata("city", city)
        await session.add_event_and_save(tool_event)

    # Build tool-focused prompt
    prompt = await build_prompt_from_session(session, PromptStrategy.TOOL_FOCUSED)
    log.info("📤 Tool-focused prompt (%d messages):", len(prompt))
    for i, msg in enumerate(prompt, 1):
        role = msg.get("role", "unknown")
        name = msg.get("name", "")
        content = msg.get("content") or ""  # Handle None content
        
        if role == "tool":
            log.info("   %d. [%s:%s]: %s", i, role, name, content[:50] + "..." if len(content) > 50 else content)
        else:
            preview = content[:60] + "..." if len(content) > 60 else content
            log.info("   %d. [%s]: %s", i, role, preview)

    log.info("✅ Tool-focused strategy demo complete - %d events, %d tokens", 
             len(session.events), session.total_tokens)


async def demo_token_truncation() -> None:
    """Demonstrate prompt truncation to stay within token limits."""
    log.info("\n✂️ === TOKEN TRUNCATION Demo ===")
    session = await Session.create()
    await session.metadata.set_property("demo", "token_truncation")

    # Create a very long conversation
    log.info("🔄 Creating long conversation (50 exchanges)...")
    for i in range(25):
        # User message
        user_msg = f"User message #{i+1}: This is a longer message to simulate a real conversation. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        user_event = await SessionEvent.create_with_tokens(
            message=user_msg,
            prompt=user_msg,
            model="gpt-4o-mini",
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
        await session.add_event_and_save(user_event)

        # Assistant response
        assistant_msg = f"Assistant response #{i+1}: This is a detailed response that provides helpful information. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
        assistant_event = await SessionEvent.create_with_tokens(
            message=assistant_msg,
            prompt="",
            completion=assistant_msg,
            model="gpt-4o-mini",
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
        await session.add_event_and_save(assistant_event)

    # Build full conversation prompt
    full_prompt = await build_prompt_from_session(session, PromptStrategy.CONVERSATION)
    log.info("📏 Full conversation prompt: %d messages, ~%d tokens", 
             len(full_prompt), session.total_tokens)

    # Truncate to different token limits
    limits = [500, 1000, 2000]
    
    for limit in limits:
        truncated = await truncate_prompt_to_token_limit(full_prompt, max_tokens=limit)
        log.info("✂️ Truncated to %d tokens: %d messages remain", limit, len(truncated))
        
        if truncated:
            first_content = truncated[0].get("content") or ""
            last_content = truncated[-1].get("content") or ""
            log.info("   First message: [%s] %s", 
                     truncated[0].get("role", "unknown"),
                     first_content[:50] + "..." if len(first_content) > 50 else first_content)
            log.info("   Last message: [%s] %s",
                     truncated[-1].get("role", "unknown"), 
                     last_content[:50] + "..." if len(last_content) > 50 else last_content)

    log.info("✅ Token truncation demo complete - original %d events, %d tokens", 
             len(session.events), session.total_tokens)


async def demo_all_strategies_comparison() -> None:
    """Compare all prompt strategies on the same session."""
    log.info("\n🔍 === ALL STRATEGIES COMPARISON ===")
    
    # Create a rich session with various event types
    session = await Session.create()
    await session.metadata.set_property("demo", "strategies_comparison")

    # Add conversation
    user_event = await SessionEvent.create_with_tokens(
        message="I need help planning a productive work session. Can you check my calendar and the weather?",
        prompt="I need help planning a productive work session. Can you check my calendar and the weather?",
        model="gpt-4o-mini",
        source=EventSource.USER,
        type=EventType.MESSAGE,
    )
    await session.add_event_and_save(user_event)

    assistant_event = await SessionEvent.create_with_tokens(
        message="I'll help you plan your work session by checking your calendar and the current weather conditions.",
        prompt="",
        completion="I'll help you plan your work session by checking your calendar and the current weather conditions.",
        model="gpt-4o-mini",
        source=EventSource.LLM,
        type=EventType.MESSAGE,
    )
    await session.add_event_and_save(assistant_event)

    # Add tool calls
    tool_events = [
        {
            "tool": "get_weather",
            "args": {"location": "San Francisco"},
            "result": {"temperature": 68, "condition": "Foggy", "location": "San Francisco"}
        },
        {
            "tool": "check_calendar", 
            "args": {"date": "2025-06-18"},
            "result": {"events": ["Team meeting at 10am", "Lunch with client at 12pm"], "free_time": "2pm-5pm"}
        }
    ]

    for tool_info in tool_events:
        tool_event = SessionEvent(
            message={
                "tool": tool_info["tool"],
                "arguments": tool_info["args"],
                "result": tool_info["result"],
                "success": True
            },
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL,
        )
        await tool_event.set_metadata("parent_event_id", assistant_event.id)
        await session.add_event_and_save(tool_event)

    # Test all strategies
    strategies = [
        (PromptStrategy.MINIMAL, "Minimal"),
        (PromptStrategy.TASK_FOCUSED, "Task Focused"),
        (PromptStrategy.TOOL_FOCUSED, "Tool Focused"),
        (PromptStrategy.CONVERSATION, "Conversation"),
        (PromptStrategy.HIERARCHICAL, "Hierarchical")
    ]

    log.info("📊 Comparing all strategies on the same session:")
    for strategy, name in strategies:
        try:
            prompt = await build_prompt_from_session(session, strategy)
            log.info("   %s: %d messages", name, len(prompt))
            
            # Show structure
            roles = [msg.get("role", "unknown") for msg in prompt]
            role_counts = {}
            for role in roles:
                role_counts[role] = role_counts.get(role, 0) + 1
            log.info("     Structure: %s", dict(role_counts))
            
        except Exception as e:
            log.warning("   %s: Failed - %s", name, e)

    log.info("✅ Strategy comparison complete")


# ── main orchestration ───────────────────────────────────────────────
async def main() -> None:
    """Run all prompt builder demos."""
    log.info("🚀 Starting Session Prompt Builder Demo")
    
    # Setup CHUK Sessions storage
    setup_chuk_sessions_storage(sandbox_id="prompt-builder-demo", default_ttl_hours=1)
    log.info("✅ CHUK Sessions storage initialized")

    # Run all demos
    await demo_minimal()
    await demo_conversation()
    await demo_hierarchical()
    await demo_tool_focused()
    await demo_token_truncation()
    await demo_all_strategies_comparison()

    # Final summary
    backend = get_backend()
    store = ChukSessionsStore(backend)
    session_ids = await store.list_sessions()
    
    log.info("\n🎉 All prompt-builder demos complete!")
    log.info("=" * 50)
    log.info("📊 Demo Summary:")
    log.info("   Sessions created: %d", len(session_ids))
    log.info("   Strategies tested: 5 (Minimal, Task, Tool, Conversation, Hierarchical)")
    log.info("   Features demonstrated:")
    log.info("     • Multiple prompt building strategies")
    log.info("     • Token counting and truncation")
    log.info("     • Tool call integration")
    log.info("     • Hierarchical session context")
    log.info("     • Offline LLM simulation")
    log.info("     • Complete session tracking")


if __name__ == "__main__":
    asyncio.run(main())