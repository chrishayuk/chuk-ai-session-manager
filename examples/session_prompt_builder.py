#!/usr/bin/env python3
# examples/session_prompt_builder.py
"""
Async demo for the Session Prompt Builder.

Highlights
----------
* Creates sessions and events entirely with the async API
* Shows Minimal / Conversation / Hierarchical / Tool-Focused strategies
* Demonstrates prompt-length management with truncate_prompt_to_token_limit()
* Uses a stubbed LLM + tool executor so it runs completely offline
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

# ── session imports ─────────────────────────────────────────────────────
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session import Session, SessionEvent
from chuk_ai_session_manager.storage import SessionStoreProvider
from chuk_ai_session_manager.storage.providers.memory import InMemorySessionStore
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

# ── stub LLM & tool helpers ─────────────────────────────────────────
async def fake_llm(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tiny stand-in for an LLM call that triggers a weather tool
    whenever the word “weather” appears in the prompt.
    """
    prompt_txt = "\n".join(f"{m['role']}: {m.get('content')}" for m in messages)
    log.info("LLM received %d msgs (%d chars)", len(messages), len(prompt_txt))

    if "weather" in prompt_txt.lower():
        return {
            "role": "assistant",
            "content": "Let me check the weather for you.",
            "tool_calls": [
                {
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
        "content": "This is a stubbed response from the fake LLM.",
    }


async def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Fake tool execution so the demo stays offline."""
    log.info("Executing tool %s → %s", name, args)
    if name == "get_weather":
        city = args.get("location", "Unknown")
        return {"temperature": 72, "condition": "Sunny", "location": city}
    return {"error": "unknown-tool"}


# ── individual strategy demos ───────────────────────────────────────
async def demo_minimal() -> None:
    log.info("\n=== MINIMAL strategy ===")
    session = await Session.create()

    await session.add_event_and_save(
        SessionEvent(
            message="What's the weather in New York?",
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
    )

    prompt = await build_prompt_from_session(session, PromptStrategy.MINIMAL)
    log.info("Prompt sent to LLM:\n%s", json.dumps(prompt, indent=2))

    llm_resp = await fake_llm(prompt)

    # add assistant message
    assistant_evt = SessionEvent(
        message=llm_resp,
        source=EventSource.LLM,
        type=EventType.MESSAGE,
    )
    await session.add_event_and_save(assistant_evt)

    # run tool(s) if present
    for call in llm_resp.get("tool_calls", []):
        fn = call.get("function", {})
        result = await execute_tool(fn.get("name"), json.loads(fn.get("arguments", "{}")))
        await session.add_event_and_save(
            SessionEvent(
                message={
                    "tool_name": fn.get("name"),
                    "arguments": json.loads(fn.get("arguments", "{}")),
                    "result": result,
                },
                source=EventSource.SYSTEM,
                type=EventType.TOOL_CALL,
                metadata={"parent_event_id": assistant_evt.id},
            )
        )

    prompt = await build_prompt_from_session(session, PromptStrategy.MINIMAL)
    log.info("Prompt after tool execution:\n%s", json.dumps(prompt, indent=2))


async def demo_conversation() -> None:
    log.info("\n=== CONVERSATION strategy ===")
    session = await Session.create()

    convo = [
        ("user", "Tell me about quantum computing."),
        ("assistant", "Quantum computers use qubits that can exist in superpositions."),
        ("user", "How is that different from classical bits?"),
        ("assistant", "Classical bits are strictly 0 or 1; qubits can be both."),
        ("user", "What practical applications does it have?"),
    ]
    for role, msg in convo:
        await session.add_event_and_save(
            SessionEvent(
                message=msg,
                source=EventSource.USER if role == "user" else EventSource.LLM,
                type=EventType.MESSAGE,
            )
        )

    prompt = await build_prompt_from_session(session, PromptStrategy.CONVERSATION)
    log.info("Conversation prompt:\n%s", json.dumps(prompt, indent=2))


async def demo_hierarchical() -> None:
    log.info("\n=== HIERARCHICAL strategy ===")
    store = InMemorySessionStore()
    SessionStoreProvider.set_store(store)

    parent = await Session.create()
    await parent.add_event_and_save(
        SessionEvent(
            message="Planning a trip to Japan.",
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
    )
    await parent.add_event_and_save(
        SessionEvent(
            message="User wants historical sites and nature.",
            source=EventSource.SYSTEM,
            type=EventType.SUMMARY,
        )
    )
    child = await Session.create(parent_id=parent.id)
    await child.add_event_and_save(
        SessionEvent(
            message="Can you suggest a 7-day itinerary?",
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
    )

    prompt = await build_prompt_from_session(
        child,
        PromptStrategy.HIERARCHICAL,
        include_parent_context=True,
    )
    log.info("Hierarchical prompt (with parent context):\n%s", json.dumps(prompt, indent=2))


async def demo_tool_focused() -> None:
    log.info("\n=== TOOL_FOCUSED strategy ===")
    session = await Session.create()
    await session.add_event_and_save(
        SessionEvent(
            message="Weather in New York, Tokyo and London?",
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
    )
    assistant_evt = SessionEvent(
        message="I'll check the weather.",
        source=EventSource.LLM,
        type=EventType.MESSAGE,
    )
    await session.add_event_and_save(assistant_evt)

    for city, cond in (("New York", "Sunny"), ("Tokyo", "Rainy"), ("London", "Cloudy")):
        await session.add_event_and_save(
            SessionEvent(
                message={
                    "tool_name": "get_weather",
                    "arguments": {"location": city},
                    "result": {"condition": cond, "temperature": 70},
                },
                source=EventSource.SYSTEM,
                type=EventType.TOOL_CALL,
                metadata={"parent_event_id": assistant_evt.id},
            )
        )

    prompt = await build_prompt_from_session(session, PromptStrategy.TOOL_FOCUSED)
    log.info("Tool-focused prompt:\n%s", json.dumps(prompt, indent=2))


async def demo_token_truncation() -> None:
    log.info("\n=== token-limit truncation demo ===")
    session = await Session.create()
    # make a long chat
    for i in range(25):
        await session.add_event_and_save(
            SessionEvent(
                message=f"User message {i+1} … Lorem ipsum dolor sit amet.",
                source=EventSource.USER,
                type=EventType.MESSAGE,
            )
        )
        await session.add_event_and_save(
            SessionEvent(
                message=f"Assistant response {i+1} … Dolor sit amet lorem ipsum.",
                source=EventSource.LLM,
                type=EventType.MESSAGE,
            )
        )

    full_prompt = await build_prompt_from_session(session, PromptStrategy.CONVERSATION)
    log.info("Full prompt has %d messages", len(full_prompt))

    truncated = await truncate_prompt_to_token_limit(full_prompt, max_tokens=500)
    log.info("After truncate_prompt_to_token_limit → %d messages", len(truncated))
    log.info("First 3 truncated messages:\n%s", json.dumps(truncated[:3], indent=2))


# ── main orchestration ───────────────────────────────────────────────
async def main() -> None:
    log.info("Setting up in-memory store")
    SessionStoreProvider.set_store(InMemorySessionStore())

    await demo_minimal()
    await demo_conversation()
    await demo_hierarchical()
    await demo_tool_focused()
    await demo_token_truncation()

    log.info("All prompt-builder demos complete")


if __name__ == "__main__":
    asyncio.run(main())
