#!/usr/bin/env python3
# examples/session_example.py
"""
Async Session-manager demo

• in-memory store
• simple session with events
• parent → children → grand-child hierarchy
• runs + run-level events
• prompt-builder showcase
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session import Session, SessionEvent
from chuk_ai_session_manager.models.session_run import SessionRun
from chuk_ai_session_manager.storage import SessionStoreProvider
from chuk_ai_session_manager.storage.providers.memory import InMemorySessionStore
from chuk_ai_session_manager.session_prompt_builder import (
    build_prompt_from_session,
    PromptStrategy,
)

# ─────────────────────────── logging ────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── helpers ────────────────────────────────
async def describe_session(sess: Session) -> None:
    log.info("\n=== Session %s ===", sess.id)
    log.info("events=%d | runs=%d | children=%d", len(sess.events), len(sess.runs), len(sess.child_ids))

    for evt in sess.events:
        log.info("  [%s/%s] %s", evt.source.value, evt.type.value, str(evt.message)[:60])

    for run in sess.runs:
        log.info("  run %s ⇒ %s", run.id, run.status.value)

    if (anc := await sess.ancestors()):
        log.info("  ancestors: %s", [a.id for a in anc])
    if (desc := await sess.descendants()):
        log.info("  descendants: %s", [d.id for d in desc])

# ─────────────────────────── main demo ──────────────────────────────
async def main() -> None:
    log.info("Bootstrapping in-memory store")
    store = InMemorySessionStore()
    SessionStoreProvider.set_store(store)

    # 1. simple session ------------------------------------------------
    simple = await Session.create()
    await simple.add_event_and_save(SessionEvent(
        message="Hello! I have a question about quantum computing.",
        source=EventSource.USER,
    ))
    await simple.add_event_and_save(SessionEvent(
        message="Sure - what would you like to know?",
        source=EventSource.LLM,
    ))
    await simple.add_event_and_save(SessionEvent(
        message="Can you explain entanglement in simple terms?",
        source=EventSource.USER,
    ))
    await simple.add_event_and_save(SessionEvent(
        message="User asked about quantum computing & entanglement.",
        source=EventSource.LLM,
        type=EventType.SUMMARY,
    ))
    await describe_session(simple)

    # 2. hierarchy -----------------------------------------------------
    log.info("\n=== Building hierarchy ===")

    parent = await Session.create()
    await parent.add_event_and_save(SessionEvent(
        message="Let's discuss AI capabilities.",
        source=EventSource.USER,
    ))

    child_a = await Session.create(parent_id=parent.id)
    await child_a.add_event_and_save(SessionEvent(
        message="Tell me about language models.",
        source=EventSource.USER,
    ))

    child_b = await Session.create(parent_id=parent.id)
    await child_b.add_event_and_save(SessionEvent(
        message="Tell me about computer vision.",
        source=EventSource.USER,
    ))

    grand = await Session.create(parent_id=child_a.id)
    await grand.add_event_and_save(SessionEvent(
        message="How do transformers work?",
        source=EventSource.USER,
    ))

    log.info("grand-child ancestors:   %s", [a.id for a in await grand.ancestors()])
    log.info("parent     descendants: %s", [d.id for d in await parent.descendants()])

    # 3. runs demo -----------------------------------------------------
    log.info("\n=== Session with runs ===")
    run_sess = await Session.create()
    await run_sess.add_event_and_save(SessionEvent(
        message="Analyse this data for me.",
        source=EventSource.USER,
    ))

    # create runs
    run1, run2, run3 = SessionRun(), SessionRun(), SessionRun()
    await run1.mark_running(); await run1.mark_completed()
    await run2.mark_running(); await run2.mark_failed()
    await run3.mark_running()

    run_sess.runs.extend([run1, run2, run3])

    # run-specific events
    await run_sess.add_event_and_save(SessionEvent(
        message="Processing dataset 1",
        source=EventSource.SYSTEM,
        task_id=run1.id,
    ))
    await run_sess.add_event_and_save(SessionEvent(
        message="Error on dataset 2",
        source=EventSource.SYSTEM,
        task_id=run2.id,
    ))
    await run_sess.add_event_and_save(SessionEvent(
        message="Dataset 3 in progress",
        source=EventSource.SYSTEM,
        task_id=run3.id,
    ))
    await describe_session(run_sess)

    # 4. prompt-builder demo ------------------------------------------
    log.info("\n=== Prompt-builder demo ===")
    tool_sess = await Session.create()
    await tool_sess.add_event_and_save(SessionEvent(
        message="What's the weather in New York?",
        source=EventSource.USER,
    ))

    assistant_evt = SessionEvent(
        message="I'll check that for you.",
        source=EventSource.LLM,
    )
    await tool_sess.add_event_and_save(assistant_evt)

    await tool_sess.add_event_and_save(SessionEvent(
        message={
            "tool_name": "get_weather",
            "result": {"temperature": 72, "condition": "Sunny", "location": "New York"},
        },
        source=EventSource.SYSTEM,
        type=EventType.TOOL_CALL,
        metadata={"parent_event_id": assistant_evt.id},
    ))

    minimal = await build_prompt_from_session(tool_sess, PromptStrategy.MINIMAL)
    focused = await build_prompt_from_session(tool_sess, PromptStrategy.TOOL_FOCUSED)

    log.info("\nMINIMAL strategy:\n%s", json_pretty(minimal))
    log.info("\nTOOL_FOCUSED strategy:\n%s", json_pretty(focused))

    # 5. list sessions -------------------------------------------------
    log.info("\n=== Store contents ===")
    for sid in await store.list_sessions():
        log.info("  • %s", sid)

    log.info("\nAll done - async demo complete ✅")

# tiny pretty-printer so we don’t need to import `json`
def json_pretty(obj):  # noqa: D401
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
