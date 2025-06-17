#!/usr/bin/env python3
# examples/session_token_usage_example.py
#!/usr/bin/env python3
"""
Async token-usage demo for chuk session manager.

• tracks per-event / per-model / per-source tokens & cost
• falls back to a 4-chars≈1-token heuristic when tiktoken isn’t installed
• shows multi-model usage and a running-total logger
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List

from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type   import EventType
from chuk_ai_session_manager.models.session      import Session, SessionEvent
from chuk_ai_session_manager.models.token_usage  import (
    TokenUsage, TIKTOKEN_AVAILABLE
)
from chuk_ai_session_manager.storage             import (
    SessionStoreProvider, InMemorySessionStore
)

# ── logging setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────
async def bootstrap_store() -> None:
    """Create & register an in-memory store so everything is self-contained."""
    SessionStoreProvider.set_store(InMemorySessionStore())
    log.info("Bootstrapping in-memory store")


async def token_usage_report(sess: Session) -> None:
    """Pretty-print token & cost stats for a session."""
    log.info("\n=== Token report for %s ===", sess.id)
    log.info("Total tokens: %d | cost: $%.6f", sess.total_tokens, sess.total_cost)

    # by-model
    log.info("\nBy model:")
    for mdl, usage in sess.token_summary.usage_by_model.items():
        log.info(
            "  %-13s prompt=%4d  completion=%4d  total=%4d  cost=$%.6f",
            mdl,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            usage.estimated_cost_usd,
        )

    # by-source
    log.info("\nBy source:")
    source_usage = await sess.get_token_usage_by_source()      # ← fixed (await!)
    for src, summary in source_usage.items():
        log.info(
            "  %-7s total=%4d  cost=$%.6f",
            src,
            summary.total_tokens,
            summary.total_estimated_cost_usd,
        )


# ── demo 1: single-model conversation ───────────────────────────────
async def create_basic_session() -> Session:
    sess = await Session.create()
    log.info("Created session %s", sess.id)

    user_q = "Hello, can you explain quantum computing in simple terms?"
    await sess.add_event_and_save(
        SessionEvent(message=user_q, source=EventSource.USER, type=EventType.MESSAGE)
    )

    answer = (
        "Quantum computing uses qubits that can be 0, 1 **or both at once**. "
        "That superposition, plus entanglement, lets quantum computers try many "
        "possibilities simultaneously - like exploring every maze path at once."
    )
    await sess.add_event_and_save(
        await SessionEvent.create_with_tokens(
            message=answer,
            prompt=user_q,
            completion=answer,
            model="gpt-4",
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
    )
    return sess


# ── demo 2: multi-model usage ───────────────────────────────────────
async def create_multi_model_session() -> Session:
    sess = await Session.create()
    log.info("Created multi-model session %s", sess.id)

    models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]
    for mdl in models:
        q = f"What is special about {mdl}?"
        await sess.add_event_and_save(
            SessionEvent(message=q, source=EventSource.USER, type=EventType.MESSAGE)
        )

        resp = f"[{mdl}] is amazing because … " * 5
        await sess.add_event_and_save(
            await SessionEvent.create_with_tokens(
                message=resp,
                prompt=q,
                completion=resp,
                model=mdl,
                source=EventSource.LLM,
                type=EventType.MESSAGE,
            )
        )
    return sess


# ── demo 3: running-total cost tracker ──────────────────────────────
async def running_cost_demo() -> Session:
    sess = await Session.create()
    log.info("Created cost-tracking session %s", sess.id)

    convo: List[Dict[str, str]] = [
        {"role": "user", "content": "Plan a 3-day Kyoto trip."},
        {"role": "assistant", "content": "Sure — here’s a draft itinerary …"},
        {"role": "user", "content": "Add a day trip to Nara."},
        {"role": "assistant", "content": "Updated plan with Nara included …"},
    ]
    model_cycle = ["gpt-4", "gpt-3.5-turbo"]
    idx = 0

    for turn in convo:
        if turn["role"] == "user":
            await sess.add_event_and_save(
                SessionEvent(
                    message=turn["content"],
                    source=EventSource.USER,
                    type=EventType.MESSAGE,
                )
            )
        else:
            mdl = model_cycle[idx % len(model_cycle)]
            idx += 1
            ev = await SessionEvent.create_with_tokens(
                message=turn["content"],
                prompt=convo[idx * 2 - 2]["content"],
                completion=turn["content"],
                model=mdl,
                source=EventSource.LLM,
                type=EventType.MESSAGE,
            )
            await sess.add_event_and_save(ev)
            log.info(
                "Added assistant msg (%s): %d tok, $%.6f → running total: %d tok, $%.6f",
                mdl,
                ev.token_usage.total_tokens,
                ev.token_usage.estimated_cost_usd,
                sess.total_tokens,
                sess.total_cost,
            )
    return sess


# ── main orchestrator ───────────────────────────────────────────────
async def main() -> None:
    log.info("Starting async token-tracking demo")
    await bootstrap_store()

    log.warning(
        "tiktoken %savailable → %s",
        "" if TIKTOKEN_AVAILABLE else "NOT ",
        "accurate counts" if TIKTOKEN_AVAILABLE else "using 4-chars≈1-token heuristic",
    )

    basic = await create_basic_session()
    await token_usage_report(basic)

    multi = await create_multi_model_session()
    await token_usage_report(multi)

    costs = await running_cost_demo()
    await token_usage_report(costs)

    log.info("Token-tracking demo complete ✅")


if __name__ == "__main__":
    asyncio.run(main())
