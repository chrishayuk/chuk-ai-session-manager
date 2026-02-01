#!/usr/bin/env python3
# examples/session_token_usage_demo_fixed.py
"""
Async token-usage demo for chuk session manager - FIXED VERSION

Features demonstrated:
• Tracks per-event / per-model / per-source tokens & cost
• Falls back to 4-chars≈1-token heuristic when tiktoken isn't installed
• Shows multi-model usage and running-total cost tracking
• Comprehensive token analytics and cost estimation
• Real-time cost monitoring for budget management
"""

from __future__ import annotations

import asyncio
import logging

from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.token_usage import TokenUsage, TIKTOKEN_AVAILABLE
from chuk_ai_session_manager.session_storage import setup_chuk_sessions_storage

# ── logging setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# Quiet down noisy loggers
logging.getLogger("chuk_sessions").setLevel(logging.WARNING)
logging.getLogger("chuk_ai_session_manager").setLevel(logging.WARNING)


# ── helpers ──────────────────────────────────────────────────────────
async def bootstrap_store() -> None:
    """Set up CHUK Sessions storage backend."""
    setup_chuk_sessions_storage(sandbox_id="token-usage-demo", default_ttl_hours=1)
    log.info("✅ CHUK Sessions storage initialized")


async def token_usage_report(sess: Session) -> None:
    """Pretty-print comprehensive token & cost stats for a session."""
    log.info("\n=== 📊 Token Report for Session %s ===", sess.id[:8] + "...")
    log.info(
        "💰 Total tokens: %d | Total cost: $%.6f", sess.total_tokens, sess.total_cost
    )

    # By-model breakdown
    log.info("\n🤖 By Model:")
    if sess.token_summary.usage_by_model:
        for model, usage in sess.token_summary.usage_by_model.items():
            log.info(
                "  %-15s prompt=%4d  completion=%4d  total=%4d  cost=$%.6f",
                model,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                usage.estimated_cost_usd or 0.0,
            )
    else:
        log.info("  No model-specific usage data")

    # By-source breakdown
    log.info("\n👥 By Source:")
    try:
        source_usage = await sess.get_token_usage_by_source()
        if source_usage:
            for source, summary in source_usage.items():
                log.info(
                    "  %-10s total=%4d  cost=$%.6f",
                    source,
                    summary.total_tokens,
                    summary.total_estimated_cost_usd,
                )
        else:
            log.info("  No source-specific usage data")
    except Exception as e:
        log.warning("  Could not get source usage: %s", e)

    # Event-level details
    log.info("\n📝 Event Details:")
    token_events = [
        e for e in sess.events if e.token_usage and e.token_usage.total_tokens > 0
    ]
    if token_events:
        for i, event in enumerate(token_events, 1):
            source_emoji = {"user": "👤", "llm": "🤖", "system": "🔧"}.get(
                event.source.value, "❓"
            )
            tokens = event.token_usage.total_tokens
            cost = event.token_usage.estimated_cost_usd or 0.0
            model = event.token_usage.model or "unknown"
            log.info(
                "  %d. %s [%s] %d tokens ($%.6f) - %s",
                i,
                source_emoji,
                model,
                tokens,
                cost,
                str(event.message)[:40] + "...",
            )
    else:
        log.info("  No events with token usage")


async def show_tiktoken_status() -> None:
    """Show tiktoken availability and implications."""
    if TIKTOKEN_AVAILABLE:
        log.info("✅ tiktoken is available - using accurate token counting")
        log.info("   • Precise counts for OpenAI models (GPT-3.5, GPT-4, etc.)")
        log.info("   • Falls back to cl100k_base encoding for unknown models")
    else:
        log.warning("⚠️ tiktoken NOT available - using 4-chars≈1-token heuristic")
        log.warning(
            "   • Install tiktoken for accurate token counting: pip install tiktoken"
        )
        log.warning("   • Current estimates may be less precise")


# ── demo 1: single-model conversation ───────────────────────────────
async def create_basic_session() -> Session:
    """Create a basic session with token tracking for a single model."""
    log.info("\n🎯 === Demo 1: Basic Single-Model Session ===")

    sess = await Session.create()
    await sess.metadata.set_property("demo", "basic_token_tracking")
    await sess.metadata.set_property("model", "gpt-4")
    log.info("📝 Created session %s", sess.id[:8] + "...")

    # User question
    user_q = "Hello, can you explain quantum computing in simple terms?"
    user_event = await SessionEvent.create_with_tokens(
        message=user_q,
        prompt=user_q,
        model="gpt-4",
        source=EventSource.USER,
        type=EventType.MESSAGE,
    )
    await sess.add_event_and_save(user_event)

    log.info("👤 User question: %d tokens", user_event.token_usage.total_tokens)

    # Assistant answer with detailed response
    answer = (
        "Quantum computing uses quantum bits (qubits) that can exist in superposition - "
        "being both 0 and 1 simultaneously, unlike classical bits that are strictly one or the other. "
        "This superposition, combined with quantum entanglement, allows quantum computers to "
        "explore many computational paths at once, like trying every route through a maze "
        "simultaneously rather than one at a time. This gives quantum computers potential "
        "exponential speedups for specific problems like cryptography, optimization, and "
        "simulating quantum systems, though they're still in early development stages."
    )

    llm_event = await SessionEvent.create_with_tokens(
        message=answer,
        prompt=user_q,
        completion=answer,
        model="gpt-4",
        source=EventSource.LLM,
        type=EventType.MESSAGE,
    )
    await sess.add_event_and_save(llm_event)

    log.info("🤖 Assistant answer: %d tokens", llm_event.token_usage.total_tokens)
    log.info("💰 Session cost so far: $%.6f", sess.total_cost)

    return sess


# ── demo 2: multi-model usage ───────────────────────────────────────
async def create_multi_model_session() -> Session:
    """Create a session demonstrating multiple model usage with cost comparison."""
    log.info("\n🤖 === Demo 2: Multi-Model Comparison ===")

    sess = await Session.create()
    await sess.metadata.set_property("demo", "multi_model_comparison")
    log.info("📝 Created multi-model session %s", sess.id[:8] + "...")

    # Test different models with the same question
    models_info = [
        ("gpt-4", "Premium OpenAI model with advanced reasoning"),
        ("gpt-3.5-turbo", "Fast and cost-effective OpenAI model"),
        ("claude-3-sonnet", "Anthropic's balanced model for general use"),
    ]

    question = "What are the key advantages of your AI model architecture?"

    for model, description in models_info:
        # User question
        user_event = await SessionEvent.create_with_tokens(
            message=f"[Testing {model}] {question}",
            prompt=f"[Testing {model}] {question}",
            model=model,
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
        await sess.add_event_and_save(user_event)

        # Simulated model response
        response = (
            f"As {model}, I offer {description}. "
            f"My architecture provides efficient processing with strong reasoning capabilities. "
            f"I'm designed for reliability, safety, and helpful responses across diverse tasks. "
            f"Key strengths include natural language understanding, contextual awareness, "
            f"and the ability to engage in nuanced conversations while maintaining accuracy."
        )

        llm_event = await SessionEvent.create_with_tokens(
            message=response,
            prompt=question,
            completion=response,
            model=model,
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
        await sess.add_event_and_save(llm_event)

        tokens = llm_event.token_usage.total_tokens
        cost = llm_event.token_usage.estimated_cost_usd or 0.0
        log.info("  %s: %d tokens, $%.6f", model, tokens, cost)

    log.info("💰 Total multi-model cost: $%.6f", sess.total_cost)
    return sess


# ── demo 3: running-total cost tracker ──────────────────────────────
async def running_cost_demo() -> Session:
    """Demonstrate real-time cost tracking during a conversation."""
    log.info("\n💰 === Demo 3: Real-Time Cost Tracking ===")

    sess = await Session.create()
    await sess.metadata.set_property("demo", "real_time_cost_tracking")
    await sess.metadata.set_property("use_case", "travel_planning")
    log.info("📝 Created cost-tracking session %s", sess.id[:8] + "...")

    # Simulate a realistic conversation with alternating models for cost comparison
    conversation = [
        {
            "role": "user",
            "content": "I'm planning a 3-day trip to Kyoto, Japan. Can you help me create an itinerary?",
            "model": "gpt-4",
        },
        {
            "role": "assistant",
            "content": "I'd be happy to help you plan your Kyoto trip! Here's a suggested 3-day itinerary: Day 1: Explore Eastern Kyoto including Kiyomizu-dera Temple, Sannenzaka and Ninenzaka streets, and Gion district for traditional atmosphere. Day 2: Visit the famous Fushimi Inari Shrine with its thousands of torii gates, then explore Arashiyama Bamboo Grove and Tenryu-ji Temple. Day 3: Discover Northern Kyoto with Kinkaku-ji (Golden Pavilion), Ryoan-ji rock garden, and Nijo Castle. Each day offers a perfect blend of temples, culture, and scenic beauty.",
            "model": "gpt-4",
        },
        {
            "role": "user",
            "content": "That sounds great! Can you add a day trip to Nara to see the deer park?",
            "model": "gpt-3.5-turbo",
        },
        {
            "role": "assistant",
            "content": "Excellent idea! I'll modify your itinerary to include Nara. Here's the updated 4-day plan: Day 1: Kyoto Eastern districts (Kiyomizu-dera, Gion). Day 2: Kyoto Fushimi and Arashiyama areas. Day 3: Day trip to Nara - visit Todai-ji Temple with the giant Buddha statue, explore Nara Park to feed the sacred deer, and see Kasuga Taisha Shrine with its thousands of stone lanterns. Day 4: Northern Kyoto (Kinkaku-ji, Ryoan-ji, Nijo Castle). The Nara day trip is only 45 minutes by train from Kyoto, making it perfect for your extended adventure!",
            "model": "gpt-3.5-turbo",
        },
        {
            "role": "user",
            "content": "What about food recommendations for each location?",
            "model": "claude-3-sonnet",
        },
        {
            "role": "assistant",
            "content": "Great question! Here are food recommendations for each location: Kyoto Eastern: Try kaiseki cuisine at Kikunoi, yudofu (hot tofu) in Arashiyama, and traditional sweets in Gion. For casual dining, visit Ganko Sushi or enjoy matcha and wagashi at historic tea houses. Nara: Sample kakinoha-zushi (persimmon leaf sushi), try local craft beer at Nara Brewing, and don't miss the regional specialty narazuke pickles. Near the deer park, enjoy traditional Japanese breakfast at local ryokan restaurants. Northern Kyoto: Visit the famous Ippudo Ramen near Kinkaku-ji, try Buddhist vegetarian cuisine (shojin ryori) at temple restaurants, and sample local tofu dishes in the Arashiyama area.",
            "model": "claude-3-sonnet",
        },
    ]

    running_total_tokens = 0
    running_total_cost = 0.0

    for i, turn in enumerate(conversation):
        if turn["role"] == "user":
            user_event = await SessionEvent.create_with_tokens(
                message=turn["content"],
                prompt=turn["content"],
                model=turn["model"],
                source=EventSource.USER,
                type=EventType.MESSAGE,
            )
            await sess.add_event_and_save(user_event)

            tokens = user_event.token_usage.total_tokens
            cost = user_event.token_usage.estimated_cost_usd or 0.0
            running_total_tokens += tokens
            running_total_cost += cost

            log.info(
                "👤 User input (%s): %d tokens, $%.6f", turn["model"], tokens, cost
            )

        else:  # assistant
            # Get the previous user message as prompt context
            prompt_context = conversation[i - 1]["content"] if i > 0 else ""

            llm_event = await SessionEvent.create_with_tokens(
                message=turn["content"],
                prompt=prompt_context,
                completion=turn["content"],
                model=turn["model"],
                source=EventSource.LLM,
                type=EventType.MESSAGE,
            )
            await sess.add_event_and_save(llm_event)

            tokens = llm_event.token_usage.total_tokens
            cost = llm_event.token_usage.estimated_cost_usd or 0.0
            running_total_tokens += tokens
            running_total_cost += cost

            log.info(
                "🤖 Assistant (%s): %d tokens, $%.6f → Running total: %d tokens, $%.6f",
                turn["model"],
                tokens,
                cost,
                sess.total_tokens,
                sess.total_cost,
            )

    log.info(
        "💰 Final conversation cost: $%.6f (%d total tokens)",
        sess.total_cost,
        sess.total_tokens,
    )
    return sess


# ── demo 4: token usage analytics ───────────────────────────────────
async def token_analytics_demo() -> Session:
    """Demonstrate advanced token usage analytics and cost optimization insights."""
    log.info("\n📊 === Demo 4: Token Usage Analytics ===")

    sess = await Session.create()
    await sess.metadata.set_property("demo", "token_analytics")
    await sess.metadata.set_property("use_case", "code_review")
    log.info("📝 Created analytics session %s", sess.id[:8] + "...")

    # Simulate a code review conversation with different complexity levels
    scenarios = [
        {
            "name": "Simple Function Review",
            "user_input": "Please review this Python function for calculating factorial.",
            "assistant_response": "The function looks good overall. Consider adding input validation for negative numbers and using iterative approach for better performance with large numbers.",
            "model": "gpt-3.5-turbo",
        },
        {
            "name": "Complex Algorithm Analysis",
            "user_input": "Can you analyze the time complexity of this sorting algorithm implementation and suggest optimizations? The code handles edge cases but might have performance issues with large datasets.",
            "assistant_response": "This sorting algorithm appears to be a modified quicksort with additional complexity. The time complexity is O(n log n) average case, O(n²) worst case. For optimization: 1) Implement median-of-three pivot selection, 2) Use insertion sort for small subarrays (< 10 elements), 3) Consider iterative implementation to avoid stack overflow, 4) Add randomization to avoid worst-case scenarios on sorted data. The edge case handling is well implemented, but you could optimize memory usage by using in-place partitioning more efficiently.",
            "model": "gpt-4",
        },
        {
            "name": "Architecture Design Review",
            "user_input": "Review this microservices architecture design for scalability, security, and maintainability. The system handles user authentication, data processing, and real-time notifications.",
            "assistant_response": "Your microservices architecture shows good separation of concerns. Strengths: Clear service boundaries, appropriate use of event-driven patterns for notifications, and proper authentication service isolation. Areas for improvement: 1) Consider implementing API Gateway for better request routing and rate limiting, 2) Add circuit breaker patterns between services, 3) Implement distributed tracing for better observability, 4) Use message queues for async communication between services, 5) Consider data consistency patterns (SAGA or Event Sourcing) for distributed transactions. Security-wise, ensure service-to-service authentication and implement proper secret management. For scalability, consider horizontal pod autoscaling and database sharding strategies.",
            "model": "gpt-4",
        },
    ]

    total_cost_by_complexity = {}

    for scenario in scenarios:
        log.info(f"\n🔍 Analyzing: {scenario['name']}")

        # User input
        user_event = await SessionEvent.create_with_tokens(
            message=scenario["user_input"],
            prompt=scenario["user_input"],
            model=scenario["model"],
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
        await sess.add_event_and_save(user_event)

        # Assistant response
        llm_event = await SessionEvent.create_with_tokens(
            message=scenario["assistant_response"],
            prompt=scenario["user_input"],
            completion=scenario["assistant_response"],
            model=scenario["model"],
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
        await sess.add_event_and_save(llm_event)

        # Calculate cost for this scenario
        scenario_tokens = (
            user_event.token_usage.total_tokens + llm_event.token_usage.total_tokens
        )
        scenario_cost = (user_event.token_usage.estimated_cost_usd or 0.0) + (
            llm_event.token_usage.estimated_cost_usd or 0.0
        )
        total_cost_by_complexity[scenario["name"]] = {
            "tokens": scenario_tokens,
            "cost": scenario_cost,
            "model": scenario["model"],
        }

        log.info(
            f"  📊 {scenario_tokens} tokens, ${scenario_cost:.6f} ({scenario['model']})"
        )

    # Analytics summary
    log.info("\n📈 Cost Analysis by Complexity:")
    for name, stats in total_cost_by_complexity.items():
        cost_per_token = stats["cost"] / stats["tokens"] if stats["tokens"] > 0 else 0
        log.info(
            f"  {name}: ${stats['cost']:.6f} ({stats['tokens']} tokens, ${cost_per_token:.8f}/token)"
        )

    return sess


# ── main orchestrator ───────────────────────────────────────────────
async def main() -> None:
    """Run all token usage demonstrations."""
    log.info("🚀 Starting comprehensive token-tracking demo")
    await bootstrap_store()
    await show_tiktoken_status()

    # Run all demos
    basic_session = await create_basic_session()
    await token_usage_report(basic_session)

    multi_model_session = await create_multi_model_session()
    await token_usage_report(multi_model_session)

    cost_tracking_session = await running_cost_demo()
    await token_usage_report(cost_tracking_session)

    analytics_session = await token_analytics_demo()
    await token_usage_report(analytics_session)

    # Final summary across all sessions
    all_sessions = [
        basic_session,
        multi_model_session,
        cost_tracking_session,
        analytics_session,
    ]
    total_tokens = sum(s.total_tokens for s in all_sessions)
    total_cost = sum(s.total_cost for s in all_sessions)

    log.info("\n🎉 Demo Complete - Final Summary")
    log.info("=" * 50)
    log.info("📊 Aggregate Statistics:")
    log.info(f"   Sessions created: {len(all_sessions)}")
    log.info(f"   Total tokens: {total_tokens}")
    log.info(f"   Total cost: ${total_cost:.6f}")
    log.info(f"   Average cost per session: ${total_cost / len(all_sessions):.6f}")

    # Model usage summary
    all_models = set()
    for session in all_sessions:
        all_models.update(session.token_summary.usage_by_model.keys())

    log.info(f"   Models tested: {len(all_models)}")
    for model in sorted(all_models):
        model_tokens = sum(
            session.token_summary.usage_by_model.get(model, TokenUsage()).total_tokens
            for session in all_sessions
        )
        model_cost = sum(
            session.token_summary.usage_by_model.get(
                model, TokenUsage()
            ).estimated_cost_usd
            or 0.0
            for session in all_sessions
        )
        if model_tokens > 0:
            log.info(f"     {model}: {model_tokens} tokens, ${model_cost:.6f}")

    log.info("\n✨ Key Features Demonstrated:")
    log.info("   • Real-time token counting and cost tracking")
    log.info("   • Multi-model cost comparison")
    log.info("   • Running total cost monitoring")
    log.info("   • Detailed usage analytics by source and model")
    log.info("   • Production-ready cost optimization insights")


if __name__ == "__main__":
    asyncio.run(main())
