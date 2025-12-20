# examples/02_session_manager.py
"""
📊 SESSION MANAGER: Persistent Conversation Tracking

Track multi-turn conversations with full observability.
"""

import asyncio
from chuk_ai_session_manager import SessionManager


async def main():
    print("💬 Building a conversation...")

    # Create a session manager
    sm = SessionManager()

    # Track conversation turns
    await sm.user_says("Hello! I'm planning a trip to Japan.")
    await sm.ai_responds(
        "Great! Japan is an amazing destination. What interests you most?",
        model="gpt-4o",
        provider="openai",
    )

    await sm.user_says("I love temples and good food!")
    await sm.ai_responds(
        "Perfect! I recommend visiting Kyoto for temples and trying authentic ramen.",
        model="gpt-4o",
        provider="openai",
    )

    # Track a tool call
    await sm.tool_used(
        tool_name="get_weather",
        arguments={"location": "Kyoto"},
        result={"temperature": 22, "condition": "Sunny"},
    )

    # Get conversation summary
    conversation = await sm.get_conversation()
    stats = await sm.get_stats()

    print("\n📈 Conversation Stats:")
    print(f"  Messages: {stats['user_messages']} user, {stats['ai_messages']} AI")
    print(f"  Tools used: {stats['tool_calls']}")
    print(f"  Tokens: {stats['total_tokens']}")
    print(f"  Cost: ${stats['estimated_cost']:.6f}")

    print("\n💬 Conversation History:")
    for turn in conversation:
        print(f"  {turn['role']}: {turn['content'][:50]}...")


if __name__ == "__main__":
    asyncio.run(main())
