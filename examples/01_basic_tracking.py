# examples/01_basic_tracking.py
"""
🚀 1-MINUTE QUICKSTART: Basic Conversation Tracking

Track any AI conversation in 3 lines of code.
"""

import asyncio
from chuk_ai_session_manager import track_conversation


async def main():
    print("📝 Tracking a simple conversation...")

    # Track any conversation in one line
    session_id = await track_conversation(
        user_message="What's the capital of France?",
        ai_response="The capital of France is Paris.",
        model="gpt-4",
        provider="openai",
    )

    print(f"✅ Tracked! Session ID: {session_id}")
    print("💡 Your conversation is now tracked with full observability!")


if __name__ == "__main__":
    asyncio.run(main())
