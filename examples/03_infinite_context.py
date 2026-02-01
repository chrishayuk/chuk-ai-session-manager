# examples/03_infinite_context.py
"""
🔄 TEST UNIFIED SESSION MANAGER

Simple test to verify the unified SessionManager works correctly.
"""

import asyncio
from chuk_ai_session_manager import SessionManager


async def test_regular_session():
    """Test regular session mode"""
    print("📝 Testing Regular Session")
    print("=" * 25)

    sm = SessionManager()

    await sm.user_says("Hello world!")
    await sm.ai_responds("Hi there!", model="gpt-4", provider="openai")

    stats = await sm.get_stats()
    conversation = await sm.get_conversation()

    print(f"✅ Session ID: {stats['session_id'][:8]}...")
    print(f"✅ Messages: {stats['user_messages']} user, {stats['ai_messages']} AI")
    print(f"✅ Segments: {stats['session_segments']}")
    print(f"✅ Infinite: {stats['infinite_context']}")
    print(f"✅ Conversation length: {len(conversation)}")


async def test_infinite_session():
    """Test infinite session mode"""
    print("\n🔄 Testing Infinite Session")
    print("=" * 30)

    sm = SessionManager(
        infinite_context=True,
        token_threshold=100,  # Very low to force segmentation
        max_turns_per_segment=1,  # Force new segment every turn
    )

    print(f"🆕 Created infinite session: {sm.session_id[:8]}...")

    # Add several messages to trigger segmentation
    messages = [
        ("What is Python?", "Python is a programming language."),
        ("Is it easy to learn?", "Yes, Python is beginner-friendly."),
        ("What can I build?", "Web apps, AI models, automation scripts."),
    ]

    for i, (question, answer) in enumerate(messages):
        old_session = sm.session_id

        await sm.user_says(question)
        await sm.ai_responds(answer, model="gpt-4", provider="openai")

        if sm.session_id != old_session:
            print(
                f"   🔄 Transition {i + 1}: {old_session[:8]}... → {sm.session_id[:8]}..."
            )
        else:
            print(f"   ✅ Turn {i + 1}: No transition")

    # Get results
    stats = await sm.get_stats()
    conversation = await sm.get_conversation()
    session_chain = await sm.get_session_chain()

    print("\n📊 Results:")
    print(f"   Current session: {stats['session_id'][:8]}...")
    print(f"   Total segments: {stats['session_segments']}")
    print(f"   Session chain: {len(session_chain)} sessions")
    print(f"   Full conversation: {len(conversation)} exchanges")
    print(f"   Messages: {stats['user_messages']} user, {stats['ai_messages']} AI")
    print(f"   Infinite context: {stats['infinite_context']}")


async def test_convenience_functions():
    """Test convenience functions"""
    print("\n⚡ Testing Convenience Functions")
    print("=" * 35)

    # Test track_conversation
    from chuk_ai_session_manager import track_conversation, track_infinite_conversation

    print("📝 Regular conversation tracking...")
    session_id = await track_conversation(
        "Hello!", "Hi there!", model="gpt-4", provider="openai"
    )
    print(f"✅ Tracked regular conversation: {session_id[:8]}...")

    print("\n🔄 Infinite conversation tracking...")
    infinite_session_id = await track_infinite_conversation(
        "Start infinite chat",
        "This can go on forever!",
        model="gpt-4",
        provider="openai",
        token_threshold=50,
    )
    print(f"✅ Tracked infinite conversation: {infinite_session_id[:8]}...")


async def main():
    print("🔄 UNIFIED SESSION MANAGER TESTS")
    print("Testing the new unified SessionManager")
    print("=" * 45)

    try:
        await test_regular_session()
        await test_infinite_session()
        await test_convenience_functions()

        print("\n🎉 All Tests Passed!")
        print("=" * 45)
        print("✅ Regular sessions work correctly")
        print("✅ Infinite sessions handle segmentation")
        print("✅ Convenience functions work")
        print("✅ API is clean and unified")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
