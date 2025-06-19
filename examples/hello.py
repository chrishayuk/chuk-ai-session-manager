# examples/hello.py
import asyncio
from chuk_ai_session_manager import SessionManager

async def main():
    # create a session manager with a system prompt
    sm = SessionManager(
        system_prompt="You are a helpful travel assistant specializing in Japanese culture and tourism. "
                     "Provide detailed, practical advice while being enthusiastic and engaging."
    )
    
    # Track conversation turns
    await sm.user_says("Hello! I'm planning a trip to Japan.")
    await sm.ai_responds("Great! Japan is an amazing destination. What interests you most?")

    # next turn
    await sm.user_says("I love temples and good food!")
    await sm.ai_responds(
        "Perfect! I recommend visiting Kyoto for temples and trying authentic ramen.",
        model="gpt-4o", 
        provider="openai"
    )

    # Get the session id
    print(f"✅ Tracked! Session ID: {sm.session_id}")

    # Show conversation stats
    stats = await sm.get_stats()
    print(f"\n📊 Context-Aware Conversation Stats:")
    print(f"   Messages: {stats['user_messages']} user, {stats['ai_messages']} AI")
    print(f"   Tokens: {stats['total_tokens']}")
    print(f"   Cost: ${stats['estimated_cost']:.6f}")

    # You can also update the system prompt mid-conversation
    await sm.update_system_prompt(
        "You are a budget-conscious travel assistant. Focus on affordable options and money-saving tips."
    )
    
    await sm.user_says("What's the best way to save money on accommodation?")
    await sm.ai_responds(
        "For budget accommodation in Japan, I highly recommend capsule hotels or business hotels!",
        model="gpt-4o-mini"
    )


if __name__ == "__main__":
    asyncio.run(main())