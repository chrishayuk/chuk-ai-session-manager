#!/usr/bin/env python3
# examples/quickstart.py
"""
Quickstart: Session Manager in 5 minutes

This script shows the absolute basics of using the session manager
with any LLM in just a few lines of code.

Run with: python quickstart.py
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def quickstart_demo():
    """5-minute demo of session tracking."""
    
    print("🚀 Session Manager Quickstart")
    print("=" * 40)
    
    # Step 1: Import the simple API
    from chuk_ai_session_manager.api.simple_api import SessionManager
    
    # Step 2: Create a session manager
    sm = SessionManager()
    print(f"✅ Created session: {sm.session_id[:8]}...")
    
    # Step 3: Track a conversation
    await sm.user_says("Hello! What's 2+2?")
    await sm.ai_responds("Hello! 2+2 equals 4.", model="gpt-4", provider="openai")
    
    await sm.user_says("Thanks! What about 5*6?")
    await sm.ai_responds("5*6 equals 30.", model="gpt-4", provider="openai")
    
    print("✅ Tracked 2-turn conversation")
    
    # Step 4: Get the conversation back
    conversation = await sm.get_conversation()
    
    print("\n💬 Conversation History:")
    for i, turn in enumerate(conversation, 1):
        role_emoji = "👤" if turn["role"] == "user" else "🤖"
        print(f"  {i}. {role_emoji} {turn['content']}")
    
    # Step 5: Get stats
    stats = await sm.get_stats()
    print(f"\n📊 Stats: {stats['user_messages']} user, {stats['ai_messages']} AI messages")
    
    return sm


async def quickstart_with_real_llm():
    """Quickstart with a real LLM call."""
    
    print("\n🤖 Real LLM Integration")
    print("=" * 40)
    
    # Option 1: With OpenAI directly
    if os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI...")
        
        from openai import AsyncOpenAI
        from chuk_ai_session_manager.api.simple_api import SessionManager
        
        client = AsyncOpenAI()
        sm = SessionManager()
        
        # User input
        user_input = "What's the capital of France? One word answer."
        await sm.user_says(user_input)
        
        # LLM call
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}],
            max_tokens=10
        )
        
        ai_response = response.choices[0].message.content
        await sm.ai_responds(ai_response, model="gpt-3.5-turbo", provider="openai")
        
        print(f"👤 {user_input}")
        print(f"🤖 {ai_response}")
        
        return sm
    
    # Option 2: With chuk_llm (if available)
    else:
        print("No OpenAI API key found.")
        
        try:
            from chuk_llm import ask_openai_gpt4o_mini
            from chuk_ai_session_manager.simple_api import track_llm_call
            
            # This is the EASIEST way - one function call tracks everything
            ai_response, session_id = await track_llm_call(
                user_input="What's 10*10?",
                llm_function=ask_openai_gpt4o_mini,
                model="gpt-4o-mini", 
                provider="openai"
            )
            
            print(f"👤 What's 10*10?")
            print(f"🤖 {ai_response}")
            print(f"📝 Session: {session_id[:8]}...")
            
        except ImportError:
            print("chuk_llm not available either. Using mock example...")
            
            # Mock example for demo
            from chuk_ai_session_manager.api.simple_api import SessionManager
            
            sm = SessionManager()
            await sm.user_says("What's 10*10?")
            await sm.ai_responds("10*10 equals 100.", model="mock", provider="demo")
            
            print("👤 What's 10*10?")
            print("🤖 10*10 equals 100.")
            
            return sm


async def show_integration_patterns():
    """Show different integration patterns."""
    
    print("\n🔧 Integration Patterns")
    print("=" * 40)
    
    from chuk_ai_session_manager.api.simple_api import SessionManager, track_llm_call
    
    print("Pattern 1: Manual tracking")
    print("   sm = SessionManager()")
    print("   await sm.user_says('Hello')")
    print("   response = await your_llm_function('Hello')")
    print("   await sm.ai_responds(response)")
    
    print("\nPattern 2: Automatic tracking")
    print("   response, session_id = await track_llm_call(")
    print("       user_input='Hello',")
    print("       llm_function=your_llm_function")
    print("   )")
    
    print("\nPattern 3: Tool tracking")
    print("   await sm.tool_called('weather', {'city': 'Tokyo'}, result)")
    
    print("\nPattern 4: Get conversation")
    print("   history = await sm.get_conversation()")
    print("   stats = await sm.get_stats()")


async def main():
    """Run the quickstart demo."""
    
    # Basic demo with mock data
    await quickstart_demo()
    
    # Try with real LLM
    await quickstart_with_real_llm()
    
    # Show patterns
    await show_integration_patterns()
    
    print("\n🎉 Quickstart Complete!")
    print("\nNext steps:")
    print("1. Check out examples/simple_api_examples.py for more patterns")
    print("2. Add session tracking to your existing LLM code")
    print("3. Use sm.get_conversation() to build conversation context")
    print("4. Use sm.get_stats() for monitoring and analytics")
    
    print("\n📚 Key classes:")
    print("   SessionManager - main class")
    print("   track_llm_call() - automatic wrapper") 
    print("   quick_conversation() - one-liner helper")


if __name__ == "__main__":
    asyncio.run(main())