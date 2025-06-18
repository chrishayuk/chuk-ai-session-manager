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
    
    # Step 1: Import the simple API - FIXED import path
    try:
        from chuk_ai_session_manager.api.simple_api import SessionManager
    except ImportError:
        print("❌ Simple API not available. Using core session manager...")
        from chuk_ai_session_manager.models.session import Session
        from chuk_ai_session_manager.models.session_event import SessionEvent
        from chuk_ai_session_manager.models.event_source import EventSource
        from chuk_ai_session_manager.models.event_type import EventType
        from chuk_ai_session_manager.session_storage import setup_chuk_sessions_storage, get_backend, ChukSessionsStore
        
        # Setup storage
        setup_chuk_sessions_storage(sandbox_id="quickstart-demo", default_ttl_hours=1)
        
        # Create session manually
        session = await Session.create()
        await session.metadata.set_property("demo", "quickstart")
        
        backend = get_backend()
        store = ChukSessionsStore(backend)
        await store.save(session)
        
        print(f"✅ Created session: {session.id[:8]}...")
        
        # Add events manually
        user_event1 = await SessionEvent.create_with_tokens(
            message="Hello! What's 2+2?",
            prompt="Hello! What's 2+2?",
            model="gpt-4",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event_and_save(user_event1)
        
        ai_event1 = await SessionEvent.create_with_tokens(
            message="Hello! 2+2 equals 4.",
            prompt="",
            completion="Hello! 2+2 equals 4.",
            model="gpt-4",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        await ai_event1.set_metadata("provider", "openai")
        await session.add_event_and_save(ai_event1)
        
        user_event2 = await SessionEvent.create_with_tokens(
            message="Thanks! What about 5*6?",
            prompt="Thanks! What about 5*6?",
            model="gpt-4",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event_and_save(user_event2)
        
        ai_event2 = await SessionEvent.create_with_tokens(
            message="5*6 equals 30.",
            prompt="",
            completion="5*6 equals 30.",
            model="gpt-4",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        await ai_event2.set_metadata("provider", "openai")
        await session.add_event_and_save(ai_event2)
        
        print("✅ Tracked 2-turn conversation")
        
        # Get fresh session with all events
        session = await store.get(session.id)
        
        # Show conversation
        print("\n💬 Conversation History:")
        message_events = [e for e in session.events if e.type == EventType.MESSAGE]
        for i, event in enumerate(message_events, 1):
            role_emoji = "👤" if event.source == EventSource.USER else "🤖"
            print(f"  {i}. {role_emoji} {event.message}")
        
        # Show stats
        user_messages = sum(1 for e in session.events if e.source == EventSource.USER)
        ai_messages = sum(1 for e in session.events if e.source == EventSource.LLM)
        
        print(f"\n📊 Stats: {user_messages} user, {ai_messages} AI messages")
        print(f"💰 Total tokens: {session.total_tokens}")
        print(f"💰 Estimated cost: ${session.total_cost:.6f}")
        
        # Create a mock SessionManager-like object for return compatibility
        class MockSessionManager:
            def __init__(self, session):
                self.session = session
                self.session_id = session.id
            
            async def get_conversation(self):
                events = [e for e in self.session.events if e.type == EventType.MESSAGE]
                return [
                    {
                        "role": "user" if e.source == EventSource.USER else "assistant",
                        "content": e.message,
                        "timestamp": e.timestamp.isoformat()
                    }
                    for e in events
                ]
            
            async def get_stats(self):
                user_msgs = sum(1 for e in self.session.events if e.source == EventSource.USER)
                ai_msgs = sum(1 for e in self.session.events if e.source == EventSource.LLM)
                return {
                    "session_id": self.session.id,
                    "user_messages": user_msgs,
                    "ai_messages": ai_msgs,
                    "total_tokens": self.session.total_tokens,
                    "estimated_cost": self.session.total_cost
                }
        
        return MockSessionManager(session)
    
    # If simple API is available, use it
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
        
        # Try to use simple API, fall back to core if needed
        try:
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
            
        except ImportError:
            print("Simple API not available, using core session manager...")
            
            # Use core session manager
            from chuk_ai_session_manager.models.session import Session
            from chuk_ai_session_manager.models.session_event import SessionEvent
            from chuk_ai_session_manager.models.event_source import EventSource
            from chuk_ai_session_manager.models.event_type import EventType
            from chuk_ai_session_manager.session_storage import setup_chuk_sessions_storage, get_backend, ChukSessionsStore
            
            setup_chuk_sessions_storage(sandbox_id="quickstart-llm-demo", default_ttl_hours=1)
            
            client = AsyncOpenAI()
            session = await Session.create()
            
            backend = get_backend()
            store = ChukSessionsStore(backend)
            await store.save(session)
            
            user_input = "What's the capital of France? One word answer."
            
            # Add user event
            user_event = await SessionEvent.create_with_tokens(
                message=user_input,
                prompt=user_input,
                model="gpt-3.5-turbo",
                source=EventSource.USER,
                type=EventType.MESSAGE
            )
            await session.add_event_and_save(user_event)
            
            # LLM call
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_input}],
                max_tokens=10
            )
            
            ai_response = response.choices[0].message.content
            
            # Add AI event
            ai_event = await SessionEvent.create_with_tokens(
                message=ai_response,
                prompt="",
                completion=ai_response,
                model="gpt-3.5-turbo",
                source=EventSource.LLM,
                type=EventType.MESSAGE
            )
            await ai_event.set_metadata("provider", "openai")
            await session.add_event_and_save(ai_event)
            
            print(f"👤 {user_input}")
            print(f"🤖 {ai_response}")
            
            return session
    
    # Option 2: With chuk_llm (if available)
    elif os.getenv("CHUK_LLM_AVAILABLE"):  # Placeholder check
        print("Trying chuk_llm integration...")
        
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
            print("chuk_llm not available. Using mock example...")
            await _mock_demo()
            
    else:
        print("No API keys found. Using mock example...")
        await _mock_demo()


async def _mock_demo():
    """Mock demo when no real LLM is available."""
    try:
        from chuk_ai_session_manager.simple_api import SessionManager
        
        sm = SessionManager()
        await sm.user_says("What's 10*10?")
        await sm.ai_responds("10*10 equals 100.", model="mock", provider="demo")
        
        print("👤 What's 10*10?")
        print("🤖 10*10 equals 100.")
        
        return sm
        
    except ImportError:
        print("Using core session manager for mock demo...")
        
        from chuk_ai_session_manager.models.session import Session
        from chuk_ai_session_manager.models.session_event import SessionEvent
        from chuk_ai_session_manager.models.event_source import EventSource
        from chuk_ai_session_manager.models.event_type import EventType
        from chuk_ai_session_manager.session_storage import setup_chuk_sessions_storage, get_backend, ChukSessionsStore
        
        setup_chuk_sessions_storage(sandbox_id="mock-demo", default_ttl_hours=1)
        
        session = await Session.create()
        backend = get_backend()
        store = ChukSessionsStore(backend)
        await store.save(session)
        
        # Add mock conversation
        user_event = await SessionEvent.create_with_tokens(
            message="What's 10*10?",
            prompt="What's 10*10?",
            model="mock",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event_and_save(user_event)
        
        ai_event = await SessionEvent.create_with_tokens(
            message="10*10 equals 100.",
            prompt="",
            completion="10*10 equals 100.",
            model="mock",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        await ai_event.set_metadata("provider", "demo")
        await session.add_event_and_save(ai_event)
        
        print("👤 What's 10*10?")
        print("🤖 10*10 equals 100.")
        
        return session


async def show_integration_patterns():
    """Show different integration patterns."""
    
    print("\n🔧 Integration Patterns")
    print("=" * 40)
    
    print("Pattern 1: Simple API (if available)")
    print("   from chuk_ai_session_manager.simple_api import SessionManager")
    print("   sm = SessionManager()")
    print("   await sm.user_says('Hello')")
    print("   response = await your_llm_function('Hello')")
    print("   await sm.ai_responds(response)")
    
    print("\nPattern 2: Core Session Manager")
    print("   from chuk_ai_session_manager.models.session import Session")
    print("   from chuk_ai_session_manager.models.session_event import SessionEvent")
    print("   session = await Session.create()")
    print("   event = await SessionEvent.create_with_tokens(...)")
    print("   await session.add_event_and_save(event)")
    
    print("\nPattern 3: Automatic tracking (if simple API available)")
    print("   from chuk_ai_session_manager.simple_api import track_llm_call")
    print("   response, session_id = await track_llm_call(")
    print("       user_input='Hello',")
    print("       llm_function=your_llm_function")
    print("   )")
    
    print("\nPattern 4: Tool tracking")
    print("   # With simple API:")
    print("   await sm.tool_called('weather', {'city': 'Tokyo'}, result)")
    print("   # With core API:")
    print("   tool_event = SessionEvent(..., type=EventType.TOOL_CALL)")
    
    print("\nPattern 5: Analytics")
    print("   # Session statistics")
    print("   print(f'Tokens: {session.total_tokens}')")
    print("   print(f'Cost: ${session.total_cost:.6f}')")
    print("   # Event breakdown")
    print("   usage_by_source = await session.get_token_usage_by_source()")


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
    print("1. Check out examples/ for more comprehensive demos")
    print("2. Add session tracking to your existing LLM code")
    print("3. Use session.total_tokens and session.total_cost for monitoring")
    print("4. Explore FastAPI example for building APIs with session management")
    
    print("\n📚 Key Components:")
    print("   Session - main conversation container")
    print("   SessionEvent - individual interactions")
    print("   CHUK Sessions - enterprise storage backend")
    print("   SessionManager - simple API wrapper (if available)")
    
    print("\n🚀 Production Features:")
    print("   • Complete conversation tracking")
    print("   • Real-time token and cost analytics")
    print("   • Hierarchical session relationships")
    print("   • Enterprise storage with Redis support")
    print("   • Tool execution tracking")
    print("   • Multi-provider LLM support")


if __name__ == "__main__":
    asyncio.run(main())