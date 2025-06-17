#!/usr/bin/env python3
"""
CHUK LLM Simple Functions + Session Manager Integration

This example shows how to use CHUK LLM's simple convenience functions
while still tracking everything in sessions.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CHUK LLM simple functions
from chuk_llm import (
    # Async functions (recommended)
    ask_openai_gpt4o_mini,
    ask_anthropic_claude_sonnet4_20250514,
    ask_anthropic_sonnet,
    ask_groq_llama,
    ask_mistral_medium,
    ask_deepseek_reasoner,
    
    # Sync functions (if needed)
    ask_sync,
    ask_openai_sync,
    ask_anthropic_sync,
)

# Session manager imports
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session import Session, SessionEvent
from chuk_ai_session_manager.storage import SessionStoreProvider
from chuk_ai_session_manager.storage.providers.memory import InMemorySessionStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Quiet down ALL the noisy loggers
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("anthropic").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("h11").setLevel(logging.ERROR)
logging.getLogger("chuk_llm").setLevel(logging.WARNING)  # Keep some chuk_llm logs but reduce them


class TrackedLLM:
    """
    A wrapper that adds session tracking to CHUK LLM simple functions.
    
    This approach is simpler than the conversation API but still provides
    complete session tracking capabilities.
    """
    
    def __init__(self, session: Session):
        self.session = session
    
    async def _track_interaction(self, user_message: str, response: str, 
                               provider: str, model: str, metadata: Optional[Dict[str, Any]] = None):
        """Track a complete LLM interaction in the session."""
        base_metadata = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model
        }
        if metadata:
            base_metadata.update(metadata)
        
        # Track user message
        await self.session.add_event_and_save(SessionEvent(
            message=user_message,
            source=EventSource.USER,
            metadata=base_metadata
        ))
        
        # Track assistant response
        await self.session.add_event_and_save(SessionEvent(
            message=response,
            source=EventSource.LLM,
            metadata=base_metadata
        ))
        
        logger.debug(f"💾 Tracked {provider}/{model} interaction")
    
    # OpenAI functions with tracking
    async def ask_openai_gpt4o_mini(self, question: str, **kwargs) -> str:
        """Ask OpenAI GPT-4o Mini with session tracking."""
        response = await ask_openai_gpt4o_mini(question, **kwargs)
        await self._track_interaction(question, response, "openai", "gpt-4o-mini")
        return response
    
    # Anthropic functions with tracking
    async def ask_anthropic_sonnet(self, question: str, **kwargs) -> str:
        """Ask Anthropic Claude Sonnet with session tracking."""
        response = await ask_anthropic_sonnet(question, **kwargs)
        await self._track_interaction(question, response, "anthropic", "claude-sonnet-4")
        return response
    
    async def ask_anthropic_claude_sonnet4(self, question: str, **kwargs) -> str:
        """Ask Anthropic Claude Sonnet 4 with session tracking."""
        response = await ask_anthropic_claude_sonnet4_20250514(question, **kwargs)
        await self._track_interaction(question, response, "anthropic", "claude-sonnet-4-20250514")
        return response
    
    # Multi-provider convenience method
    async def ask_parallel(self, question: str, providers: list = None) -> Dict[str, str]:
        """Ask multiple providers in parallel and track all responses."""
        if providers is None:
            providers = ["openai", "anthropic"]
        
        tasks = []
        for provider in providers:
            if provider == "openai":
                tasks.append(("openai", self.ask_openai_gpt4o_mini(question)))
            elif provider == "anthropic":
                tasks.append(("anthropic", self.ask_anthropic_sonnet(question)))
        
        # Execute in parallel
        results = {}
        for provider, task in tasks:
            try:
                results[provider] = await task
            except Exception as e:
                logger.error(f"❌ Error with {provider}: {e}")
                results[provider] = f"Error: {str(e)}"
        
        return results
    
    # Sync function wrapper (for compatibility)
    def ask_sync_tracked(self, question: str, provider: str = "openai") -> str:
        """Synchronous ask with tracking (creates new event loop if needed)."""
        if provider == "openai":
            response = ask_openai_sync(question)
        elif provider == "anthropic":
            response = ask_anthropic_sync(question)
        else:
            response = ask_sync(question)
        
        # Track in background (requires running event loop)
        asyncio.create_task(
            self._track_interaction(question, response, provider, "default")
        )
        return response


async def demonstrate_simple_tracking():
    """Demonstrate session tracking with simple functions."""
    print("🚀 CHUK LLM Simple Functions + Session Tracking")
    print("="*60)
    
    # Initialize session store and set as provider
    store = InMemorySessionStore()
    SessionStoreProvider.set_store(store)
    
    # Create session using the proper API
    session = await Session.create(metadata={
        "example": "simple_functions_tracking",
        "description": "Using CHUK LLM simple functions with session tracking"
    })
    print(f"📝 Created session: {session.id}")
    
    # Create tracked LLM wrapper
    llm = TrackedLLM(session)
    
    print("\n💬 Single Provider Interactions:")
    print("-" * 40)
    
    # OpenAI interaction
    print("🤖 Asking OpenAI GPT-4o Mini...")
    response = await llm.ask_openai_gpt4o_mini("What's 2+2? Explain briefly.")
    print(f"Response: {response}")
    
    # Anthropic interaction
    print("\n🧠 Asking Anthropic Claude Sonnet...")
    response = await llm.ask_anthropic_sonnet("What's 3+3? Be concise.")
    print(f"Response: {response}")
    
    print("\n⚡ Parallel Provider Interactions:")
    print("-" * 40)
    
    # Parallel interactions
    print("🔄 Asking both providers simultaneously...")
    results = await llm.ask_parallel(
        "What's the capital of France? One word answer.",
        providers=["openai", "anthropic"]
    )
    
    for provider, response in results.items():
        print(f"{provider.capitalize()}: {response}")
    
    # Show session tracking results
    print("\n📊 Session Tracking Results:")
    print("-" * 40)
    
    print(f"📋 Session ID: {session.id}")
    print(f"📅 Created: {session.metadata.created_at}")
    print(f"🔄 Updated: {session.metadata.updated_at}")
    print(f"📝 Total Events: {len(session.events)}")
    
    print(f"\n🗃️ Event History:")
    for i, event in enumerate(session.events, 1):
        timestamp = event.metadata.get('timestamp', 'Unknown') if event.metadata else 'Unknown'
        provider = event.metadata.get('provider', 'Unknown') if event.metadata else 'Unknown'
        model = event.metadata.get('model', 'Unknown') if event.metadata else 'Unknown'
        
        print(f"  {i}. [{event.source.value}] {provider}/{model}")
        print(f"     Content: {str(event.message)[:60]}...")
        print(f"     Time: {timestamp}")
        print()


async def demonstrate_conversation_vs_simple():
    """Compare conversation API vs simple functions approach."""
    print("\n🔄 Conversation API vs Simple Functions Comparison")
    print("="*60)
    
    # Simple functions approach
    print("📌 Simple Functions (Stateless):")
    print("✅ Pros: Easy to use, no context management, great for one-offs")
    print("🔶 Trade-offs: No conversation memory, each call is independent")
    
    response1 = await ask_openai_gpt4o_mini("My name is Alice")
    print(f"Call 1: {response1}")
    
    response2 = await ask_openai_gpt4o_mini("What's my name?")
    print(f"Call 2: {response2}")  # Won't remember Alice
    
    print("\n📌 Conversation API (Stateful):")
    print("✅ Pros: Maintains context, natural conversation flow")
    print("🔶 Trade-offs: More setup, need to manage conversation lifecycle")
    
    try:
        # Use the correct conversation API import
        from chuk_llm.api.conversation import conversation
        
        async with conversation(provider="openai", model="gpt-4o-mini") as conv:
            response1 = await conv.say("My name is Alice")
            print(f"Call 1: {response1}")
            
            response2 = await conv.say("What's my name?")
            print(f"Call 2: {response2}")  # Will remember Alice!
            
    except Exception as e:
        print(f"⚠️ Conversation API demo failed: {e}")
        print("💡 Using fallback demonstration instead...")
        
        # Fallback: show context preservation with a single conversation simulation
        print("📝 Simulating conversational context:")
        context_prompt = """Previous conversation:
User: My name is Alice
Assistant: Nice to meet you, Alice!

Current question: What's my name?"""
        
        response = await ask_openai_gpt4o_mini(context_prompt)
        print(f"Simulated context-aware response: {response}")


async def main():
    """Main demonstration."""
    try:
        await demonstrate_simple_tracking()
        await demonstrate_conversation_vs_simple()
        
        print("\n✅ DEMONSTRATION COMPLETE")
        print("="*60)
        print("🎯 Key Takeaways:")
        print("  • Simple functions: Great for stateless interactions")
        print("  • Conversation API: Better for multi-turn conversations") 
        print("  • Session tracking: Works with both approaches")
        print("  • Parallel requests: Easy with simple functions")
        print("  • Choose based on your use case!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())