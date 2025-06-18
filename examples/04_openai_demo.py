# examples/04_openai_demo.py
"""
🤖 REAL OPENAI DEMO

Super simple demo using actual OpenAI API with session tracking.

Setup:
    1. Create .env file in project root with:
       OPENAI_API_KEY=your-api-key-here
    
    2. Or export environment variable:
       export OPENAI_API_KEY="your-api-key-here"

Run:
    uv run examples/quick_start/10_real_openai_demo.py
"""
import asyncio
import os
from dotenv import load_dotenv
from chuk_ai_session_manager import SessionManager

# Load environment variables from .env file
load_dotenv()

# Check for OpenAI setup
try:
    from openai import AsyncOpenAI
except ImportError:
    print("❌ OpenAI library not installed. Install with: pip install openai")
    exit(1)

async def get_openai_client():
    """Get configured OpenAI client"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        print("\n📋 Setup Instructions:")
        print("   1. Create a .env file in your project root")
        print("   2. Add this line: OPENAI_API_KEY=your-api-key-here")
        print("   3. Get your API key from: https://platform.openai.com/api-keys")
        return None
    
    return AsyncOpenAI(api_key=api_key)

# Simple track_llm_call function that handles OpenAI objects properly
async def track_openai_call(user_input, openai_client, session_manager=None):
    """
    Track an OpenAI call with proper response handling.
    
    Args:
        user_input: The user's question
        openai_client: AsyncOpenAI client
        session_manager: SessionManager instance (creates new if None)
    
    Returns:
        Tuple of (response_text, session_id)
    """
    if session_manager is None:
        session_manager = SessionManager()
    
    # Track user input
    await session_manager.user_says(user_input)
    
    # Call OpenAI
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_input}],
        max_tokens=100
    )
    
    # Extract text from OpenAI response object
    response_text = response.choices[0].message.content
    
    # Track AI response
    session_id = await session_manager.ai_responds(
        response_text, 
        model="gpt-4o-mini", 
        provider="openai"
    )
    
    return response_text, session_id

async def demo_auto_tracking():
    """Auto-tracking OpenAI calls"""
    print("⚡ Auto-Tracking OpenAI Calls")
    print("=" * 30)
    
    client = await get_openai_client()
    if not client:
        return
    
    question = "Explain quantum computing in one sentence."
    print(f"❓ Question: {question}")
    print("🔄 Auto-tracking OpenAI call...")
    
    # Use our track function
    response, session_id = await track_openai_call(question, client)
    
    print(f"✅ Response: {response}")
    print(f"📝 Session: {session_id[:8]}...")

async def demo_conversation_with_context():
    """Show how conversation context works with session tracking"""
    print("\n💬 Conversation Context Demo")
    print("=" * 30)
    
    client = await get_openai_client()
    if not client:
        return
    
    # Create session manager
    sm = SessionManager()
    
    # Questions that build on each other
    questions = [
        "I'm learning Python programming.",
        "What are the key concepts I should focus on first?",
        "Can you give me a simple example?"
    ]
    
    print("🔄 Having a context-building conversation...")
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Turn {i} ---")
        print(f"👤 User: {question}")
        
        # For realistic context, include conversation history in the prompt
        if i == 1:
            # First question - just ask directly
            full_prompt = question
        else:
            # Build context from session
            conversation = await sm.get_conversation()
            
            # Create context-aware prompt
            context = "\n".join([
                f"{turn['role']}: {turn['content']}" 
                for turn in conversation[-4:]  # Last 4 messages for context
            ])
            full_prompt = f"Context:\n{context}\n\nUser: {question}"
        
        # Track the user's actual question (not the full prompt)
        await sm.user_says(question)
        
        # Call OpenAI with context
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=150
        )
        
        response_text = response.choices[0].message.content
        
        # Track the response
        await sm.ai_responds(response_text, model="gpt-4o-mini", provider="openai")
        
        print(f"🤖 AI: {response_text}")
    
    # Show results
    stats = await sm.get_stats()
    print(f"\n📊 Context-Aware Conversation Results:")
    print(f"   Messages: {stats['user_messages']} user, {stats['ai_messages']} AI")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Total cost: ${stats['estimated_cost']:.6f}")

async def demo_infinite_with_context():
    """Infinite context with conversation continuity"""
    print("\n🔄 Infinite Context with Continuity")
    print("=" * 35)
    
    client = await get_openai_client()
    if not client:
        return
    
    # Create infinite session manager
    sm = SessionManager(
        infinite_context=True,
        token_threshold=300,  # Higher threshold for more realistic segmentation
        max_turns_per_segment=2
    )
    
    # A natural conversation flow
    conversation_flow = [
        "I want to build a web application. What technology stack would you recommend?",
        "I'm most interested in Python for the backend. What framework should I use?",
        "What about the frontend? Should I use React or something else?",
        "How do I handle user authentication in this setup?",
        "What database would work best for this kind of application?"
    ]
    
    print("🔄 Extended technical discussion...")
    
    for i, question in enumerate(conversation_flow, 1):
        old_session = sm.session_id
        
        print(f"\n--- Turn {i} ---")
        print(f"👤 Question: {question}")
        
        # Build context from full conversation so far
        if i > 1:
            conversation = await sm.get_conversation()
            context_messages = []
            
            # Add conversation history as context
            for turn in conversation[-6:]:  # Last 6 messages for context
                context_messages.append({
                    "role": turn["role"], 
                    "content": turn["content"]
                })
            
            # Add current question
            context_messages.append({"role": "user", "content": question})
        else:
            context_messages = [{"role": "user", "content": question}]
        
        # Track user question
        await sm.user_says(question)
        
        # Get contextual response from OpenAI
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context_messages,
            max_tokens=200,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content
        
        # Track AI response
        await sm.ai_responds(response_text, model="gpt-4o-mini", provider="openai")
        
        if sm.session_id != old_session:
            print(f"   🔄 TRANSITION: {old_session[:8]}... → {sm.session_id[:8]}...")
        
        print(f"🤖 AI: {response_text[:100]}...")
    
    # Show infinite context results
    stats = await sm.get_stats()
    session_chain = await sm.get_session_chain()
    full_conversation = await sm.get_conversation()
    
    print(f"\n📊 Infinite Context Results:")
    print(f"   Technical questions: {len(conversation_flow)}")
    print(f"   Session segments: {stats['session_segments']}")
    print(f"   Session chain: {' → '.join([sid[:8] + '...' for sid in session_chain])}")
    print(f"   Full conversation: {len(full_conversation)} exchanges")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Total cost: ${stats['estimated_cost']:.6f}")
    print(f"   🧠 Context maintained across {len(session_chain)} sessions!")

async def main():
    print("🤖 REAL OPENAI API DEMO")
    print("OpenAI integration with session tracking and conversation context")
    print("=" * 65)
    
    # Check API key first
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No OpenAI API key found!")
        print("\n📋 Setup Instructions:")
        print("   1. Create a .env file in your project root")
        print("   2. Add this line: OPENAI_API_KEY=your-api-key-here")
        print("   3. Get your API key from: https://platform.openai.com/api-keys")
        return
    
    try:
        await demo_auto_tracking()
        await demo_conversation_with_context()
        await demo_infinite_with_context()
        
        print(f"\n🎉 All OpenAI Demos Complete!")
        print("=" * 65)
        print("✅ Auto-tracking with clean response handling")
        print("✅ Context-aware conversations")
        print("✅ Infinite context with conversation continuity")
        print("✅ Complete session tracking and cost monitoring")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("   Check your OpenAI API key and internet connection")

if __name__ == "__main__":
    asyncio.run(main())