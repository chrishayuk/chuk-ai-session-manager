# examples/system_prompt_demo.py
"""
Demonstrates how to use system prompts with the SessionManager.

System prompts help set the context and behavior for AI assistants,
ensuring consistent responses aligned with your application's needs.
"""

import asyncio
from chuk_ai_session_manager import SessionManager

async def customer_support_demo():
    """Demo: Customer support assistant with specific guidelines."""
    print("=== Customer Support Assistant Demo ===\n")
    
    # Create a session with a detailed system prompt
    sm = SessionManager(
        system_prompt="""You are a professional customer support assistant for TechCorp.
        
        Guidelines:
        - Be polite, empathetic, and solution-oriented
        - Always acknowledge the customer's concern first
        - Provide clear, step-by-step solutions
        - If you can't solve an issue, offer to escalate to a human agent
        - Keep responses concise but complete
        - Use the customer's name when provided
        """
    )
    
    # Simulate a support conversation
    await sm.user_says("Hi, my laptop won't turn on and I have a presentation in 2 hours!")
    await sm.ai_responds(
        "I understand how stressful this must be with your presentation coming up. "
        "Let me help you troubleshoot this quickly. First, let's check if it's a power issue - "
        "is your laptop plugged in and do you see any lights on the power adapter?",
        model="gpt-4o-mini"
    )
    
    await sm.user_says("Yes, the adapter light is on but nothing happens when I press power.")
    await sm.ai_responds(
        "Thank you for checking that. Let's try a power reset: "
        "1. Unplug the power adapter "
        "2. Remove the battery if possible "
        "3. Hold the power button for 30 seconds "
        "4. Reconnect everything and try again. "
        "This often resolves power-related issues.",
        model="gpt-4o-mini"
    )
    
    # Get the formatted messages including system prompt
    messages = await sm.get_messages_for_llm()
    print(f"Total messages (including system): {len(messages)}")
    print(f"Session ID: {sm.session_id}\n")


async def multi_language_demo():
    """Demo: Multi-language assistant that adapts based on context."""
    print("=== Multi-Language Assistant Demo ===\n")
    
    # Start with English system prompt
    sm = SessionManager(
        system_prompt="You are a helpful assistant. Respond in the same language as the user."
    )
    
    # English interaction
    await sm.user_says("What's the weather like today?")
    await sm.ai_responds(
        "I don't have access to real-time weather data, but I'd be happy to help you "
        "find weather information for your location if you tell me where you are!",
        model="gpt-4o-mini"
    )
    
    # Update system prompt for Spanish context
    await sm.update_system_prompt(
        "Eres un asistente útil. Responde en español con un tono amigable y profesional. "
        "Usa 'usted' para mantener formalidad."
    )
    
    # Spanish interaction
    await sm.user_says("¿Cuál es la capital de Francia?")
    await sm.ai_responds(
        "La capital de Francia es París. Es una hermosa ciudad conocida por "
        "la Torre Eiffel, el Louvre y su rica cultura.",
        model="gpt-4o-mini"
    )
    
    stats = await sm.get_stats()
    print(f"Conversation stats: {stats['total_messages']} messages, {stats['total_tokens']} tokens\n")


async def role_playing_demo():
    """Demo: Role-playing scenarios with dynamic prompt updates."""
    print("=== Role-Playing Assistant Demo ===\n")
    
    # Create a session manager for a role-playing scenario
    sm = SessionManager()
    
    # Set initial role
    await sm.update_system_prompt(
        "You are a friendly tour guide in Paris. You're knowledgeable about history, "
        "culture, and local attractions. Speak with enthusiasm and include interesting facts."
    )
    
    await sm.user_says("Hello! I just arrived in Paris. What should I see first?")
    await sm.ai_responds(
        "Bonjour and welcome to Paris! How exciting that you've just arrived! "
        "I'd recommend starting with the Eiffel Tower - did you know it was originally "
        "intended to be temporary? Built in 1889, it's now our most iconic landmark!",
        model="gpt-4o"
    )
    
    # Change role mid-conversation
    print("\n[Switching role to restaurant critic...]\n")
    await sm.update_system_prompt(
        "You are now a discerning food critic with expertise in French cuisine. "
        "Provide sophisticated recommendations while being approachable."
    )
    
    await sm.user_says("Where should I eat dinner tonight?")
    await sm.ai_responds(
        "For an authentic Parisian dining experience, I'd suggest Le Comptoir du Relais "
        "in Saint-Germain. Chef Yves Camdeborde's neo-bistro cuisine beautifully balances "
        "tradition with innovation. The duck confit is exceptional.",
        model="gpt-4o"
    )
    
    # Demonstrate session persistence
    print(f"\nSession saved with ID: {sm.session_id}")
    print("System prompt history is preserved in session metadata")


async def main():
    """Run all demos."""
    await customer_support_demo()
    print("\n" + "="*50 + "\n")
    
    await multi_language_demo()
    print("\n" + "="*50 + "\n")
    
    await role_playing_demo()
    
    print("\n✅ All demos completed!")
    print("\nKey features demonstrated:")
    print("- Setting initial system prompts")
    print("- Updating prompts mid-conversation")
    print("- System prompts persisted with session")
    print("- Getting messages with system prompt for LLM calls")


if __name__ == "__main__":
    asyncio.run(main())