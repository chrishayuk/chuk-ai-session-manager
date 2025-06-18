# examples/04_any_llm_provider.py
"""
🌐 ANY LLM PROVIDER: Track calls to any AI service

Works with OpenAI, Anthropic, local models, or any API.
"""
import asyncio
from chuk_ai_session_manager import SessionManager

# Mock different providers
async def mock_anthropic(prompt):
    return f"Claude response: {prompt} (This would be from Anthropic API)"

async def mock_local_llm(prompt):  
    return f"Local model response: {prompt} (This would be from Ollama/local)"

async def mock_custom_api(prompt):
    return f"Custom API response: {prompt}"

async def main():
    print("🌐 Tracking multiple AI providers...")
    
    sm = SessionManager()
    
    # Track different providers in same conversation
    providers = [
        ("anthropic", "claude-3-sonnet", mock_anthropic),
        ("local", "llama-3.1-8b", mock_local_llm),
        ("custom", "my-model-v1", mock_custom_api)
    ]
    
    user_question = "What's the most important skill for a software engineer?"
    await sm.user_says(user_question)
    
    for provider, model, llm_func in providers:
        response = await llm_func(user_question)
        await sm.ai_responds(
            response=response,
            model=model,
            provider=provider
        )
        print(f"✅ {provider}/{model}: {response[:40]}...")
    
    # Compare responses
    conversation = await sm.get_conversation()
    stats = await sm.get_stats()
    
    print(f"\n📊 Tracked responses from {len(providers)} different providers")
    print(f"📈 Session: {stats['total_tokens']} tokens, ${stats['estimated_cost']:.6f}")

if __name__ == "__main__":
    asyncio.run(main())