# chuk-ai-session-manager

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The easiest way to add conversation tracking to any AI application.**

Track conversations, monitor costs, and manage infinite context with just 3 lines of code. Built for production, designed for simplicity.

## 🚀 30-Second Start

```bash
uv add chuk-ai-session-manager
```

```python
from chuk_ai_session_manager import track_conversation

# Track any AI conversation in one line
await track_conversation("Hello!", "Hi there! How can I help?")
```

That's it! 🎉 Your conversation is now tracked with full observability.

## ✨ Why Choose CHUK?

- **🔥 Stupidly Simple**: 3 lines to track any conversation
- **💰 Cost Smart**: Automatic token counting and cost tracking
- **♾️ Infinite Context**: No more "conversation too long" errors
- **🔧 Any LLM**: Works with OpenAI, Anthropic, local models, anything
- **📊 Full Observability**: See exactly what's happening in your AI app
- **🚀 Production Ready**: Used in real applications, not just demos

## 🎯 Perfect For

- **Building chatbots** that remember conversations
- **Tracking LLM costs** across your entire application  
- **Managing long conversations** without hitting token limits
- **Debugging AI applications** with complete audit trails
- **Production AI systems** that need reliable session management

## 📱 Quick Examples

### Track Any Conversation
```python
from chuk_ai_session_manager import track_conversation

# Works with any LLM response
session_id = await track_conversation(
    user_message="What's the weather like?",
    ai_response="It's sunny and 75°F in your area.",
    model="gpt-4",
    provider="openai"
)
```

### Persistent Conversations
```python
from chuk_ai_session_manager import SessionManager

# Create a conversation that remembers context
sm = SessionManager()

await sm.user_says("My name is Alice")
await sm.ai_responds("Nice to meet you, Alice!")

await sm.user_says("What's my name?")
await sm.ai_responds("Your name is Alice!")

# Get conversation stats
stats = await sm.get_stats()
print(f"Cost: ${stats['estimated_cost']:.6f}")
print(f"Tokens: {stats['total_tokens']}")
```

### Infinite Context (Never Run Out of Space)
```python
# Automatically handles conversations of any length
sm = SessionManager(
    infinite_context=True,          # 🔥 Magic happens here
    token_threshold=4000           # When to create new segment
)

# Keep chatting forever - context is preserved automatically
for i in range(100):  # This would normally hit token limits
    await sm.user_says(f"Question {i}: Tell me about AI")
    await sm.ai_responds("AI is fascinating...")

# Still works! Automatic summarization keeps context alive
conversation = await sm.get_conversation()
print(f"Full conversation: {len(conversation)} exchanges")
```

### Cost Tracking (Know What You're Spending)
```python
# Automatic cost monitoring across all interactions
sm = SessionManager()

await sm.user_says("Write a long story about dragons")
await sm.ai_responds("Once upon a time..." * 500)  # Long response

stats = await sm.get_stats()
print(f"💰 That story cost: ${stats['estimated_cost']:.6f}")
print(f"📊 Used {stats['total_tokens']} tokens")
print(f"📈 {stats['user_messages']} user messages, {stats['ai_messages']} AI responses")
```

### Multi-Provider Support
```python
# Works with any LLM provider
import openai
import anthropic

sm = SessionManager()

# OpenAI
await sm.user_says("Hello!")
openai_response = await openai.chat.completions.create(...)
await sm.ai_responds(openai_response.choices[0].message.content, model="gpt-4", provider="openai")

# Anthropic
await sm.user_says("How are you?")
anthropic_response = await anthropic.messages.create(...)
await sm.ai_responds(anthropic_response.content[0].text, model="claude-3", provider="anthropic")

# See costs across all providers
stats = await sm.get_stats()
print(f"Total cost across all providers: ${stats['estimated_cost']:.6f}")
```

## 🛠️ Advanced Features

### Conversation Analytics
```python
# Get detailed insights into your conversations
conversation = await sm.get_conversation()
stats = await sm.get_stats()

print(f"📊 Conversation Analytics:")
print(f"   Messages: {stats['user_messages']} user, {stats['ai_messages']} AI")
print(f"   Average response length: {stats['avg_response_length']}")
print(f"   Most expensive response: ${stats['max_response_cost']:.6f}")
print(f"   Session duration: {stats['duration_minutes']:.1f} minutes")
```

### Tool Integration
```python
# Track tool usage alongside conversations
await sm.tool_used(
    tool_name="web_search",
    arguments={"query": "latest AI news"},
    result={"articles": ["AI breakthrough...", "New model released..."]},
    cost=0.001
)

stats = await sm.get_stats()
print(f"Tool calls: {stats['tool_calls']}")
```

### Session Export/Import
```python
# Export conversations for analysis
conversation_data = await sm.export_conversation()
with open('conversation.json', 'w') as f:
    json.dump(conversation_data, f)

# Import previous conversations
sm = SessionManager()
await sm.import_conversation('conversation.json')
```

## 🎨 Real-World Examples

### Customer Support Bot
```python
async def handle_support_ticket(user_message: str, ticket_id: str):
    # Each ticket gets its own session
    sm = SessionManager(session_id=ticket_id)
    
    await sm.user_says(user_message)
    
    # Your AI logic here
    ai_response = await your_ai_model(user_message)
    await sm.ai_responds(ai_response, model="gpt-4", provider="openai")
    
    # Automatic cost tracking per ticket
    stats = await sm.get_stats()
    print(f"Ticket {ticket_id} cost: ${stats['estimated_cost']:.6f}")
    
    return ai_response
```

### AI Assistant with Memory
```python
async def ai_assistant():
    sm = SessionManager(infinite_context=True)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        await sm.user_says(user_input)
        
        # Get conversation context for AI
        conversation = await sm.get_conversation()
        context = "\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation[-5:]])
        
        # Your AI call with context
        ai_response = await your_ai_model(f"Context:\n{context}\n\nUser: {user_input}")
        await sm.ai_responds(ai_response)
        
        print(f"AI: {ai_response}")
    
    # Show final stats
    stats = await sm.get_stats()
    print(f"\n💰 Total conversation cost: ${stats['estimated_cost']:.6f}")
```

### Multi-User Chat Application
```python
class ChatApplication:
    def __init__(self):
        self.user_sessions = {}
    
    async def handle_message(self, user_id: str, message: str):
        # Each user gets their own session
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = SessionManager(infinite_context=True)
        
        sm = self.user_sessions[user_id]
        await sm.user_says(message)
        
        # AI processes with user's personal context
        ai_response = await self.generate_response(sm, message)
        await sm.ai_responds(ai_response)
        
        return ai_response
    
    async def get_user_stats(self, user_id: str):
        if user_id in self.user_sessions:
            return await self.user_sessions[user_id].get_stats()
        return None
```

## 📊 Monitoring Dashboard

```python
# Get comprehensive analytics across all sessions
from chuk_ai_session_manager import get_global_stats

stats = await get_global_stats()
print(f"""
🚀 AI Application Dashboard
==========================
Total Sessions: {stats['total_sessions']}
Total Messages: {stats['total_messages']}
Total Cost: ${stats['total_cost']:.2f}
Average Session Length: {stats['avg_session_length']:.1f} messages
Most Active Hour: {stats['peak_hour']}
Top Models Used: {', '.join(stats['top_models'])}
""")
```

## 🔧 Installation Options

```bash
# Basic installation
uv add chuk-ai-session-manager

# With Redis support (for production)
uv add chuk-ai-session-manager[redis]

# Full installation (all features)
uv add chuk-ai-session-manager[full]

# Or with pip
pip install chuk-ai-session-manager
```

## 🌟 What Makes CHUK Special?

| Feature | Other Libraries | CHUK AI Session Manager |
|---------|----------------|------------------------|
| **Setup Complexity** | Complex configuration | 3 lines of code |
| **Cost Tracking** | Manual calculation | Automatic across all providers |
| **Long Conversations** | Token limit errors | Infinite context with auto-segmentation |
| **Multi-Provider** | Provider-specific code | Works with any LLM |
| **Production Ready** | Requires additional work | Built for production |
| **Learning Curve** | Steep | 5 minutes to productivity |

## 📖 More Examples

Check out the `/examples` directory for complete working examples:

- `simple_tracking.py` - Basic conversation tracking
- `openai_integration.py` - OpenAI API integration
- `infinite_context.py` - Handling long conversations
- `cost_monitoring.py` - Cost tracking and analytics
- `multi_provider.py` - Using multiple LLM providers
- `production_app.py` - Production-ready application

## 🎯 Quick Decision Guide

**Choose CHUK AI Session Manager if you want:**
- ✅ Simple conversation tracking with zero configuration
- ✅ Automatic cost monitoring across all LLM providers
- ✅ Infinite conversation length without token limit errors
- ✅ Production-ready session management out of the box
- ✅ Complete conversation analytics and observability
- ✅ Framework-agnostic solution that works with any LLM library

## 🤝 Community & Support

- 📖 **Documentation**: [Full docs with tutorials](link-to-docs)
- 🐛 **Issues**: Report bugs on GitHub
- 💡 **Feature Requests**: Suggest new features
- 📧 **Support**: enterprise@chuk.dev for production support

## 📝 License

MIT License - build amazing AI applications with confidence!

---

**🎉 Ready to build better AI applications?**

```bash
uv add chuk-ai-session-manager
```

**Get started in 30 seconds with one line of code!**