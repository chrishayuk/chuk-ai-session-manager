# chuk session manager

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, async-first session management system for AI applications, with robust support for conversations, tool calls, hierarchical relationships, and comprehensive observability.

## 🚀 Quick Install

```bash
# Install with uv (recommended)
uv pip install chuk-ai-session-manager

# With Redis support
uv pip install chuk-ai-session-manager[redis]

# Full install with all dependencies
uv pip install chuk-ai-session-manager[full]
```

## ✨ Key Features

- 🔄 **Fully Async**: Built from the ground up for non-blocking I/O and high concurrency
- 🗃️ **Multiple Storage Backends**: Choose from in-memory, file-based, or Redis storage
- 🌳 **Hierarchical Sessions**: Create parent-child relationships for complex workflows
- 📝 **Event Tracking**: Record all conversation interactions with complete audit trails
- 💰 **Token & Cost Tracking**: Real-time token counting and cost estimation across providers
- 🛠️ **Tool Integration**: Session-aware tool execution with caching and retry logic
- ♾️ **Infinite Conversations**: Automatic segmentation for conversations exceeding token limits
- 🔄 **Retry Patterns**: Built-in LLM cooperation and tool execution reliability
- 🤖 **OpenAI Integration**: Production-ready patterns with auto-discovery
- 📊 **Complete Observability**: Performance monitoring, error tracking, and analytics

## 🎯 Production Highlights

This isn't just a demo framework - it's designed for production AI applications with features like:

- **Real OpenAI Integration**: Tested with live GPT-4o-mini API calls
- **Concurrent Tool Execution**: Multiple tools executed in parallel (200ms for 3 tools)
- **Precise Cost Tracking**: Token usage down to fractions of a penny ($0.000845 for complex workflows)
- **Error Recovery**: Multi-layer retry patterns with complete failure tracking
- **Auto-Discovery**: Registry-based tool detection with zero manual configuration
- **Complete Audit Trails**: Every operation logged with parent-child relationships

## 🏃‍♂️ Quick Start

### Basic Session with Events

```python
import asyncio
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.storage import SessionStoreProvider, InMemorySessionStore

async def main():
    # Set up storage
    store = InMemorySessionStore()
    SessionStoreProvider.set_store(store)
    
    # Create a session
    session = await Session.create()
    
    # Add events with automatic token tracking
    await session.add_event_and_save(await SessionEvent.create_with_tokens(
        message="How do I calculate the area of a circle?",
        prompt="How do I calculate the area of a circle?",
        model="gpt-4o-mini",
        source=EventSource.USER
    ))
    
    await session.add_event_and_save(await SessionEvent.create_with_tokens(
        message="The area of a circle is calculated using the formula: A = πr²",
        prompt="How do I calculate the area of a circle?",
        completion="The area of a circle is calculated using the formula: A = πr²",
        model="gpt-4o-mini",
        source=EventSource.LLM
    ))
    
    # Print session info with cost tracking
    print(f"Session ID: {session.id}")
    print(f"Event count: {len(session.events)}")
    print(f"Total tokens: {session.total_tokens}")
    print(f"Estimated cost: ${session.total_cost:.6f}")

if __name__ == "__main__":
    asyncio.run(main())
```

### OpenAI Integration with Auto-Discovery

```python
import asyncio
import json
from openai import AsyncOpenAI
from chuk_tool_processor.registry import initialize
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.storage import SessionStoreProvider, InMemorySessionStore

# Import tools - auto-registers via decorators
from your_tools import sample_tools

async def openai_integration_demo():
    # Setup
    store = InMemorySessionStore()
    SessionStoreProvider.set_store(store)
    session = await Session.create()
    
    # Auto-discover tools from registry
    registry = await initialize()
    tools_list = await registry.list_tools()
    print(f"🔧 Auto-discovered {len(tools_list)} tools")
    
    # Generate OpenAI function schemas automatically
    openai_tools = await generate_openai_functions_from_registry(registry)
    
    # Call OpenAI with auto-discovered tools
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in Tokyo and calculate 15.5 × 23.2?"}],
        tools=openai_tools,
        tool_choice="auto"
    )
    
    # Execute tools and track in session
    processor = await CleanSessionAwareToolProcessor.create(session_id=session.id)
    tool_results = await processor.process_llm_message(response.choices[0].message.model_dump())
    
    # Results automatically logged with complete observability
    print(f"Executed {len(tool_results)} tools successfully!")
    print(f"Total cost: ${session.total_cost:.6f}")

asyncio.run(openai_integration_demo())
```

## 📚 Storage Options

### In-Memory (Default)
```python
from chuk_ai_session_manager.storage import InMemorySessionStore, SessionStoreProvider

# Great for testing or single-process applications
store = InMemorySessionStore()
SessionStoreProvider.set_store(store)
```

### File Storage
```python
from chuk_ai_session_manager.storage.providers.file import create_file_session_store

# Persistent JSON file storage with async I/O
store = await create_file_session_store(directory="./sessions")
SessionStoreProvider.set_store(store)
```

### Redis Storage
```python
from chuk_ai_session_manager.storage.providers.redis import create_redis_session_store

# Distributed storage for production with TTL
store = await create_redis_session_store(
    host="localhost",
    port=6379,
    expiration_seconds=86400  # 24-hour TTL
)
SessionStoreProvider.set_store(store)
```

## 🌳 Hierarchical Sessions

```python
# Create parent-child relationships for complex workflows
parent = await Session.create()
child1 = await Session.create(parent_id=parent.id)
child2 = await Session.create(parent_id=parent.id)

# Navigate the hierarchy efficiently
ancestors = await child1.ancestors()
descendants = await parent.descendants()

# Build prompts with inherited context
from chuk_ai_session_manager.session_prompt_builder import build_prompt_from_session, PromptStrategy

prompt = await build_prompt_from_session(
    child1,
    strategy=PromptStrategy.HIERARCHICAL,
    include_parent_context=True
)
```

## 💰 Token & Cost Tracking

```python
# Automatic token counting with cost estimation
event = await SessionEvent.create_with_tokens(
    message="Explain quantum computing in simple terms",
    prompt="Explain quantum computing in simple terms",
    completion="Quantum computing uses qubits that can be both 0 and 1...",
    model="gpt-4",
    source=EventSource.LLM
)
await session.add_event_and_save(event)

# Real-time usage analytics
print(f"Total tokens: {session.total_tokens}")
print(f"Estimated cost: ${session.total_cost:.6f}")

# Per-model breakdown
for model, usage in session.token_summary.usage_by_model.items():
    print(f"{model}: {usage.total_tokens} tokens (${usage.estimated_cost_usd:.6f})")

# Usage by source (user, llm, system)
usage_by_source = await session.get_token_usage_by_source()
```

## 🛠️ Tool Processing with Registry

```python
from chuk_tool_processor.registry import register_tool

# Clean tool registration with decorators
@register_tool(name="weather", namespace="default", description="Get weather info")
class WeatherTool:
    async def execute(self, location: str) -> Dict[str, Any]:
        # Your tool implementation
        return {"location": location, "temperature": 22.5, "condition": "Sunny"}

# Session-aware tool execution with retry and caching
processor = await SessionAwareToolProcessor.create(session_id=session.id)

# Process LLM response with tool calls
llm_response = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "function": {
                "name": "weather",
                "arguments": '{"location": "London"}'
            }
        }
    ]
}

# Execute tools with automatic session tracking
results = await processor.process_llm_message(llm_response, llm_callback)
```

## ♾️ Infinite Conversations

```python
from chuk_ai_session_manager.infinite_conversation import InfiniteConversationManager, SummarizationStrategy

# Handle conversations that exceed token limits
manager = InfiniteConversationManager(
    token_threshold=3000,
    summarization_strategy=SummarizationStrategy.KEY_POINTS
)

# Automatic segmentation with context preservation
new_session_id = await manager.process_message(
    session_id, 
    message,
    source,
    llm_callback
)

# Retrieve complete history across all segments
history = await manager.get_full_conversation_history(new_session_id)
```

## 🔄 LLM Retry Patterns

```python
class LLMRetryManager:
    """Production-ready LLM retry logic with session tracking."""
    
    async def get_valid_tool_calls(self, llm, messages, processor, max_attempts=5):
        for attempt in range(1, max_attempts + 1):
            # Call LLM
            response = await llm.chat_completion(messages)
            
            # Log attempt in session
            await session.add_event_and_save(SessionEvent(
                message={"attempt": attempt, "response": response},
                type=EventType.MESSAGE,
                source=EventSource.LLM
            ))
            
            # Try to execute tools
            try:
                tool_results = await processor.process_llm_message(response)
                
                # Check for failures
                failed_tools = [r for r in tool_results if r.error]
                if not failed_tools:
                    return response, tool_results  # Success!
                    
            except Exception as e:
                continue  # Retry on failure
        
        raise RuntimeError(f"Failed after {max_attempts} attempts")

# Complete audit trail of all retry attempts
# Separation of concerns: LLM cooperation vs tool reliability
# Automatic recovery with detailed error tracking
```

## 📊 Production Observability

### Complete Event Hierarchy
```python
# Every operation creates a traceable event tree
• user_message    [abc123...]
• llm_message     [def456...]
  • tool_call     [ghi789...] - weather ✅ Success
  • tool_call     [jkl012...] - calculator ✅ Success  
  • tool_call     [mno345...] - search ✅ Success

# Parent-child relationships maintained automatically
# Performance monitoring with execution spans
# Error tracking with detailed stack traces
```

### Real-Time Analytics
```python
# Token usage across all operations
session.total_tokens  # 441 tokens
session.total_cost    # $0.000845

# Per-model breakdown
session.token_summary.usage_by_model
# gpt-4o-mini: 230 tokens ($0.000432)
# tool-execution: 211 tokens ($0.000413)

# Performance metrics
execution_time  # 202ms for 3 concurrent tools
success_rate    # 100% with retry patterns
```

## 🎯 Real Production Results

Based on actual demo runs with live OpenAI API:

```
🚀 Clean OpenAI Demo with Registry Auto-Discovery

🔧 Auto-discovered 3 tools from registry:
   • default.calculator: Perform basic arithmetic operations
   • default.weather: Get current weather information  
   • default.search: Search for information on the internet

🤖 Calling OpenAI with 3 auto-discovered tools...

📞 LLM wants to call 3 tools:
   • weather({"location": "Tokyo"})
   • calculator({"operation": "multiply", "a": 15.5, "b": 23.2})
   • search({"query": "renewable energy"})

✅ Tool Results:
   🌤️ Tokyo: 21.0°C, Sunny (Humidity: 42%, Wind: 4.1 km/h)
   🧮 15.5 multiply 23.2 = 359.6
   🔍 'renewable energy': Found 2 results

💰 Token Usage:
   Total tokens: 441 | Estimated cost: $0.000845
   📊 gpt-4o-mini: 230 tokens ($0.000432)
   📊 tool-execution: 211 tokens ($0.000413)

🎉 All tools executed successfully in 202ms!
```

## 📖 Examples

### Production OpenAI Integration
```bash
# Complete OpenAI integration with auto-discovery
uv run examples/clean_openai_demo.py
```

### LLM Retry Patterns
```bash
# Demonstrates retry logic for uncooperative LLMs
uv run examples/llm_retry_demo.py
```

### Token Cost Tracking
```bash
# Real-time token usage and cost monitoring
uv run examples/session_token_usage_example.py
```

### Infinite Conversations
```bash
# Automatic conversation segmentation
uv run examples/example_infinite_conversation.py
```

### FastAPI Integration
```bash
# Complete REST API with session management
uv run examples/fastapi_session_example.py
```

### Basic Session Management
```bash
# Fundamental session and event operations
uv run examples/session_example.py
```

## 🏗️ Architecture

The CHUK AI Session Manager provides a comprehensive foundation for production AI applications:

- **Session Layer**: Hierarchical conversation management with async operations
- **Event Layer**: Complete audit trails with parent-child relationships  
- **Storage Layer**: Pluggable backends (memory, file, Redis) with async I/O
- **Tool Layer**: Registry-based auto-discovery with session-aware execution
- **Cost Layer**: Real-time token tracking and cost estimation
- **Retry Layer**: Multi-level error recovery patterns
- **Observability Layer**: Performance monitoring and analytics

## 🔧 Advanced Configuration

### Custom Tool Processor
```python
class CustomSessionAwareToolProcessor:
    """Production tool processor with registry integration."""
    
    @classmethod
    async def create(cls, session_id: str):
        registry = await get_default_registry()
        strategy = InProcessStrategy(registry)
        executor = ToolExecutor(registry=registry, strategy=strategy)
        return cls(session_id, registry, executor)
```

### Session Runs for Workflows
```python
# Track multi-step processes
run = await SessionRun.create(metadata={"task": "data_analysis"})
await run.mark_running()

# Associate events with runs
await session.add_event_and_save(SessionEvent(
    message="Processing dataset...",
    source=EventSource.SYSTEM,
    task_id=run.id
))

await run.mark_completed()
```

### Prompt Building Strategies
```python
# Multiple strategies for different use cases
strategies = [
    PromptStrategy.MINIMAL,        # Basic task + latest results
    PromptStrategy.TASK_FOCUSED,   # Emphasizes original task  
    PromptStrategy.TOOL_FOCUSED,   # Detailed tool information
    PromptStrategy.CONVERSATION,   # Recent message history
    PromptStrategy.HIERARCHICAL,   # Includes parent context
]

prompt = await build_prompt_from_session(session, strategy=PromptStrategy.CONVERSATION)
```

## 🤝 Contributing

We welcome contributions! This project is designed for production use and follows best practices for async Python development.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Ready for Production** • **Async Native** • **Complete Observability** • **Cost Optimized**