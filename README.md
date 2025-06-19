# CHUK AI Session Manager Documentation

A powerful session management system for AI applications that provides automatic conversation tracking, token usage monitoring, tool call logging, infinite context support, and hierarchical session relationships.

## Table of Contents

1. [Overview](#overview)
2. [Import Issues & Fixes](#import-issues--fixes)
3. [Core Architecture](#core-architecture)
4. [Quick Start](#quick-start)
5. [Simple API](#simple-api)
6. [Advanced Usage](#advanced-usage)
7. [Core Models](#core-models)
8. [Infinite Context](#infinite-context)
9. [Tool Integration](#tool-integration)
10. [Session Storage](#session-storage)
11. [Prompt Building](#prompt-building)
12. [Configuration](#configuration)

## Overview

The CHUK AI Session Manager is designed to solve common challenges in AI application development:

- **Conversation Tracking**: Automatically track user-AI interactions
- **Token Management**: Monitor usage and costs across different models
- **Infinite Context**: Handle conversations that exceed token limits through automatic summarization
- **Tool Integration**: Log tool calls and results seamlessly
- **Session Hierarchy**: Create parent-child relationships between conversation segments
- **Flexible Storage**: Built on CHUK Sessions for reliable persistence

### Key Features

- **Zero-friction API**: Simple functions for common tasks
- **Async-first**: Built for modern Python async/await patterns
- **Token-aware**: Automatic token counting and cost estimation
- **Provider-agnostic**: Works with any LLM provider (OpenAI, Anthropic, etc.)
- **Hierarchical sessions**: Support for complex conversation structures
- **Automatic summarization**: Maintains context across session segments

## Import Structure

With the clean `__init__.py`, all components should be available at the top level:

```python
from chuk_ai_session_manager import (
    # Simple API - Primary interface for most users
    SessionManager,
    track_conversation,
    track_llm_call,
    quick_conversation,
    track_infinite_conversation,
    track_tool_use,
    get_session_stats,
    get_conversation_history,
    
    # Core Models
    Session,
    SessionEvent,
    SessionMetadata,
    SessionRun,
    RunStatus,
    
    # Enums
    EventSource,
    EventType,
    
    # Token Management
    TokenUsage,
    TokenSummary,
    
    # Advanced Components
    InfiniteConversationManager,
    SummarizationStrategy,
    SessionAwareToolProcessor,
    build_prompt_from_session,
    PromptStrategy,
    truncate_prompt_to_token_limit,
    
    # Storage
    setup_chuk_sessions_storage,
    
    # Exceptions
    SessionManagerError,
    SessionNotFound,
    SessionAlreadyExists,
    InvalidSessionOperation,
    TokenLimitExceeded,
    StorageError,
    ToolProcessingError,
    
    # Utilities
    configure_storage,
    get_version,
    is_available
)
```

### Verifying Installation

Check that everything is working:

```python
import chuk_ai_session_manager as casm

print(f"Version: {casm.get_version()}")
print("Available components:", casm.is_available())

# This should show all components as True
# {
#   "core_enums": True,
#   "core_models": True, 
#   "simple_api": True,
#   "storage": True,
#   "infinite_context": True,
#   "tool_processor": True,
#   "prompt_builder": True,
#   "token_tracking": True,
#   "exceptions": True,
#   "session_manager": True
# }
```

## Core Architecture

The system is built around several key components working together to provide seamless conversation management:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Simple API    │    │ SessionManager   │    │ Core Models     │
│                 │    │                  │    │                 │
│ track_conversation() │    │ High-level API   │    │ Session         │
│ track_llm_call()     │    │ Infinite context │    │ SessionEvent    │
│ quick_conversation() │    │ Auto-summarization│   │ TokenUsage      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Tool Processor  │    │ Storage Backend  │    │ Prompt Builder  │
│                 │    │                  │    │                 │
│ Session-aware   │    │ CHUK Sessions    │    │ Multiple        │
│ Tool execution  │    │ JSON persistence │    │ strategies      │
│ Retry & caching │    │ TTL management   │    │ Token limits    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

**Key Components:**
- **Simple API**: One-line functions for common operations
- **SessionManager**: High-level conversation management with infinite context
- **Core Models**: Session, SessionEvent, TokenUsage for data modeling
- **Tool Processor**: Automatic tool call tracking with retry and caching
- **Storage Backend**: CHUK Sessions for reliable persistence
- **Prompt Builder**: Intelligent context building for LLM calls

## Quick Start

### Installation

```bash
# Install the package
uv add chuk-ai-session-manager

# Or with pip
pip install chuk-ai-session-manager
```

### Basic Usage

```python
from chuk_ai_session_manager import track_conversation

# Track a simple conversation
session_id = await track_conversation(
    user_message="What's the weather like?",
    ai_response="I don't have access to real-time weather data.",
    model="gpt-3.5-turbo",
    provider="openai"
)

print(f"Conversation tracked in session: {session_id}")
```

### With Statistics

```python
from chuk_ai_session_manager import quick_conversation

stats = await quick_conversation(
    user_message="Explain quantum computing",
    ai_response="Quantum computing uses quantum mechanical phenomena...",
    model="gpt-4",
    provider="openai"
)

print(f"Tokens used: {stats['total_tokens']}")
print(f"Estimated cost: ${stats['estimated_cost']:.4f}")
```

### Basic Integration with Your LLM

```python
from chuk_ai_session_manager import track_llm_call
import openai

async def my_openai_call(prompt):
    response = await openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Track your LLM call automatically
response, session_id = await track_llm_call(
    user_input="Explain machine learning",
    llm_function=my_openai_call,
    model="gpt-3.5-turbo",
    provider="openai"
)

print(f"AI Response: {response}")
print(f"Tracked in session: {session_id}")
```

## Simple API

The Simple API provides convenient functions for common tasks:

```python
from chuk_ai_session_manager import (
    track_conversation,
    track_llm_call,
    quick_conversation,
    track_infinite_conversation,
    track_tool_use,
    get_session_stats,
    get_conversation_history
)
```

### `track_conversation()`

The simplest way to track a conversation turn - perfect for one-off tracking:

```python
session_id = await track_conversation(
    user_message="Hello!",
    ai_response="Hi there! How can I help?",
    model="gpt-3.5-turbo",
    provider="openai",
    session_id=None,  # Optional: continue existing session
    infinite_context=False,  # Enable infinite context
    token_threshold=4000  # Token limit for segmentation
)

# Returns the session ID for continuing the conversation later
```

### `track_llm_call()`

Wrap your LLM function calls for automatic tracking:

```python
async def call_openai(prompt):
    # Your OpenAI API call here
    response = await openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

response, session_id = await track_llm_call(
    user_input="Explain machine learning",
    llm_function=call_openai,
    model="gpt-3.5-turbo",
    provider="openai"
)
```

### `track_infinite_conversation()`

For long conversations that might exceed token limits:

```python
# Start a conversation
session_id = await track_infinite_conversation(
    user_message="Tell me about the history of computing",
    ai_response="Computing history begins with ancient calculating devices...",
    model="gpt-4",
    token_threshold=4000,  # Auto-segment after 4000 tokens
    max_turns=20  # Or after 20 conversation turns
)

# Continue the conversation
session_id = await track_infinite_conversation(
    user_message="What about quantum computers?",
    ai_response="Quantum computing represents a fundamental shift...",
    session_id=session_id,  # Continue the same conversation
    model="gpt-4"
)
```

### `track_tool_use()`

Track tool/function calls:

```python
session_id = await track_tool_use(
    tool_name="calculator",
    arguments={"operation": "add", "a": 5, "b": 3},
    result={"result": 8},
    session_id=session_id,
    error=None  # Optional error message
)
```

## SessionManager Class

For more control and persistent conversations, use the `SessionManager` class directly:

```python
from chuk_ai_session_manager import SessionManager

# Create a session manager
sm = SessionManager(
    system_prompt="You are a helpful assistant specialized in Python programming.",
    infinite_context=True,
    token_threshold=4000,
    max_turns_per_segment=20
)

# Track conversations
await sm.user_says("How do I create a list comprehension?")
await sm.ai_responds(
    "A list comprehension is a concise way to create lists in Python...",
    model="gpt-4",
    provider="openai"
)

# Track tool usage
await sm.tool_used(
    tool_name="code_executor",
    arguments={"code": "print([x**2 for x in range(5)])"},
    result={"output": "[0, 1, 4, 9, 16]"}
)

# Get session statistics
stats = await sm.get_stats()
print(f"Session {stats['session_id']}: {stats['total_messages']} messages, ${stats['estimated_cost']:.4f}")
```

### Working with System Prompts

```python
# Set initial system prompt
sm = SessionManager(system_prompt="You are a creative writing assistant.")

# Update system prompt later
await sm.update_system_prompt("You are now a technical documentation writer.")

# Get messages including system prompt for your LLM calls
messages = await sm.get_messages_for_llm(include_system=True)
# [{"role": "system", "content": "You are now a technical documentation writer."}, ...]
```

### SessionManager Properties

```python
sm = SessionManager()

# Access session information
print(f"Session ID: {sm.session_id}")
print(f"System Prompt: {sm.system_prompt}")
print(f"Infinite Context: {sm.is_infinite}")

# Check if this is a new session
print(f"Is new session: {sm._is_new}")  # Useful for initialization logic
```

### Managing Long Conversations

```python
# Enable infinite context with custom settings
sm = SessionManager(
    infinite_context=True,
    token_threshold=3000,  # Segment at 3000 tokens
    max_turns_per_segment=15  # Or 15 conversation turns
)

# The session will auto-segment when limits are reached
# You don't need to do anything - it happens automatically!

# Get full conversation across all segments
full_conversation = await sm.get_conversation(include_all_segments=True)

# Get session chain (list of session IDs in the conversation)
session_chain = await sm.get_session_chain()
print(f"Conversation spans {len(session_chain)} sessions: {session_chain}")
```

## Core Models

### Session

The main container for a conversation:

```python
from chuk_ai_session_manager import Session

# Create a new session
session = await Session.create(
    parent_id=None,  # Optional parent session
    metadata={"user_id": "user123", "topic": "programming"}
)

# Session properties
print(f"Session ID: {session.id}")
print(f"Created: {session.metadata.created_at}")
print(f"Total tokens: {session.total_tokens}")
print(f"Total cost: ${session.total_cost:.4f}")

# Add events
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType

event = await SessionEvent.create_with_tokens(
    message="Hello world!",
    prompt="Hello world!",
    model="gpt-3.5-turbo",
    source=EventSource.USER,
    type=EventType.MESSAGE
)

await session.add_event_and_save(event)
```

### SessionEvent

Individual events within a session:

```python
from chuk_ai_session_manager import SessionEvent, EventSource, EventType

# Create an event with automatic token counting
event = await SessionEvent.create_with_tokens(
    message="What is machine learning?",
    prompt="What is machine learning?",
    completion=None,  # For user messages
    model="gpt-3.5-turbo",
    source=EventSource.USER,
    type=EventType.MESSAGE
)

# Event properties
print(f"Event ID: {event.id}")
print(f"Tokens used: {event.token_usage.total_tokens}")
print(f"Source: {event.source.value}")
print(f"Type: {event.type.value}")

# Update metadata
await event.set_metadata("user_id", "user123")
await event.set_metadata("intent", "question")

# Check metadata
user_id = await event.get_metadata("user_id")
has_intent = await event.has_metadata("intent")
```

### TokenUsage

Tracks token consumption and costs:

```python
from chuk_ai_session_manager import TokenUsage

# Create from text
usage = await TokenUsage.from_text(
    prompt="What is the capital of France?",
    completion="The capital of France is Paris.",
    model="gpt-3.5-turbo"
)

print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")
print(f"Estimated cost: ${usage.estimated_cost_usd:.6f}")

# Update token usage
await usage.update(prompt_tokens=10, completion_tokens=5)

# Count tokens for any text
token_count = await TokenUsage.count_tokens("Hello world!", "gpt-4")
```

### Event Source and Type Enums

```python
from chuk_ai_session_manager import EventSource, EventType

# Event sources
EventSource.USER     # User input
EventSource.LLM      # AI model response  
EventSource.SYSTEM   # System/tool events

# Event types
EventType.MESSAGE       # Conversation messages
EventType.TOOL_CALL     # Tool/function calls
EventType.SUMMARY       # Session summaries
EventType.REFERENCE     # References to other content
EventType.CONTEXT_BRIDGE # Context bridging events
```

## Infinite Context

The infinite context system automatically handles conversations that exceed token limits by creating linked sessions with summaries.

### InfiniteConversationManager

```python
from chuk_ai_session_manager import (
    InfiniteConversationManager,
    SummarizationStrategy,
    EventSource
)

# Create manager with custom settings
icm = InfiniteConversationManager(
    token_threshold=3000,
    max_turns_per_segment=15,
    summarization_strategy=SummarizationStrategy.KEY_POINTS
)

# Process messages (automatically segments when needed)
async def my_llm_callback(messages):
    # Your LLM call here
    return "Summary of the conversation..."

current_session_id = await icm.process_message(
    session_id="session-123",
    message="Tell me about quantum computing",
    source=EventSource.USER,
    llm_callback=my_llm_callback,
    model="gpt-4"
)

# Build context for LLM calls
context = await icm.build_context_for_llm(
    session_id=current_session_id,
    max_messages=10,
    include_summaries=True
)

# Get session chain
chain = await icm.get_session_chain(current_session_id)
print(f"Conversation chain: {[s.id for s in chain]}")
```

### Summarization Strategies

```python
from chuk_ai_session_manager import SummarizationStrategy

# Different summarization approaches
SummarizationStrategy.BASIC        # General overview
SummarizationStrategy.KEY_POINTS   # Focus on key information
SummarizationStrategy.TOPIC_BASED  # Organize by topics
SummarizationStrategy.QUERY_FOCUSED # Focus on user questions
```

## Tool Integration

### SessionAwareToolProcessor

Integrates with `chuk_tool_processor` for automatic tool call tracking:

```python
from chuk_ai_session_manager import SessionAwareToolProcessor

# Create processor for a session
processor = await SessionAwareToolProcessor.create(
    session_id="session-123",
    enable_caching=True,
    max_retries=2,
    retry_delay=1.0
)

# Process LLM message with tool calls
llm_response = {
    "tool_calls": [
        {
            "function": {
                "name": "calculator",
                "arguments": '{"operation": "add", "a": 5, "b": 3}'
            }
        }
    ]
}

results = await processor.process_llm_message(llm_response, None)
for result in results:
    print(f"Tool: {result.tool}, Result: {result.result}")
```

### Sample Tools

```python
# The package includes sample tools for demonstration
from chuk_ai_session_manager.sample_tools import (
    CalculatorTool,
    WeatherTool, 
    SearchTool
)

# These are registered with chuk_tool_processor
# You can see how to structure your own tools
```

## Session Storage

### CHUK Sessions Backend

The storage is built on CHUK Sessions:

```python
from chuk_ai_session_manager import (
    setup_chuk_sessions_storage,
    SessionStorage,
    ChukSessionsStore
)

# Setup storage (usually done automatically)
backend = setup_chuk_sessions_storage(
    sandbox_id="my-ai-app",
    default_ttl_hours=48
)

# Get the store
store = ChukSessionsStore(backend)

# Manual session operations
session = await store.get("session-123")
await store.save(session)
await store.delete("session-123")
session_ids = await store.list_sessions(prefix="user-")
```

### Storage Configuration

```python
# Configure storage at import time
from chuk_ai_session_manager import configure_storage

success = configure_storage(
    sandbox_id="my-application",
    default_ttl_hours=72  # 3 day TTL
)

if success:
    print("Storage configured successfully")
else:
    print("Storage configuration failed")
```

## Prompt Building

### Building Prompts from Sessions

```python
from chuk_ai_session_manager import (
    build_prompt_from_session,
    PromptStrategy,
    truncate_prompt_to_token_limit
)

# Build prompts with different strategies
prompt = await build_prompt_from_session(
    session,
    strategy=PromptStrategy.CONVERSATION,  # Include conversation history
    max_tokens=3000,
    model="gpt-4",
    include_parent_context=True,
    max_history=10
)

# Prompt strategies
PromptStrategy.MINIMAL      # Just task and latest context
PromptStrategy.TASK_FOCUSED # Focus on the task
PromptStrategy.TOOL_FOCUSED # Emphasize tool usage
PromptStrategy.CONVERSATION # Include conversation history
PromptStrategy.HIERARCHICAL # Include parent session context
```

### Token Limit Management

```python
from chuk_ai_session_manager import truncate_prompt_to_token_limit

# Ensure prompt fits within token limits
truncated_prompt = await truncate_prompt_to_token_limit(
    prompt=messages,
    max_tokens=3000,
    model="gpt-3.5-turbo"
)
```

## Configuration

### Package Configuration

```python
import chuk_ai_session_manager as casm

# Check what's available
print("Package version:", casm.get_version())
availability = casm.is_available()
print("Available components:", availability)

# Configure storage
success = casm.configure_storage(
    sandbox_id="my-app",
    default_ttl_hours=24
)
```

### Environment Setup

The package depends on several components:

1. **Required**: `chuk_sessions` - for storage backend
2. **Required**: `pydantic` - for data models  
3. **Optional**: `tiktoken` - for accurate token counting (falls back to approximation)
4. **Optional**: `chuk_tool_processor` - for tool integration

### Error Handling

```python
from chuk_ai_session_manager import (
    SessionManagerError,
    SessionNotFound,
    TokenLimitExceeded,
    StorageError
)

try:
    session_id = await track_conversation("Hello", "Hi there")
except SessionNotFound as e:
    print(f"Session not found: {e}")
except TokenLimitExceeded as e:
    print(f"Token limit exceeded: {e}")
except StorageError as e:
    print(f"Storage error: {e}")
except SessionManagerError as e:
    print(f"General session error: {e}")
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
| **Tool Integration** | Manual tracking | Automatic tool call logging |
| **Session Management** | Build from scratch | Complete session hierarchy |

## 🎯 Quick Decision Guide

**Choose CHUK AI Session Manager if you want:**
- ✅ Simple conversation tracking with zero configuration
- ✅ Automatic cost monitoring across all LLM providers
- ✅ Infinite conversation length without token limit errors
- ✅ Production-ready session management out of the box
- ✅ Complete conversation analytics and observability
- ✅ Framework-agnostic solution that works with any LLM library
- ✅ Built-in tool call tracking and retry mechanisms
- ✅ Hierarchical session relationships for complex workflows

## 📊 Monitoring & Analytics

```python
# Get comprehensive session analytics
stats = await sm.get_stats(include_all_segments=True)

print(f"""
🚀 Session Analytics Dashboard
============================
Session ID: {stats['session_id']}
Total Messages: {stats['total_messages']}
User Messages: {stats['user_messages']}
AI Messages: {stats['ai_messages']}
Tool Calls: {stats['tool_calls']}
Total Tokens: {stats['total_tokens']}
Total Cost: ${stats['estimated_cost']:.6f}
Session Segments: {stats.get('session_segments', 1)}
Created: {stats['created_at']}
Last Update: {stats['last_update']}
Infinite Context: {stats.get('infinite_context', False)}
""")

# Get conversation history
conversation = await sm.get_conversation(include_all_segments=True)
for i, turn in enumerate(conversation):
    print(f"{i+1}. {turn['role']}: {turn['content'][:50]}...")
```

## 🛡️ Error Handling

The package provides specific exceptions for different error conditions:

```python
from chuk_ai_session_manager import (
    SessionManagerError,
    SessionNotFound,
    TokenLimitExceeded,
    StorageError
)

try:
    session_id = await track_conversation("Hello", "Hi there")
except SessionNotFound as e:
    print(f"Session not found: {e}")
except TokenLimitExceeded as e:
    print(f"Token limit exceeded: {e}")
except StorageError as e:
    print(f"Storage error: {e}")
except SessionManagerError as e:
    print(f"General session error: {e}")
```

## 🔧 Environment Setup

The package requires several dependencies that should be automatically installed:

1. **Required**: `chuk_sessions` - for storage backend
2. **Required**: `pydantic` - for data models  
3. **Optional**: `tiktoken` - for accurate token counting (falls back to approximation)
4. **Optional**: `chuk_tool_processor` - for tool integration

### Dependencies Check

```python
import chuk_ai_session_manager as casm

# Check if all components are available
availability = casm.is_available()
for component, available in availability.items():
    status = "✅" if available else "❌"
    print(f"{status} {component}")
```

## 🤝 Community & Support

- 📖 **Full Documentation**: Complete API reference and tutorials
- 🐛 **Issues**: Report bugs and request features on GitHub  
- 💡 **Examples**: Check `/examples` directory for working code
- 📧 **Support**: Enterprise support available

## 📝 License

MIT License - build amazing AI applications with confidence!

---

**🎉 Ready to build better AI applications?**

```bash
uv add chuk-ai-session-manager
```

**Get started in 30 seconds with one line of code!**