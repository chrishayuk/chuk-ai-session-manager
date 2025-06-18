#!/usr/bin/env python3
# examples/06_chuk_llm.py
"""
🚀 REAL CHUK LLM DEMO - Production Integration

Comprehensive demo showcasing CHUK LLM integration with session management.
Demonstrates multiple providers, conversation tracking, and production patterns.

Setup:
    1. Create .env file in project root with your API keys:
       OPENAI_API_KEY=your-openai-key-here
       ANTHROPIC_API_KEY=your-anthropic-key-here
       GROQ_API_KEY=your-groq-key-here (optional)
    
    2. Install dependencies:
       uv add chuk-llm python-dotenv

Run:
    uv run examples/real_chuk_llm_demo.py

Features:
    ✅ Multiple LLM providers (OpenAI, Anthropic, Groq)
    ✅ Simple functions with session tracking
    ✅ Advanced client API integration
    ✅ Parallel provider comparisons
    ✅ Tool integration and execution tracking
    ✅ Complete conversation context management
    ✅ Infinite context with automatic segmentation
    ✅ Cost and token monitoring
    ✅ Production-ready error handling
"""

import asyncio
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CHUK LLM imports
try:
    from chuk_llm import (
        # Simple async functions
        ask_openai_gpt4o_mini,
        ask_anthropic_claude_sonnet4_20250514,
        ask_anthropic_sonnet,
        ask_groq_llama,
        # Advanced client
        get_client
    )
    from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator
except ImportError as e:
    print("❌ CHUK LLM not available. Install with: uv add chuk-llm")
    print(f"   Error: {e}")
    sys.exit(1)

# Session manager imports
from chuk_ai_session_manager import SessionManager, track_conversation
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.session_storage import setup_chuk_sessions_storage

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Quiet external loggers
for logger_name in ["httpx", "httpcore", "urllib3", "openai", "anthropic", "chuk_llm"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


class TrackedChukLLM:
    """Enhanced wrapper for CHUK LLM with comprehensive session tracking."""
    
    def __init__(self, session: Session):
        self.session = session
        self.available_providers = self._check_available_providers()
        
    def _check_available_providers(self) -> Dict[str, bool]:
        """Check which providers are configured."""
        return {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "groq": bool(os.getenv("GROQ_API_KEY")),
        }
    
    async def _track_llm_interaction(
        self, 
        user_message: str, 
        response: str, 
        provider: str, 
        model: str,
        response_time: float = None,
        token_info: Dict = None,
        metadata: Dict = None
    ):
        """Track a complete LLM interaction with detailed metadata."""
        base_metadata = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "response_time_seconds": response_time,
            **(metadata or {})
        }
        
        # Track user message
        user_event = await SessionEvent.create_with_tokens(
            message=user_message,
            prompt=user_message,
            model=model,
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        for key, value in base_metadata.items():
            if value is not None:
                await user_event.set_metadata(key, value)
        
        await self.session.add_event_and_save(user_event)
        
        # Track AI response
        assistant_event = await SessionEvent.create_with_tokens(
            message=response,
            prompt="",
            completion=response,
            model=model,
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        for key, value in base_metadata.items():
            if value is not None:
                await assistant_event.set_metadata(key, value)
                
        if token_info:
            for key, value in token_info.items():
                await assistant_event.set_metadata(f"token_{key}", value)
        
        await self.session.add_event_and_save(assistant_event)
    
    # Simple function wrappers with tracking
    async def ask_openai_gpt4o_mini(self, question: str, **kwargs) -> str:
        """OpenAI GPT-4o Mini with session tracking."""
        if not self.available_providers["openai"]:
            raise ValueError("OpenAI API key not configured")
        
        start_time = asyncio.get_event_loop().time()
        response = await ask_openai_gpt4o_mini(question, **kwargs)
        response_time = asyncio.get_event_loop().time() - start_time
        
        await self._track_llm_interaction(
            question, response, "openai", "gpt-4o-mini", 
            response_time=response_time
        )
        return response
    
    async def ask_anthropic_sonnet(self, question: str, **kwargs) -> str:
        """Anthropic Claude Sonnet with session tracking."""
        if not self.available_providers["anthropic"]:
            raise ValueError("Anthropic API key not configured")
        
        start_time = asyncio.get_event_loop().time()
        response = await ask_anthropic_sonnet(question, **kwargs)
        response_time = asyncio.get_event_loop().time() - start_time
        
        await self._track_llm_interaction(
            question, response, "anthropic", "claude-sonnet-4", 
            response_time=response_time
        )
        return response
    
    async def ask_groq_llama(self, question: str, **kwargs) -> str:
        """Groq Llama with session tracking."""
        if not self.available_providers["groq"]:
            raise ValueError("Groq API key not configured")
        
        start_time = asyncio.get_event_loop().time()
        response = await ask_groq_llama(question, **kwargs)
        response_time = asyncio.get_event_loop().time() - start_time
        
        await self._track_llm_interaction(
            question, response, "groq", "llama-3", 
            response_time=response_time
        )
        return response
    
    async def ask_parallel(
        self, 
        question: str, 
        providers: List[str] = None,
        include_timing: bool = True
    ) -> Dict[str, Any]:
        """Ask multiple providers in parallel with comprehensive tracking."""
        if providers is None:
            providers = [p for p, available in self.available_providers.items() if available]
        
        if not providers:
            raise ValueError("No providers available or configured")
        
        print(f"🔄 Asking {len(providers)} providers simultaneously...")
        
        # Create tasks for parallel execution
        tasks = []
        for provider in providers:
            if provider == "openai" and self.available_providers["openai"]:
                tasks.append(("openai", self.ask_openai_gpt4o_mini(question)))
            elif provider == "anthropic" and self.available_providers["anthropic"]:
                tasks.append(("anthropic", self.ask_anthropic_sonnet(question)))
            elif provider == "groq" and self.available_providers["groq"]:
                tasks.append(("groq", self.ask_groq_llama(question)))
        
        # Execute in parallel with timing
        start_time = asyncio.get_event_loop().time()
        results = {}
        timings = {}
        
        for provider, task in tasks:
            try:
                task_start = asyncio.get_event_loop().time()
                response = await task
                task_time = asyncio.get_event_loop().time() - task_start
                
                results[provider] = response
                timings[provider] = task_time
                
            except Exception as e:
                logger.error(f"❌ Error with {provider}: {e}")
                results[provider] = f"Error: {str(e)}"
                timings[provider] = None
        
        total_time = asyncio.get_event_loop().time() - start_time
        
        # Log parallel execution summary
        summary_event = SessionEvent(
            message={
                "parallel_execution": True,
                "providers": list(results.keys()),
                "total_time": total_time,
                "individual_timings": timings,
                "question": question
            },
            source=EventSource.SYSTEM,
            type=EventType.MESSAGE
        )
        await summary_event.set_metadata("timestamp", datetime.now().isoformat())
        await summary_event.set_metadata("execution_type", "parallel_llm_calls")
        await summary_event.set_metadata("provider_count", len(providers))
        
        await self.session.add_event_and_save(summary_event)
        
        return {
            "results": results,
            "timings": timings if include_timing else None,
            "total_time": total_time,
            "providers_used": list(results.keys())
        }


class AdvancedChukLLMClient:
    """Advanced CHUK LLM client with tool integration and comprehensive tracking."""
    
    def __init__(self, session: Session, provider: str, model: str):
        self.session = session
        self.provider = provider
        self.model = model
        self.client = get_client(provider=provider, model=model)
        self.system_prompt_generator = SystemPromptGenerator()
    
    async def create_completion_with_tools(
        self, 
        user_prompt: str, 
        system_context: Dict[str, Any] = None,
        tools: List[Dict] = None
    ) -> Dict[str, Any]:
        """Create completion with tools and comprehensive session tracking."""
        
        # Track user input
        user_event = await SessionEvent.create_with_tokens(
            message=user_prompt,
            prompt=user_prompt,
            model=self.model,
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await user_event.set_metadata("timestamp", datetime.now().isoformat())
        await user_event.set_metadata("provider", self.provider)
        await user_event.set_metadata("model", self.model)
        await user_event.set_metadata("has_tools", bool(tools))
        
        await self.session.add_event_and_save(user_event)
        
        # Generate system prompt
        system_prompt = self.system_prompt_generator.generate_prompt(system_context or {})
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Prepare completion arguments
        completion_kwargs = {"messages": messages}
        if tools:
            completion_kwargs["tools"] = tools
            completion_kwargs["tool_choice"] = "auto"
        
        # Execute LLM call with error handling
        start_time = asyncio.get_event_loop().time()
        try:
            completion = await self.client.create_completion(**completion_kwargs)
            response_time = asyncio.get_event_loop().time() - start_time
            
            # Extract response data
            response_content, tool_calls = self._extract_completion_data(completion)
            
            # Track LLM response
            llm_event = await SessionEvent.create_with_tokens(
                message={
                    "content": response_content,
                    "tool_calls": tool_calls,
                    "completion_raw": str(completion)[:500]  # Truncated raw response
                },
                prompt="",
                completion=response_content,
                model=self.model,
                source=EventSource.LLM,
                type=EventType.MESSAGE
            )
            await llm_event.set_metadata("timestamp", datetime.now().isoformat())
            await llm_event.set_metadata("provider", self.provider)
            await llm_event.set_metadata("model", self.model)
            await llm_event.set_metadata("response_time", response_time)
            await llm_event.set_metadata("has_tool_calls", len(tool_calls) > 0)
            await llm_event.set_metadata("tool_call_count", len(tool_calls))
            
            await self.session.add_event_and_save(llm_event)
            
            # Execute tools if present
            tool_results = []
            if tool_calls:
                tool_results = await self._execute_tools(tool_calls)
            
            return {
                "response": response_content,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "response_time": response_time,
                "session_id": self.session.id
            }
            
        except Exception as e:
            response_time = asyncio.get_event_loop().time() - start_time
            
            # Track error
            error_event = SessionEvent(
                message=f"LLM Error: {str(e)}",
                source=EventSource.SYSTEM,
                type=EventType.MESSAGE
            )
            await error_event.set_metadata("timestamp", datetime.now().isoformat())
            await error_event.set_metadata("provider", self.provider)
            await error_event.set_metadata("model", self.model)
            await error_event.set_metadata("error_type", type(e).__name__)
            await error_event.set_metadata("response_time", response_time)
            
            await self.session.add_event_and_save(error_event)
            raise
    
    def _extract_completion_data(self, completion) -> Tuple[str, List[Dict]]:
        """Extract response content and tool calls from completion."""
        if isinstance(completion, dict):
            response_content = completion.get("response", completion.get("content", ""))
            tool_calls = completion.get("tool_calls", [])
        elif hasattr(completion, 'choices') and completion.choices:
            choice = completion.choices[0]
            response_content = choice.message.content or ""
            tool_calls = []
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in choice.message.tool_calls
                ]
        else:
            response_content = str(completion) if completion is not None else ""
            tool_calls = []
        
        return response_content or "", tool_calls
    
    async def _execute_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls with session tracking."""
        tool_results = []
        
        for call in tool_calls:
            if isinstance(call, dict) and "function" in call:
                func = call["function"]
                tool_name = func.get("name", "")
                
                try:
                    arguments = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}
                
                # Execute tool (demo implementation)
                start_time = asyncio.get_event_loop().time()
                result = await self._demo_tool_execution(tool_name, arguments)
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Track tool execution
                tool_event = SessionEvent(
                    message={
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": result,
                        "call_id": call.get("id", "unknown"),
                        "execution_time": execution_time
                    },
                    source=EventSource.SYSTEM,
                    type=EventType.TOOL_CALL
                )
                await tool_event.set_metadata("timestamp", datetime.now().isoformat())
                await tool_event.set_metadata("tool_name", tool_name)
                await tool_event.set_metadata("execution_time", execution_time)
                await tool_event.set_metadata("success", result.get("success", True) if isinstance(result, dict) else True)
                
                await self.session.add_event_and_save(tool_event)
                
                tool_results.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "execution_time": execution_time,
                    "call_id": call.get("id", "unknown")
                })
        
        return tool_results
    
    async def _demo_tool_execution(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Demo tool execution with realistic responses."""
        await asyncio.sleep(0.1)  # Simulate tool execution time
        
        if tool_name == "get_weather":
            location = arguments.get("location", "Unknown")
            # Realistic weather data based on location
            weather_data = {
                "tokyo": {"temp": 18, "condition": "Cloudy", "humidity": 75},
                "london": {"temp": 12, "condition": "Rainy", "humidity": 85},
                "new york": {"temp": 15, "condition": "Partly cloudy", "humidity": 60},
                "sydney": {"temp": 22, "condition": "Sunny", "humidity": 55},
            }
            data = weather_data.get(location.lower(), {"temp": 20, "condition": "Clear", "humidity": 50})
            
            return {
                "location": location,
                "temperature": data["temp"],
                "condition": data["condition"],
                "humidity": data["humidity"],
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        elif tool_name == "calculate":
            expression = arguments.get("expression", "0")
            try:
                # Safe evaluation for demo
                safe_expr = expression.replace("×", "*").replace("÷", "/")
                result = eval(safe_expr)  # In production, use a safe math evaluator
                return {
                    "expression": expression,
                    "result": result,
                    "success": True
                }
            except Exception as e:
                return {
                    "expression": expression,
                    "error": str(e),
                    "success": False
                }
                
        elif tool_name == "search_web":
            query = arguments.get("query", "")
            return {
                "query": query,
                "results": [
                    {"title": f"Comprehensive guide to {query}", "url": "https://example.com/guide"},
                    {"title": f"Latest research on {query}", "url": "https://research.example.com"},
                    {"title": f"{query} - Best practices", "url": "https://bestpractices.example.com"}
                ],
                "results_count": 3,
                "success": True
            }
        
        return {"error": f"Unknown tool: {tool_name}", "success": False}


async def demo_simple_functions():
    """Demonstrate simple CHUK LLM functions with session tracking."""
    print("⚡ Simple CHUK LLM Functions with Session Tracking")
    print("=" * 55)
    
    # Setup session
    session = await Session.create()
    await session.metadata.set_property("demo", "simple_functions")
    
    llm = TrackedChukLLM(session)
    
    print(f"📋 Available providers: {[p for p, avail in llm.available_providers.items() if avail]}")
    
    if llm.available_providers["openai"]:
        print("\n🤖 OpenAI GPT-4o Mini:")
        response = await llm.ask_openai_gpt4o_mini("What's the most important programming concept?")
        print(f"   {response}")
    
    if llm.available_providers["anthropic"]:
        print("\n🧠 Anthropic Claude Sonnet:")
        response = await llm.ask_anthropic_sonnet("What's the key to good software architecture?")
        print(f"   {response}")
    
    # Show session stats
    stats = await get_session_stats(session)
    print(f"\n📊 Session Stats: {stats['events']} events, {stats['tokens']} tokens, ${stats['cost']:.6f}")


async def demo_parallel_providers():
    """Demonstrate parallel provider execution with timing analysis."""
    print("\n🔄 Parallel Provider Execution")
    print("=" * 35)
    
    session = await Session.create()
    await session.metadata.set_property("demo", "parallel_providers")
    
    llm = TrackedChukLLM(session)
    
    question = "Explain the future of AI in one concise paragraph."
    print(f"❓ Question: {question}")
    
    # Execute in parallel
    result = await llm.ask_parallel(question)
    
    print(f"\n⚡ Results from {len(result['providers_used'])} providers:")
    for provider, response in result["results"].items():
        timing = result["timings"].get(provider)
        timing_str = f" ({timing:.2f}s)" if timing else ""
        print(f"\n{provider.capitalize()}{timing_str}:")
        print(f"   {response}")
    
    print(f"\n⏱️ Total parallel execution time: {result['total_time']:.2f}s")


async def demo_advanced_client_with_tools():
    """Demonstrate advanced client with tool integration."""
    print("\n🛠️ Advanced Client with Tool Integration")
    print("=" * 45)
    
    # Check if we have OpenAI (best for tool calling)
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OpenAI API key required for tool demo")
        return
    
    session = await Session.create()
    await session.metadata.set_property("demo", "advanced_tools")
    
    client = AdvancedChukLLMClient(session, "openai", "gpt-4o-mini")
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    prompt = "What's the weather in Tokyo? Also calculate 25.5 * 18.7 for me."
    print(f"👤 User: {prompt}")
    
    result = await client.create_completion_with_tools(
        user_prompt=prompt,
        system_context={"current_date": "2025-06-18"},
        tools=tools
    )
    
    print(f"\n🤖 AI Response: {result['response']}")
    
    if result['tool_results']:
        print(f"\n🔧 Tool Executions:")
        for i, tool_result in enumerate(result['tool_results'], 1):
            tool_name = tool_result['tool']
            exec_time = tool_result.get('execution_time', 0)
            print(f"   {i}. {tool_name} ({exec_time:.3f}s)")
            
            result_data = tool_result['result']
            if tool_name == "get_weather" and result_data.get("success"):
                print(f"      🌤️ {result_data['location']}: {result_data['temperature']}°C, {result_data['condition']}")
            elif tool_name == "calculate" and result_data.get("success"):
                print(f"      🧮 {result_data['expression']} = {result_data['result']}")


async def demo_conversation_context():
    """Demonstrate conversation context management."""
    print("\n💬 Conversation Context Management")
    print("=" * 38)
    
    # Use SessionManager for conversation tracking
    sm = SessionManager()
    
    # Check available provider
    provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic" if os.getenv("ANTHROPIC_API_KEY") else None
    
    if not provider:
        print("⚠️ No API keys available for conversation demo")
        return
    
    print(f"🔄 Using {provider} for conversation context demo")
    
    # Conversation flow
    conversations = [
        "I'm building a Python web application for my startup.",
        "What database would you recommend for user data?",
        "How should I handle user authentication?",
        "What about deployment options for scaling?"
    ]
    
    # Use appropriate CHUK LLM function based on available provider
    for i, question in enumerate(conversations, 1):
        print(f"\n--- Turn {i} ---")
        print(f"👤 User: {question}")
        
        # Track user input
        await sm.user_says(question)
        
        # Get conversation history for context
        if i > 1:
            conversation = await sm.get_conversation()
            context = " ".join([
                f"{turn['role']}: {turn['content']}" 
                for turn in conversation[-4:]  # Last 4 messages
            ])
            contextual_prompt = f"Context: {context}\n\nCurrent question: {question}"
        else:
            contextual_prompt = question
        
        # Call appropriate provider
        try:
            if provider == "openai":
                response = await ask_openai_gpt4o_mini(contextual_prompt)
            else:
                response = await ask_anthropic_sonnet(contextual_prompt)
            
            # Track AI response
            await sm.ai_responds(response, model=f"{provider}-model", provider=provider)
            
            print(f"🤖 AI: {response}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    # Show conversation stats
    stats = await sm.get_stats()
    print(f"\n📊 Context-Aware Conversation Stats:")
    print(f"   Messages: {stats['user_messages']} user, {stats['ai_messages']} AI")
    print(f"   Tokens: {stats['total_tokens']}")
    print(f"   Cost: ${stats['estimated_cost']:.6f}")


async def demo_infinite_context():
    """Demonstrate infinite context with automatic segmentation."""
    print("\n🔄 Infinite Context with Auto-Segmentation")
    print("=" * 45)
    
    # Use SessionManager with infinite context
    sm = SessionManager(
        infinite_context=True,
        token_threshold=500,  # Low threshold to trigger segmentation
        max_turns_per_segment=3
    )
    
    provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic" if os.getenv("ANTHROPIC_API_KEY") else None
    
    if not provider:
        print("⚠️ No API keys available for infinite context demo")
        return
    
    # Extended technical discussion
    technical_questions = [
        "I want to build a machine learning pipeline for real-time data processing.",
        "What streaming technologies would work best for this use case?",
        "How should I handle model versioning and deployment in production?",
        "What monitoring and observability tools do you recommend?",
        "How can I implement A/B testing for my ML models?",
        "What's the best approach for handling data drift and model retraining?"
    ]
    
    print(f"🔄 Extended technical discussion with auto-segmentation...")
    
    for i, question in enumerate(technical_questions, 1):
        old_session = sm.session_id
        
        print(f"\n--- Turn {i} ---")
        print(f"👤 Question: {question}")
        
        # Track question
        await sm.user_says(question)
        
        # Build context for LLM call
        conversation = await sm.get_conversation()
        if len(conversation) > 1:
            context_messages = [
                f"{turn['role']}: {turn['content']}" 
                for turn in conversation[-6:]  # Include recent context
            ]
            contextual_prompt = "\n".join(context_messages) + f"\nUser: {question}"
        else:
            contextual_prompt = question
        
        # Call LLM with context
        try:
            if provider == "openai":
                response = await ask_openai_gpt4o_mini(contextual_prompt)
            else:
                response = await ask_anthropic_sonnet(contextual_prompt)
            
            # Track response
            await sm.ai_responds(response, model=f"{provider}-model", provider=provider)
            
            # Check for session transition
            if sm.session_id != old_session:
                print(f"   🔄 TRANSITION: {old_session[:8]}... → {sm.session_id[:8]}...")
            
            print(f"🤖 AI: {response[:150]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    # Show infinite context results
    stats = await sm.get_stats()
    session_chain = await sm.get_session_chain()
    
    print(f"\n📊 Infinite Context Results:")
    print(f"   Questions: {len(technical_questions)}")
    print(f"   Session segments: {stats['session_segments']}")
    print(f"   Session chain: {' → '.join([sid[:8] + '...' for sid in session_chain])}")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Total cost: ${stats['estimated_cost']:.6f}")


async def get_session_stats(session: Session) -> Dict[str, Any]:
    """Get session statistics."""
    return {
        "events": len(session.events),
        "tokens": session.total_tokens,
        "cost": session.total_cost
    }


async def main():
    """Main demo function."""
    print("🚀 REAL CHUK LLM DEMO - Production Integration")
    print("Multi-provider LLM integration with comprehensive session management")
    print("=" * 75)
    
    # Setup session storage
    setup_chuk_sessions_storage(sandbox_id="chuk-llm-demo", default_ttl_hours=2)
    
    # Check available providers
    available_providers = {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY"))
    }
    
    configured_count = sum(available_providers.values())
    print(f"🔑 API Keys configured: {configured_count}/3 providers")
    for provider, available in available_providers.items():
        status = "✅" if available else "❌"
        print(f"   {status} {provider.upper()}")
    
    if configured_count == 0:
        print("\n⚠️ No API keys found!")
        print("📋 Setup Instructions:")
        print("   1. Create a .env file in your project root")
        print("   2. Add your API keys:")
        print("      OPENAI_API_KEY=your-openai-key-here")
        print("      ANTHROPIC_API_KEY=your-anthropic-key-here")
        print("      GROQ_API_KEY=your-groq-key-here")
        return
    
    try:
        # Run demos
        await demo_simple_functions()
        
        if configured_count > 1:
            await demo_parallel_providers()
        
        if available_providers["openai"]:  # Tools work best with OpenAI
            await demo_advanced_client_with_tools()
        
        await demo_conversation_context()
        await demo_infinite_context()
        
        print(f"\n🎉 All CHUK LLM Demos Complete!")
        print("=" * 75)
        print("✅ Simple functions with session tracking")
        print("✅ Parallel provider execution and timing")
        print("✅ Advanced client with tool integration")
        print("✅ Conversation context management")
        print("✅ Infinite context with auto-segmentation")
        print("✅ Complete observability and cost tracking")
        print("✅ Production-ready error handling")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        print(f"\n❌ Demo error: {e}")
        print("   Check your API keys and internet connection")


if __name__ == "__main__":
    asyncio.run(main())