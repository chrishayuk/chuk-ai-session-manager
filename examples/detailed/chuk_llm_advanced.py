#!/usr/bin/env python3
"""
CHUK LLM Advanced API Integration with Session Management
========================================================

This example demonstrates integration with chuk-llm's advanced client API,
providing complete session management, tool integration, and observability
for production AI applications.

Run with:
    uv run examples/chuk_llm_advanced_demo.py \
        --provider openai \
        --model gpt-4o-mini \
        --prompt "I need to know the weather in Tokyo and calculate 15.5 × 23.2"

For completely clean output (no debug logs):
    PYTHONDEVMODE=0 uv run examples/chuk_llm_advanced_demo.py

Features demonstrated:
- Advanced chuk-llm client integration
- Session-aware tool processing
- Complete message and tool tracking
- Token usage and cost monitoring
- Production-ready error handling
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List
from datetime import datetime
from dotenv import load_dotenv

# CHUK LLM imports
from chuk_llm.llm.client import get_client
from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator

# Session manager imports - FIXED to work with current architecture
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.session_storage import (
    get_backend,
    ChukSessionsStore,
    setup_chuk_sessions_storage,
)

# Load environment variables
load_dotenv()

# Set up logging with quieter external loggers
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Quiet down noisy loggers
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("anthropic").setLevel(logging.ERROR)
logging.getLogger("chuk_llm").setLevel(logging.WARNING)
logging.getLogger("chuk_sessions").setLevel(logging.WARNING)
logging.getLogger("chuk_ai_session_manager").setLevel(logging.WARNING)


class SessionAwareLLMClient:
    """Wrapper around chuk-llm client that adds comprehensive session tracking."""

    def __init__(self, session: Session, provider: str, model: str):
        self.session = session
        self.provider = provider
        self.model = model
        self.client = get_client(provider=provider, model=model)
        self.system_prompt_generator = SystemPromptGenerator()

    async def create_completion_with_tools(
        self, user_prompt: str, system_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create completion with tool support and session tracking."""

        # Log user message
        user_event = await SessionEvent.create_with_tokens(
            message=user_prompt,
            prompt=user_prompt,
            model=self.model,
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
        await user_event.set_metadata("timestamp", self._get_timestamp())
        await user_event.set_metadata("provider", self.provider)
        await user_event.set_metadata("model", self.model)
        await user_event.set_metadata("type", "user_input")

        await self.session.add_event_and_save(user_event)

        # Generate system prompt
        system_prompt = self.system_prompt_generator.generate_prompt(
            system_context or {}
        )

        # Get available tools (simulated for demo)
        available_tools = self._get_demo_tools()
        if available_tools:
            print(
                f"🔧 Available tools: {[tool['function']['name'] for tool in available_tools]}"
            )

        # Prepare messages
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Add tools to the completion request if available
        completion_kwargs = {"messages": messages}
        if available_tools:
            completion_kwargs["tools"] = available_tools
            completion_kwargs["tool_choice"] = "auto"

        # Call LLM
        print(f"🤖 Calling {self.provider}/{self.model}...")
        try:
            completion = await self.client.create_completion(**completion_kwargs)

            # Extract response and tool calls
            response_content, tool_calls = self._extract_completion_data(completion)

        except Exception as e:
            # Log error
            error_event = SessionEvent(
                message=f"LLM Error: {str(e)}",
                source=EventSource.SYSTEM,
                type=EventType.MESSAGE,  # Using MESSAGE since ERROR doesn't exist
            )
            await error_event.set_metadata("timestamp", self._get_timestamp())
            await error_event.set_metadata("provider", self.provider)
            await error_event.set_metadata("model", self.model)
            await error_event.set_metadata("error_type", type(e).__name__)

            await self.session.add_event_and_save(error_event)
            raise

        # Log LLM response
        llm_event = await SessionEvent.create_with_tokens(
            message={"content": response_content, "tool_calls": tool_calls},
            prompt="",  # No prompt for LLM response
            completion=response_content,
            model=self.model,
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
        await llm_event.set_metadata("timestamp", self._get_timestamp())
        await llm_event.set_metadata("provider", self.provider)
        await llm_event.set_metadata("model", self.model)
        await llm_event.set_metadata("has_tool_calls", len(tool_calls) > 0)

        await self.session.add_event_and_save(llm_event)

        # Process tool calls if any
        tool_results = []
        if tool_calls:
            print(f"🔧 Processing {len(tool_calls)} tool calls...")
            tool_results = await self._execute_tools(tool_calls)

        return {
            "response": response_content,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "session_id": self.session.id,
        }

    def _extract_completion_data(self, completion) -> tuple[str, List[Dict]]:
        """Extract response content and tool calls from completion."""
        if isinstance(completion, dict):
            response_content = completion.get("response", completion.get("content", ""))
            tool_calls = completion.get("tool_calls", [])
        elif hasattr(completion, "choices") and completion.choices:
            # OpenAI-style response
            choice = completion.choices[0]
            response_content = choice.message.content or ""
            tool_calls = []
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]
        else:
            response_content = str(completion) if completion is not None else ""
            tool_calls = []

        # Ensure response_content is never None
        if response_content is None:
            response_content = ""

        return response_content, tool_calls

    def _get_demo_tools(self) -> List[Dict[str, Any]]:
        """Get demo tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city or location to get weather for",
                            }
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate (e.g., '15.5 * 23.2')",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 3,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    async def _execute_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls and log results (demo implementation)."""
        tool_results = []

        for call in tool_calls:
            if isinstance(call, dict) and "function" in call:
                func = call["function"]
                tool_name = func.get("name", "")

                try:
                    arguments = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}

                # Demo tool execution (replace with actual tool integration)
                result = await self._demo_tool_execution(tool_name, arguments)

                # Log tool execution
                tool_event = SessionEvent(
                    message={
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": result,
                        "call_id": call.get("id", "unknown"),
                    },
                    source=EventSource.SYSTEM,
                    type=EventType.TOOL_CALL,
                )
                await tool_event.set_metadata("timestamp", self._get_timestamp())
                await tool_event.set_metadata("tool_name", tool_name)
                await tool_event.set_metadata(
                    "success",
                    result.get("success", True) if isinstance(result, dict) else True,
                )

                await self.session.add_event_and_save(tool_event)

                tool_results.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": result,
                        "call_id": call.get("id", "unknown"),
                    }
                )

        return tool_results

    async def _demo_tool_execution(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Demo tool execution (replace with actual tool integration)."""
        if tool_name == "get_weather":
            location = arguments.get("location", "Unknown")
            return {
                "location": location,
                "temperature": 22,
                "condition": "Partly cloudy",
                "humidity": 65,
                "success": True,
            }
        elif tool_name == "calculate":
            expression = arguments.get("expression", "0")
            try:
                # Simple evaluation (in production, use a safe math evaluator)
                result = eval(expression.replace("×", "*").replace("÷", "/"))
                return {"expression": expression, "result": result, "success": True}
            except Exception as e:
                return {"expression": expression, "error": str(e), "success": False}
        elif tool_name == "search_web":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 3)
            return {
                "query": query,
                "results": [
                    {
                        "title": f"Result 1 for '{query}'",
                        "url": "https://example.com/1",
                    },
                    {
                        "title": f"Result 2 for '{query}'",
                        "url": "https://example.com/2",
                    },
                    {
                        "title": f"Result 3 for '{query}'",
                        "url": "https://example.com/3",
                    },
                ][:max_results],
                "results_count": min(3, max_results),
                "success": True,
            }
        else:
            return {"error": f"Unknown tool: {tool_name}", "success": False}

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()


async def run_session_aware_llm(provider: str, model: str, prompt: str) -> None:
    """Run LLM with complete session tracking and tool integration."""

    # Validate environment
    if provider.lower() == "openai" and not os.getenv("OPENAI_API_KEY"):
        sys.exit("[ERROR] OPENAI_API_KEY environment variable is not set")

    print("🚀 CHUK LLM Advanced API + Session Manager Integration")
    print("=" * 60)

    # Setup session manager with CHUK Sessions backend
    setup_chuk_sessions_storage(sandbox_id="chuk-llm-demo", default_ttl_hours=1)

    # Create session with metadata
    session = await Session.create()
    await session.metadata.set_property("example", "advanced_llm_demo")
    await session.metadata.set_property("provider", provider)
    await session.metadata.set_property("model", model)
    await session.metadata.set_property(
        "description", "Advanced LLM integration with tools and session tracking"
    )

    # Save the session with updated metadata
    backend = get_backend()
    store = ChukSessionsStore(backend)
    await store.save(session)

    print(f"📝 Created session: {session.id}")

    # Create session-aware LLM client
    client = SessionAwareLLMClient(session=session, provider=provider, model=model)

    # Process the request
    print(f"\n👤 User: {prompt}")

    # Track events before the call
    events_before = len(session.events)

    # Make the completion call
    try:
        result = await client.create_completion_with_tools(
            user_prompt=prompt,
            system_context={
                "current_date": "2025-06-17",
                "user_timezone": "UTC",
                "capabilities": ["weather", "calculation", "web_search"],
            },
        )

        # Display results
        print("\n🤖 AI Response:")
        response_text = result.get("response", "")
        if response_text and response_text.strip():
            print(f"   {response_text}")
        else:
            print("   (Response generated through tool usage)")

        # Display tool results if any
        if result.get("tool_results"):
            print("\n🛠️ Tool Execution Results:")
            for i, tool_result in enumerate(result["tool_results"], 1):
                tool_name = tool_result.get("tool", "unknown")
                result_data = tool_result.get("result", {})

                print(f"   Tool {i}: {tool_name}")

                if isinstance(result_data, dict):
                    if tool_name == "get_weather" and result_data.get("success"):
                        print(
                            f"   🌤️ {result_data['location']}: {result_data['temperature']}°C, {result_data['condition']}"
                        )
                        print(f"       Humidity: {result_data['humidity']}%")
                    elif tool_name == "calculate" and result_data.get("success"):
                        print(
                            f"   🧮 {result_data['expression']} = {result_data['result']}"
                        )
                    elif tool_name == "search_web" and result_data.get("success"):
                        print(f"   🔍 Query: '{result_data['query']}'")
                        print(f"       Found {result_data['results_count']} results:")
                        for idx, res in enumerate(result_data["results"], 1):
                            print(f"       {idx}. {res['title']}")
                    elif not result_data.get("success"):
                        print(
                            f"   ❌ Error: {result_data.get('error', 'Unknown error')}"
                        )
                    else:
                        print(f"   ✅ Result: {result_data}")
                else:
                    print(f"   ✅ Result: {result_data}")

        # Get updated session with latest events
        try:
            fresh_session = await store.get(session.id)
            if fresh_session and hasattr(fresh_session, "events"):
                session = fresh_session
        except Exception:
            # If we can't refresh, the current session object should still have the events
            pass

        events_added = len(session.events) - events_before

        # Show session statistics
        print("\n📊 Session Statistics:")
        print("=" * 40)
        print(f"   Session ID: {session.id}")
        print(f"   Total events: {len(session.events)}")
        print(f"   Events this call: {events_added}")
        print(f"   Created: {session.metadata.created_at}")
        print(f"   Updated: {session.metadata.updated_at}")
        print(f"   Total tokens: {session.total_tokens}")
        print(f"   Estimated cost: ${session.total_cost:.6f}")

        # Show event breakdown
        event_types = {}
        for event in session.events:
            event_type = f"{event.source.value}"
            if event.type != EventType.MESSAGE:
                event_type += f":{event.type.value}"
            event_types[event_type] = event_types.get(event_type, 0) + 1

        if event_types:
            print("\n📈 Event Breakdown:")
            for event_type, count in event_types.items():
                print(f"   {event_type}: {count}")

        # Show detailed event tree
        print("\n📝 Session Event Tree:")
        print("=" * 40)
        for i, event in enumerate(session.events, 1):
            source_emoji = {"user": "👤", "llm": "🤖", "system": "🔧"}.get(
                event.source.value, "❓"
            )

            event_type_str = (
                f"[{event.type.value}]" if event.type != EventType.MESSAGE else ""
            )

            if event.type == EventType.TOOL_CALL:
                tool_name = "unknown"
                if isinstance(event.message, dict):
                    tool_name = event.message.get("tool", "unknown")
                print(f"   {i}. {source_emoji} {event_type_str} {tool_name}")
                if isinstance(event.message, dict) and "arguments" in event.message:
                    args = event.message["arguments"]
                    print(f"       Args: {args}")
            else:
                content = str(event.message)
                if isinstance(event.message, dict):
                    content = event.message.get("content", str(event.message))
                if len(content) > 60:
                    content = content[:57] + "..."
                print(f"   {i}. {source_emoji} {event_type_str} {content}")

            # Show timestamp from metadata
            timestamp = await event.get_metadata("timestamp", "Unknown")
            print(f"       ⏰ {timestamp}")

        print("\n✅ Advanced LLM integration demo completed!")
        print("   Complete session tracking with tools and observability")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        print(f"❌ Error: {e}")

        # Log the error to session
        error_event = SessionEvent(
            message=f"Demo failed: {str(e)}",
            source=EventSource.SYSTEM,
            type=EventType.MESSAGE,
        )
        await error_event.set_metadata(
            "timestamp",
            client._get_timestamp()
            if "client" in locals()
            else datetime.now().isoformat(),
        )
        await error_event.set_metadata("error_type", type(e).__name__)
        await error_event.set_metadata("provider", provider)
        await error_event.set_metadata("model", model)

        await session.add_event_and_save(error_event)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CHUK LLM Advanced API + Session Manager Integration"
    )
    parser.add_argument(
        "--provider", default="openai", help="LLM provider (openai, anthropic)"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument(
        "--prompt",
        default="I need to know the weather in Tokyo and calculate 15.5 × 23.2. Also search for information about renewable energy.",
        help="Prompt text",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_session_aware_llm(args.provider, args.model, args.prompt))
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
