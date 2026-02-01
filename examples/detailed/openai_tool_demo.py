#!/usr/bin/env python3
"""
examples/clean_openai_demo.py
────────────────────────────
Clean OpenAI integration using registry auto-discovery instead of manual function generation.

This shows:
• Clean tool registration with decorators
• Automatic tool discovery from registry
• Simple OpenAI function schema generation
• Session-aware tool execution
• Complete session tracking
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Any
import inspect

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Session manager imports - FIXED for current architecture
from chuk_ai_session_manager.session_storage import (
    get_backend,
    ChukSessionsStore,
    setup_chuk_sessions_storage,
)
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType

# Tool processor imports
from chuk_tool_processor.registry import initialize, get_default_registry
from chuk_tool_processor.models.tool_call import ToolCall
from chuk_tool_processor.execution.strategies.inprocess_strategy import (
    InProcessStrategy,
)
from chuk_tool_processor.execution.tool_executor import ToolExecutor

# Import sample tools - this triggers auto-registration

# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Quiet down noisy loggers
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("anthropic").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("h11").setLevel(logging.ERROR)
logging.getLogger("chuk_llm").setLevel(logging.WARNING)
logging.getLogger("chuk_sessions").setLevel(logging.WARNING)
logging.getLogger("chuk_ai_session_manager").setLevel(logging.WARNING)
logging.getLogger("chuk_tool_processor").setLevel(logging.WARNING)


##############################################################################
# Clean Tool Processor with Registry Auto-Discovery
##############################################################################


class CleanSessionAwareToolProcessor:
    """Clean tool processor using registry auto-discovery."""

    def __init__(self, session_id: str, registry, executor):
        self.session_id = session_id
        self.registry = registry
        self.executor = executor

    @classmethod
    async def create(cls, session_id: str):
        """Create processor with auto-discovered tools."""
        registry = await get_default_registry()
        strategy = InProcessStrategy(registry)
        executor = ToolExecutor(registry=registry, strategy=strategy)
        return cls(session_id, registry, executor)

    async def process_llm_message(self, llm_msg: dict) -> list:
        """Process tool calls from LLM message."""
        backend = get_backend()
        store = ChukSessionsStore(backend)
        session = await store.get(self.session_id)

        # Log LLM message
        llm_event = await SessionEvent.create_with_tokens(
            message=llm_msg,
            prompt="",
            completion=json.dumps(llm_msg, ensure_ascii=False),
            model="gpt-4o-mini",
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
        await session.add_event_and_save(llm_event)

        # Extract and execute tool calls
        tool_calls = llm_msg.get("tool_calls", [])
        if not tool_calls:
            return []

        # Convert to ToolCall objects
        chuk_tool_calls = []
        for call in tool_calls:
            func = call.get("function", {})
            tool_name = func.get("name", "")
            try:
                arguments = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}

            chuk_tool_calls.append(ToolCall(tool=tool_name, arguments=arguments))

        # Execute tools
        results = await self.executor.execute(chuk_tool_calls)

        # Log each result
        for result in results:
            tool_event = await SessionEvent.create_with_tokens(
                message={
                    "tool": result.tool,
                    "arguments": getattr(result, "arguments", None),
                    "result": result.result,
                    "error": result.error,
                },
                prompt=f"{result.tool}({json.dumps(getattr(result, 'arguments', None), default=str)})",
                completion=str(result.result) if result.result is not None else "null",
                model="tool-execution",
                source=EventSource.SYSTEM,
                type=EventType.TOOL_CALL,
            )
            await tool_event.set_metadata("parent_event_id", llm_event.id)
            await session.add_event_and_save(tool_event)

        return results


##############################################################################
# Clean OpenAI Function Generation from Registry
##############################################################################


async def generate_openai_functions_from_registry(registry) -> List[Dict[str, Any]]:
    """Generate OpenAI function definitions from registry auto-discovery."""
    openai_tools = []

    # Get all registered tools
    tools_list = await registry.list_tools()
    print(f"🔧 Auto-discovered {len(tools_list)} tools from registry:")

    for namespace, tool_name in tools_list:
        try:
            # Get tool metadata and class
            metadata = await registry.get_metadata(tool_name, namespace)
            tool_class = await registry.get_tool(tool_name, namespace)

            print(f"   • {namespace}.{tool_name}: {metadata.description}")

            # Get execute method signature
            tool_instance = tool_class()
            execute_method = getattr(tool_instance, "execute")
            sig = inspect.signature(execute_method)

            # Create OpenAI function definition
            openai_func = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": metadata.description or f"Execute {tool_name}",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }

            # Extract parameters from method signature
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                # Determine parameter type from annotation
                param_type = "string"
                if param.annotation is int:
                    param_type = "integer"
                elif param.annotation is float:
                    param_type = "number"
                elif param.annotation is bool:
                    param_type = "boolean"

                # Add parameter
                openai_func["function"]["parameters"]["properties"][param_name] = {
                    "type": param_type,
                    "description": f"Parameter: {param_name}",
                }

                # Mark as required if no default value
                if param.default == inspect.Parameter.empty:
                    openai_func["function"]["parameters"]["required"].append(param_name)

            openai_tools.append(openai_func)

        except Exception as e:
            print(f"   ❌ Error processing {namespace}.{tool_name}: {e}")
            continue

    return openai_tools


##############################################################################
# Helper Functions
##############################################################################


async def get_openai_client() -> AsyncOpenAI:
    """Get OpenAI client with API key verification."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in environment or .env file")

    client = AsyncOpenAI(api_key=api_key)

    # Verify connection
    try:
        await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1,
        )
        print("✅ OpenAI client verified")
    except Exception as e:
        print(f"❌ OpenAI client error: {e}")
        raise

    return client


async def pretty_print_session_tree(session: Session) -> None:
    """Pretty print the session event tree."""
    children = {}
    for evt in session.events:
        parent = await evt.get_metadata("parent_event_id")
        if parent:
            children.setdefault(parent, []).append(evt)

    async def _print_event(evt: SessionEvent, depth: int = 0) -> None:
        pad = "  " * depth
        print(f"{pad}• {evt.type.value:10} [{evt.id[:8]}...]")

        if evt.type == EventType.TOOL_CALL and evt.message:
            tool_name = evt.message.get("tool", "unknown")
            error = evt.message.get("error")
            result = evt.message.get("result")

            print(f"{pad}  ↳ {tool_name}")
            if error:
                print(f"{pad}    ❌ {error}")
            elif result and isinstance(result, dict):
                if tool_name == "weather":
                    print(
                        f"{pad}    🌤️ {result.get('location')}: {result.get('temperature')}°C, {result.get('condition')}"
                    )
                elif tool_name == "calculator":
                    print(
                        f"{pad}    🧮 {result.get('a')} {result.get('operation')} {result.get('b')} = {result.get('result')}"
                    )
                elif tool_name == "search":
                    print(
                        f"{pad}    🔍 '{result.get('query')}': {result.get('results_count')} results"
                    )
                else:
                    print(f"{pad}    ✅ Success")
            else:
                print(f"{pad}    ✅ Success")
        elif evt.type == EventType.MESSAGE:
            content = str(evt.message)[:60]
            print(
                f"{pad}  Content: {content}{'...' if len(str(evt.message)) > 60 else ''}"
            )

        # Print children
        for child in sorted(children.get(evt.id, []), key=lambda e: e.timestamp):
            await _print_event(child, depth + 1)

    # Print root events
    roots = [e for e in session.events if not await e.get_metadata("parent_event_id")]
    for root in sorted(roots, key=lambda e: e.timestamp):
        await _print_event(root)


##############################################################################
# Main Demo
##############################################################################


async def main() -> None:
    """Run the clean OpenAI demo."""
    print("🚀 Clean OpenAI Demo with Registry Auto-Discovery")
    print("=" * 60)

    try:
        # Initialize tool registry
        print("\n🔧 Initializing tool registry...")
        registry = await initialize()

        # Generate OpenAI functions from registry
        openai_tools = await generate_openai_functions_from_registry(registry)

        if not openai_tools:
            print("❌ No tools available")
            return

        # Setup OpenAI client
        client = await get_openai_client()

        # Setup session manager with CHUK Sessions backend
        setup_chuk_sessions_storage(sandbox_id="clean-openai-demo", default_ttl_hours=1)
        backend = get_backend()
        store = ChukSessionsStore(backend)

        session = await Session.create()
        await session.metadata.set_property("demo", "clean_openai_integration")
        await session.metadata.set_property("provider", "openai")
        await store.save(session)

        # Create clean tool processor
        processor = await CleanSessionAwareToolProcessor.create(session_id=session.id)

        # User request
        user_prompt = (
            "I need to know the weather in Tokyo and calculate 15.5 × 23.2. "
            "Also search for information about renewable energy."
        )

        print("\n👤 USER REQUEST:")
        print(f"   {user_prompt}")

        # Add user event
        user_event = await SessionEvent.create_with_tokens(
            message=user_prompt,
            prompt=user_prompt,
            model="gpt-4o-mini",
            source=EventSource.USER,
        )
        await session.add_event_and_save(user_event)

        # Call OpenAI
        print(f"\n🤖 Calling OpenAI with {len(openai_tools)} auto-discovered tools...")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_prompt}],
            tools=openai_tools,
            tool_choice="auto",
            temperature=0.7,
        )

        # Process response
        assistant_msg = response.choices[0].message.model_dump()
        tool_calls = assistant_msg.get("tool_calls", [])

        print(f"\n📞 LLM wants to call {len(tool_calls)} tools:")
        for call in tool_calls:
            func = call.get("function", {})
            print(f"   • {func.get('name')}({func.get('arguments')})")

        # Execute tools through clean processor
        if tool_calls:
            print("\n🔧 Executing tools...")
            tool_results = await processor.process_llm_message(assistant_msg)

            print("\n✅ Tool Results:")
            for i, result in enumerate(tool_results, 1):
                print(f"\n   Tool {i}: {result.tool}")
                if result.error:
                    print(f"   ❌ Error: {result.error}")
                elif isinstance(result.result, dict):
                    if result.tool == "weather":
                        r = result.result
                        print(
                            f"   🌤️ {r.get('location')}: {r.get('temperature')}°C, {r.get('condition')}"
                        )
                        print(
                            f"       Humidity: {r.get('humidity')}%, Wind: {r.get('wind_speed')} km/h"
                        )
                    elif result.tool == "calculator":
                        r = result.result
                        print(
                            f"   🧮 {r.get('a')} {r.get('operation')} {r.get('b')} = {r.get('result')}"
                        )
                    elif result.tool == "search":
                        r = result.result
                        print(f"   🔍 Query: '{r.get('query')}'")
                        print(f"       Found {r.get('results_count')} results:")
                        for j, res in enumerate(r.get("results", [])[:2], 1):
                            print(f"         {j}. {res.get('title')}")
                            print(f"            {res.get('url')}")
                    else:
                        print(f"   📊 Result: {result.result}")
                else:
                    print(f"   📊 Result: {result.result}")

        # Refresh session from store to get all events
        session = await store.get(session.id)

        # Show session tree
        print("\n📊 Session Event Tree:")
        print("=" * 40)
        await pretty_print_session_tree(session)

        # Show token usage
        if session.total_tokens > 0:
            print("\n💰 Token Usage:")
            print(f"   Total tokens: {session.total_tokens}")
            print(f"   Estimated cost: ${session.total_cost:.6f}")

            for model, usage in session.token_summary.usage_by_model.items():
                print(
                    f"   📊 {model}: {usage.total_tokens} tokens (${usage.estimated_cost_usd:.6f})"
                )

        # Show session statistics
        print("\n📈 Session Statistics:")
        print(f"   Session ID: {session.id}")
        print(f"   Total events: {len(session.events)}")
        print(f"   Created: {session.metadata.created_at}")
        print(f"   Updated: {session.metadata.updated_at}")

        # Event breakdown
        event_types = {}
        for event in session.events:
            event_type = f"{event.source.value}:{event.type.value}"
            event_types[event_type] = event_types.get(event_type, 0) + 1

        print("   Event breakdown:")
        for event_type, count in event_types.items():
            print(f"     {event_type}: {count}")

        print("\n🎉 Clean demo completed successfully!")
        print("=" * 60)
        print("🎯 Key Achievements:")
        print("  • Auto-discovered tools from registry")
        print("  • Clean OpenAI function generation")
        print("  • Session-aware tool execution")
        print("  • Complete conversation tracking")
        print("  • Token usage and cost analytics")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


##############################################################################
# Setup logging
##############################################################################

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

# Quiet down noisy loggers
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

if __name__ == "__main__":
    asyncio.run(main())
