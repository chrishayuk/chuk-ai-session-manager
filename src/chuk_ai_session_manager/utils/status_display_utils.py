#!/usr/bin/env python3
"""
examples/retry_prompt_demo_fixed.py
──────────────────────────────────
Demonstrates LLM-level retry patterns with chuk_ai_session_manager.

This shows:
• Retrying LLM calls until they produce valid tool calls
• Using chuk_tool_processor's built-in reliability features
• Session tracking of the entire retry process
• Proper separation: LLM retries vs tool execution reliability
"""

from __future__ import annotations

import asyncio
import json
import logging
import pprint
import sys
import os
from typing import Dict, List

# Add current directory to path
sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Session imports - FIXED import paths
from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore, setup_chuk_sessions_storage
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.session_prompt_builder import build_prompt_from_session

# Status display utilities
def format_status(success: bool, success_msg: str = "SUCCESS", failure_msg: str = "FAILED") -> str:
    """Format status with correct emoji."""
    if success:
        return f"✅ {success_msg}"
    else:
        return f"❌ {failure_msg}"

# Import from chuk_tool_processor (using the working pattern)
from chuk_tool_processor.registry import initialize, get_default_registry
from chuk_tool_processor.models.tool_call import ToolCall
from chuk_tool_processor.execution.strategies.inprocess_strategy import InProcessStrategy
from chuk_tool_processor.execution.tool_executor import ToolExecutor

# Import sample tools - this will trigger registration
import sample_tools

##############################################################################
# Custom Tool Processor (based on working OpenAI demo pattern)
##############################################################################

class CustomSessionAwareToolProcessor:
    """Custom tool processor that properly integrates with chuk_tool_processor."""
    
    def __init__(self, session_id: str, registry, executor):
        self.session_id = session_id
        self.registry = registry
        self.executor = executor
    
    @classmethod
    async def create(cls, session_id: str):
        """Create a custom session-aware tool processor."""
        # Get the registry
        registry = await get_default_registry()
        
        # Create execution strategy and executor
        strategy = InProcessStrategy(registry)
        executor = ToolExecutor(registry=registry, strategy=strategy)
        
        return cls(session_id, registry, executor)
    
    async def process_llm_message(self, llm_msg: dict) -> list:
        """Process tool calls from an LLM message."""
        # Get the session
        backend = get_backend()
        store = ChukSessionsStore(backend)
        session = await store.get(self.session_id)
        if not session:
            raise ValueError(f"Session {self.session_id} not found")
        
        # Add the LLM message as an event
        llm_event = await SessionEvent.create_with_tokens(
            message=llm_msg,
            prompt="",
            completion=json.dumps(llm_msg, ensure_ascii=False),
            model="gpt-4o-mini",
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
        await session.add_event_and_save(llm_event)
        
        # Extract tool calls
        tool_calls = llm_msg.get('tool_calls', [])
        if not tool_calls:
            return []
        
        # Convert to ToolCall objects
        chuk_tool_calls = []
        for call in tool_calls:
            func = call.get('function', {})
            tool_name = func.get('name', '')
            try:
                arguments = json.loads(func.get('arguments', '{}'))
            except json.JSONDecodeError:
                arguments = {}
            
            chuk_tool_calls.append(ToolCall(
                tool=tool_name,
                arguments=arguments
            ))
        
        # Execute the tools
        print(f"🔧 Executing {len(chuk_tool_calls)} tools...")
        results = await self.executor.execute(chuk_tool_calls)
        
        # Log each result as a session event
        for result in results:
            # Convert result to string for session storage
            result_str = str(result.result) if result.result is not None else "null"
            
            tool_event = await SessionEvent.create_with_tokens(
                message={
                    "tool": result.tool,
                    "arguments": getattr(result, "arguments", None),
                    "result": result.result,
                    "error": result.error,
                },
                prompt=f"{result.tool}({json.dumps(getattr(result, 'arguments', None), default=str)})",
                completion=result_str,
                model="tool-execution",
                source=EventSource.SYSTEM,
                type=EventType.TOOL_CALL,
            )
            await tool_event.set_metadata("parent_event_id", llm_event.id)
            await session.add_event_and_save(tool_event)
        
        return results

##############################################################################
# LLM Simulation: Unreliable at first, then cooperative
##############################################################################

class UnreliableLLM:
    """Simulates an LLM that sometimes doesn't follow tool-calling instructions."""
    
    def __init__(self):
        self.call_count = 0
        self.scenarios = [
            # Scenario 1: Refuses to use tools
            {
                "role": "assistant", 
                "content": "I don't need to use any tools. The weather in London is probably fine!",
                "tool_calls": []
            },
            # Scenario 2: Tries to use non-existent tool
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function", 
                        "function": {
                            "name": "nonexistent_weather_api",
                            "arguments": '{"city": "London"}'
                        }
                    }
                ]
            },
            # Scenario 3: Invalid JSON in arguments
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "weather", 
                            "arguments": '{"location": London}'  # Missing quotes - invalid JSON
                        }
                    }
                ]
            },
            # Scenario 4: Finally cooperates correctly
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"location": "London"}'
                        }
                    }
                ]
            }
        ]
    
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """Simulate OpenAI chat completion with unreliable behavior."""
        self.call_count += 1
        
        if self.call_count <= len(self.scenarios):
            response = self.scenarios[self.call_count - 1]
            print(f"   📞 LLM Call {self.call_count}: {self._describe_response(response)}")
            return response
        else:
            # After all scenarios, always cooperate
            return self.scenarios[-1]
    
    def _describe_response(self, response: Dict) -> str:
        """Describe what the LLM response contains."""
        if response.get("tool_calls"):
            tool_calls = response["tool_calls"]
            if len(tool_calls) == 1:
                func_name = tool_calls[0].get("function", {}).get("name", "unknown")
                return f"Wants to call '{func_name}'"
            else:
                return f"Wants to call {len(tool_calls)} tools"
        elif response.get("content"):
            return f"Text response: '{response['content'][:50]}...'"
        else:
            return "Empty response"

##############################################################################
# Retry Logic for LLM Cooperation
##############################################################################

class LLMRetryManager:
    """Manages retrying LLM calls until they produce valid, executable tool calls."""
    
    def __init__(self, session_id: str, max_attempts: int = 5):
        self.session_id = session_id
        self.max_attempts = max_attempts
    
    async def get_valid_tool_calls(self, llm, messages: List[Dict], processor: CustomSessionAwareToolProcessor) -> tuple[Dict, List]:
        """
        Keep calling the LLM until it produces valid, executable tool calls.
        
        Returns:
            Tuple of (successful_llm_response, tool_results)
        """
        backend = get_backend()
        store = ChukSessionsStore(backend)
        session = await store.get(self.session_id)
        
        for attempt in range(1, self.max_attempts + 1):
            print(f"\n🔄 LLM Attempt {attempt}/{self.max_attempts}")
            
            # Call LLM
            response = await llm.chat_completion(messages)
            
            # Log the LLM response attempt
            attempt_event = SessionEvent(
                message={
                    "attempt": attempt,
                    "response": response,
                    "success": False  # Will update if successful
                },
                type=EventType.MESSAGE,
                source=EventSource.LLM,
            )
            await session.add_event_and_save(attempt_event)
            
            # Check if response has tool calls
            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                print(f"   {format_status(False, failure_msg='No tool calls in response')}")
                continue
            
            # Try to execute the tool calls
            try:
                print(f"   🔧 Attempting to execute {len(tool_calls)} tool calls...")
                
                # Check what tools are available vs requested
                registry = await get_default_registry()
                tools_list = await registry.list_tools()
                available_tools = [name for namespace, name in tools_list]
                requested_tool = tool_calls[0].get("function", {}).get("name", "unknown")
                print(f"   🔍 Requested tool: {requested_tool}")
                print(f"   🔍 Available tools: {available_tools}")
                
                tool_results = await processor.process_llm_message(response)
                
                # Check if all tools executed successfully
                failed_tools = [r for r in tool_results if r.error]
                if failed_tools:
                    print(f"   {format_status(False, failure_msg=f'{len(failed_tools)} tools failed:')}")
                    for failed in failed_tools:
                        print(f"      • {failed.tool}: {failed.error}")
                    continue
                
                # Success! All tools executed
                print(f"   {format_status(True, success_msg=f'All {len(tool_results)} tools executed successfully')}")
                
                # Update the last event to mark success
                session = await store.get(self.session_id)
                if session.events:
                    # Find the most recent LLM attempt event
                    for event in reversed(session.events):
                        if (event.type == EventType.MESSAGE and 
                            event.source == EventSource.LLM and
                            isinstance(event.message, dict) and
                            "attempt" in event.message):
                            event.message["success"] = True
                            await store.save(session)
                            break
                
                return response, tool_results
                
            except Exception as e:
                print(f"   {format_status(False, failure_msg=f'Tool execution failed: {e}')}")
                continue
        
        # If we get here, all attempts failed
        raise RuntimeError(f"Failed to get valid tool calls after {self.max_attempts} attempts")

##############################################################################
# Demo Flow
##############################################################################

async def main() -> None:
    print("🚀 Starting LLM Retry Demo")
    print("   (Demonstrates retry logic for uncooperative LLMs)")
    print("   (Tool execution uses chuk_tool_processor's built-in reliability)")
    
    # Setup session storage - FIXED
    setup_chuk_sessions_storage(sandbox_id="retry-prompt-demo", default_ttl_hours=1)
    backend = get_backend()
    store = ChukSessionsStore(backend)
    
    # Initialize tool registry first
    print("\n🔧 Initializing tool registry...")
    registry = await initialize()
    tools_list = await registry.list_tools()
    print(f"📋 Found {len(tools_list)} registered tools:")
    for namespace, tool_name in tools_list:
        print(f"   • {namespace}.{tool_name}")
    
    # Create session
    session = await Session.create()
    await session.metadata.set_property("demo", "retry_prompt")
    await store.save(session)
    
    # Add user request
    user_prompt = "What's the weather like in London? I need to know if I should bring an umbrella."
    user_event = await SessionEvent.create_with_tokens(
        message=user_prompt,
        prompt=user_prompt,
        model="gpt-4o-mini",
        source=EventSource.USER,
        type=EventType.MESSAGE
    )
    await session.add_event_and_save(user_event)
    print(f"\n👤 User: {user_prompt}")
    
    # Create components
    llm = UnreliableLLM()
    processor = await CustomSessionAwareToolProcessor.create(session_id=session.id)
    retry_manager = LLMRetryManager(session_id=session.id, max_attempts=6)
    
    # Build initial messages for LLM
    messages = [
        {"role": "system", "content": "You are a helpful assistant. When users ask about weather, use the weather tool to get current information."},
        {"role": "user", "content": user_prompt}
    ]
    
    # Attempt to get valid tool calls with retries
    try:
        print(f"\n🎯 Attempting to get valid tool calls (max {retry_manager.max_attempts} attempts)...")
        final_response, tool_results = await retry_manager.get_valid_tool_calls(llm, messages, processor)
        
        print(f"\n{'='*60}")
        print("🎉 SUCCESS! LLM cooperated and tools executed successfully")
        print(f"{'='*60}")
        
        # Show tool results
        print("\n🛠️ Tool Results:")
        for i, result in enumerate(tool_results, 1):
            print(f"\n   Tool {i}: {result.tool}")
            if result.error:
                print(f"   ❌ Error: {result.error}")
            elif isinstance(result.result, dict):
                print(f"   📊 Result:")
                for key, value in result.result.items():
                    print(f"      {key}: {value}")
            else:
                print(f"   📊 Result: {result.result}")
        
    except RuntimeError as e:
        print(f"\n❌ FAILED: {e}")
        
        # Still show the session events for debugging
        print("\n🔍 Debugging: Session events created:")
        session = await store.get(session.id)
        for i, event in enumerate(session.events, 1):
            print(f"   {i}. {event.type.value}/{event.source.value}: {str(event.message)[:100]}...")
        return
    
    # Show session event tree
    session = await store.get(session.id)
    print(f"\n{'='*60}")
    print("📊 Session Event Tree (Complete Retry History):")
    print(f"{'='*60}")
    
    for i, event in enumerate(session.events, 1):
        event_id = event.id[:8] + "..."
        if event.type == EventType.MESSAGE and event.source == EventSource.USER:
            print(f"{i}. USER MESSAGE [{event_id}]")
            print(f"   Content: {event.message}")
        elif event.type == EventType.MESSAGE and event.source == EventSource.LLM:
            if isinstance(event.message, dict) and "attempt" in event.message:
                attempt = event.message["attempt"]
                success = event.message.get("success", False)
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"{i}. LLM ATTEMPT {attempt} [{event_id}] - {status}")
            else:
                print(f"{i}. LLM MESSAGE [{event_id}]")
        elif event.type == EventType.TOOL_CALL:
            tool_msg = event.message or {}
            tool_name = tool_msg.get("tool", "unknown")
            error = tool_msg.get("error")
            print(f"{i}. TOOL CALL [{event_id}] - {tool_name}")
            if error:
                print(f"   ❌ Error: {error}")
            else:
                print(f"   ✅ Success")
    
    # Show final prompt for next turn
    print(f"\n{'='*60}")
    print("🔄 Final Prompt for Next LLM Turn:")
    print(f"{'='*60}")
    next_prompt = await build_prompt_from_session(session)
    pprint.pp(next_prompt, width=80)
    
    # Show session statistics
    print(f"\n{'='*60}")
    print("📈 Session Statistics:")
    print(f"{'='*60}")
    print(f"   Session ID: {session.id}")
    print(f"   Total events: {len(session.events)}")
    print(f"   Total tokens: {session.total_tokens}")
    print(f"   Estimated cost: ${session.total_cost:.6f}")
    
    # Event breakdown
    event_types = {}
    for event in session.events:
        event_type = f"{event.source.value}:{event.type.value}"
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    print(f"   Event breakdown:")
    for event_type, count in event_types.items():
        print(f"     {event_type}: {count}")
    
    print(f"\n{'='*60}")
    print("🎯 Key Takeaways:")
    print("   • LLM retries handled at application level")
    print("   • Tool execution reliability handled by chuk_tool_processor")
    print("   • Complete audit trail in session events")
    print("   • Separation of concerns: LLM cooperation vs tool reliability")
    print("   • Session tracks all attempts for debugging and analytics")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())