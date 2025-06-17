# sample_tools_debug.py
"""
Sample tools with debugging
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, Any

print("📦 sample_tools_debug: Starting import...")

from chuk_tool_processor.registry import register_tool

print("📦 sample_tools_debug: Imported register_tool")

@register_tool(name="debug_calculator", description="Debug calculator tool")
class DebugCalculatorTool:
    """Calculator tool with debug output."""
    
    async def execute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        print(f"🧮 DEBUG Calculator executing: {a} {operation} {b}")
        
        if operation == "multiply":
            result = a * b
        elif operation == "add":
            result = a + b
        else:
            result = 0
            
        return {
            "operation": operation,
            "a": a,
            "b": b,
            "result": result
        }

print("✅ sample_tools_debug: debug_calculator registered")

@register_tool(name="debug_weather", description="Debug weather tool")
class DebugWeatherTool:
    """Weather tool with debug output."""
    
    async def execute(self, location: str) -> Dict[str, Any]:
        print(f"🌤️ DEBUG Weather executing for: {location}")
        
        return {
            "location": location,
            "temperature": 20.0,
            "condition": "Sunny",
            "description": f"Mock weather for {location}"
        }

print("✅ sample_tools_debug: debug_weather registered")
print("📦 sample_tools_debug: Import complete!")
