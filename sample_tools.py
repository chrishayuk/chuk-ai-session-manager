# sample_tools.py
"""
Sample tools for chuk session manager demos.
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, Any

from chuk_tool_processor.registry import register_tool


@register_tool(name="calculator", description="Perform basic arithmetic operations")
class CalculatorTool:
    """Calculator tool for basic arithmetic."""
    
    async def execute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        """Perform a basic arithmetic operation."""
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        return {
            "operation": operation,
            "a": a,
            "b": b,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }


@register_tool(name="weather", description="Get current weather information for a location")
class WeatherTool:
    """Weather tool that returns mock weather data."""
    
    async def execute(self, location: str) -> Dict[str, Any]:
        """Get weather information for a specific location."""
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Mock weather data
        base_temp = 18
        if "new york" in location.lower():
            base_temp = 15
        elif "miami" in location.lower():
            base_temp = 28
            
        temperature = base_temp + random.randint(-5, 8)
        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain"]
        condition = random.choice(conditions)
        
        return {
            "location": location,
            "temperature": float(temperature),
            "condition": condition,
            "humidity": random.randint(35, 85),
            "wind_speed": round(random.uniform(2.0, 15.0), 1),
            "description": f"Current weather in {location} is {condition.lower()} with temperature {temperature}°C",
            "timestamp": datetime.now().isoformat()
        }


@register_tool(name="search", description="Search for information on the internet")
class SearchTool:
    """Search tool that returns mock search results."""
    
    async def execute(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Search for information on the internet."""
        await asyncio.sleep(0.2)  # Simulate API delay
        
        # Mock search results
        if "climate" in query.lower():
            results = [
                {
                    "title": "Climate Change Adaptation Strategies - IPCC Report",
                    "url": "https://www.ipcc.ch/adaptation-strategies",
                    "snippet": "Comprehensive guide to climate change adaptation strategies for communities and businesses."
                },
                {
                    "title": "Building Climate Resilience: A Practical Guide", 
                    "url": "https://www.resilience.org/climate-guide",
                    "snippet": "Practical steps for building climate resilience including early warning systems."
                }
            ]
        else:
            results = [
                {
                    "title": f"Information about {query}",
                    "url": f"https://example.com/{query.lower().replace(' ', '-')}",
                    "snippet": f"Comprehensive information and resources about {query}."
                }
            ]
        
        return {
            "query": query,
            "results_count": len(results),
            "results": results[:max_results],
            "timestamp": datetime.now().isoformat()
        }


print("✅ sample_tools.py: 3 tools defined with @register_tool decorator")
