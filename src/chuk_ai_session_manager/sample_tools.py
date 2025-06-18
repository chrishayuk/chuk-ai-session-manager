# chuk_ai_session_manager/sample_tools.py
"""
Sample tools for chuk session manager demos - corrected version following registry example
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, Any

from chuk_tool_processor.registry import register_tool


@register_tool(name="calculator", namespace="default", description="Perform basic arithmetic operations")
class CalculatorTool:
    """Calculator tool for basic arithmetic."""
    
    async def execute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        """
        Perform a basic arithmetic operation.
        
        Args:
            operation: One of "add", "subtract", "multiply", "divide"
            a: First operand
            b: Second operand
            
        Returns:
            Dictionary with the result
        """
        print(f"🧮 Calculator executing: {a} {operation} {b}")
        
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


@register_tool(name="weather", namespace="default", description="Get current weather information for a location")
class WeatherTool:
    """Weather tool that returns mock weather data."""
    
    async def execute(self, location: str) -> Dict[str, Any]:
        """
        Get weather information for a specific location.
        
        Args:
            location: The city or location to get weather for
            
        Returns:
            Dictionary with weather information
        """
        print(f"🌤️ Weather tool executing for: {location}")
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Mock realistic weather based on location
        base_temp = 15  # Default moderate temperature
        if any(city in location.lower() for city in ["miami", "phoenix", "dubai", "singapore"]):
            base_temp = 28
        elif any(city in location.lower() for city in ["moscow", "montreal", "oslo", "anchorage"]):
            base_temp = -5
        elif any(city in location.lower() for city in ["london", "seattle", "vancouver"]):
            base_temp = 12
        elif any(city in location.lower() for city in ["tokyo", "new york", "paris", "berlin"]):
            base_temp = 18
        
        # Add some randomness
        temperature = base_temp + random.randint(-8, 12)
        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Heavy Rain", "Snow", "Thunderstorm", "Foggy"]
        condition = random.choice(conditions)
        humidity = random.randint(35, 85)
        wind_speed = random.uniform(2.0, 25.0)
        feels_like = temperature + random.randint(-3, 3)
        
        # Adjust conditions based on temperature
        if temperature < 0:
            condition = random.choice(["Snow", "Cloudy", "Partly Cloudy"])
        elif temperature > 30:
            condition = random.choice(["Sunny", "Partly Cloudy", "Hot"])
        
        description = f"Current weather in {location} is {condition.lower()} with temperature {temperature}°C"
        
        return {
            "location": location,
            "temperature": float(temperature),
            "condition": condition,
            "humidity": humidity,
            "wind_speed": round(wind_speed, 1),
            "description": description,
            "feels_like": float(feels_like),
            "timestamp": datetime.now().isoformat()
        }


@register_tool(name="search", namespace="default", description="Search for information on the internet")  
class SearchTool:
    """Search tool that returns mock search results."""
    
    async def execute(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Search for information on the internet.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        print(f"🔍 Search tool executing for: {query}")
        await asyncio.sleep(0.2)  # Simulate API delay
        
        results = []
        query_lower = query.lower()
        
        # Generate contextually relevant mock results based on query
        if "climate" in query_lower or "environment" in query_lower:
            result_templates = [
                {
                    "title": "Climate Change Adaptation Strategies - IPCC Report",
                    "url": "https://www.ipcc.ch/adaptation-strategies",
                    "snippet": "Comprehensive guide to climate change adaptation strategies for communities, businesses, and governments. Includes resilience planning and risk assessment."
                },
                {
                    "title": "Environmental Adaptation Solutions | Climate.gov", 
                    "url": "https://www.climate.gov/adaptation-solutions",
                    "snippet": "Evidence-based climate adaptation solutions including infrastructure improvements, ecosystem restoration, and community planning approaches."
                },
                {
                    "title": "Building Climate Resilience: A Practical Guide",
                    "url": "https://www.resilience.org/climate-guide", 
                    "snippet": "Practical steps for building climate resilience in your community. Covers early warning systems, green infrastructure, and adaptation planning."
                }
            ]
        elif "weather" in query_lower:
            result_templates = [
                {
                    "title": "Weather Forecast and Current Conditions",
                    "url": "https://weather.com/forecast",
                    "snippet": "Get accurate weather forecasts, current conditions, and severe weather alerts for your location."
                },
                {
                    "title": "Climate and Weather Patterns Explained",
                    "url": "https://www.weatherpatterns.org",
                    "snippet": "Understanding weather patterns, climate systems, and meteorological phenomena that affect daily weather."
                }
            ]
        else:
            # Generic results for other queries
            result_templates = [
                {
                    "title": f"Everything You Need to Know About {query.title()}",
                    "url": f"https://encyclopedia.com/{query.lower().replace(' ', '-')}",
                    "snippet": f"Comprehensive information and resources about {query}. Expert insights, latest research, and practical applications."
                },
                {
                    "title": f"{query.title()} - Latest News and Updates",
                    "url": f"https://news.example.com/{query.lower().replace(' ', '-')}",
                    "snippet": f"Stay up to date with the latest news, trends, and developments related to {query}."
                },
                {
                    "title": f"Guide to {query.title()} - Best Practices",
                    "url": f"https://guides.com/{query.lower().replace(' ', '-')}",
                    "snippet": f"Expert guide covering best practices, tips, and strategies for {query}. Includes real-world examples and case studies."
                }
            ]
        
        # Select results up to max_results
        selected_results = result_templates[:max_results]
        
        return {
            "query": query,
            "results_count": len(selected_results),
            "results": selected_results,
            "timestamp": datetime.now().isoformat()
        }


print("✅ sample_tools.py: 3 tools defined with @register_tool decorator (corrected version)")