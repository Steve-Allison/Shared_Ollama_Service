"""Complete POML + Shared Ollama Service Example: Weather Assistant with Tool Calling.

This example demonstrates:
- POML template with tool definitions
- Tool calling workflow
- Conversation history management
- JSON schema output validation
- Integration with Shared Ollama Service

Requirements:
    pip install poml requests

Usage:
    python weather_assistant.py
"""

import json
from collections.abc import Callable
from typing import Any, TypedDict, cast

import poml
import requests

PomlRender = Callable[..., dict[str, Any]]
POML_RENDER: PomlRender = cast(Any, poml).poml

# API endpoint
API_BASE = "http://0.0.0.0:8000/api/v1"


class WeatherData(TypedDict):
    temperature: float
    condition: str
    humidity: int


def get_weather(location: str, unit: str = "celsius") -> WeatherData:
    """Simulate weather API call.

    In production, this would call a real weather API.
    """
    # Simulated weather data
    weather_data: dict[str, WeatherData] = {
        "Paris": {"temperature": 22.0, "condition": "sunny", "humidity": 65},
        "London": {"temperature": 15.0, "condition": "rainy", "humidity": 80},
        "New York": {"temperature": 25.0, "condition": "partly cloudy", "humidity": 70},
    }

    data = weather_data.get(
        location, {"temperature": 20.0, "condition": "unknown", "humidity": 50}
    )

    # Convert to Fahrenheit if requested
    if unit == "fahrenheit":
        temp = data["temperature"]
        data["temperature"] = round(temp * 9 / 5 + 32, 2)

    return data


def calculate(expression: str) -> dict:
    """Safely evaluate mathematical expressions.

    In production, use a proper math parser library.
    """
    try:
        # WARNING: eval() is unsafe for untrusted input!
        # Use a proper expression parser in production (e.g., sympy, numexpr)
        result = eval(expression, {"__builtins__": {}}, {})
        safe_result: float | int | str
        safe_result = result if isinstance(result, int | float) else str(result)
        return {"result": safe_result, "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}


def main():
    """Run the weather assistant example."""
    print("=" * 80)
    print("POML Weather Assistant - Shared Ollama Service Integration")
    print("=" * 80)
    print()

    # Initial user question
    user_question = "What's the weather in Paris? Also, what's 15 + 27?"
    print(f"User: {user_question}")
    print()

    # Context for POML template
    context: dict[str, object] = {
        "user_question": user_question,
        "tool_request": None,
        "tool_response": None,
    }

    # Load POML template and generate request
    print("📝 Generating request from POML template...")
    params = POML_RENDER("chat_with_tools.poml", context=context, format="openai_chat")
    print(f"✓ Tools defined: {len(params.get('tools', []))}")
    print()

    # Send initial request to Shared Ollama Service
    print("🚀 Sending request to Shared Ollama Service...")
    response = requests.post(
        f"{API_BASE}/chat",
        json=params
    ).json()

    print(f"✓ Response received (latency: {response['latency_ms']:.1f}ms)")
    print()

    # Check if model wants to call tools
    message = response["message"]

    if message.get("tool_calls"):
        print("🔧 Model is calling tools:")
        print()

        # Process each tool call
        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])

            print(f"   Tool: {func_name}")
            print(f"   Arguments: {json.dumps(func_args, indent=2)}")

            # Execute the appropriate function
            if func_name == "get_weather":
                result = get_weather(**func_args)
                print(f"   Result: {json.dumps(result, indent=2)}")
            elif func_name == "calculate":
                result = calculate(**func_args)
                print(f"   Result: {json.dumps(result, indent=2)}")
            else:
                result = {"error": f"Unknown function: {func_name}"}
                print("   Error: Unknown function")

            print()

            # Update context with tool response
            context["tool_response"] = {
                "id": tool_call["id"],
                "name": func_name,
                "result": result,
            }

        # Generate new request with tool results
        print("📝 Generating follow-up request with tool results...")
        params = POML_RENDER("chat_with_tools.poml", context=context, format="openai_chat")

        # Send follow-up request
        print("🚀 Sending follow-up request...")
        final_response = requests.post(
            f"{API_BASE}/chat",
            json=params
        ).json()

        print(f"✓ Response received (latency: {final_response['latency_ms']:.1f}ms)")
        print()

        # Display final answer
        print("🤖 Assistant's Final Answer:")
        print("-" * 80)
        print(final_response["message"]["content"])
        print("-" * 80)
    else:
        # No tool calls, direct response
        print("🤖 Assistant's Response:")
        print("-" * 80)
        print(message["content"])
        print("-" * 80)

    print()
    print("✓ Example completed successfully!")
    print()

    # Show metrics
    print("📊 Metrics:")
    print(f"   Model: {response['model']}")
    print(f"   Request ID: {response['request_id']}")
    print(f"   Total tokens: {response.get('prompt_eval_count', 0) + response.get('generation_eval_count', 0)}")
    print()


if __name__ == "__main__":
    main()
