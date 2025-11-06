"""
Quick Start Example - Using Shared Ollama Service

This example demonstrates how to use the shared Ollama service
from any project with minimal setup.
"""

import sys
from pathlib import Path

# Add parent directory to path to import shared client
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared_ollama_client import (
    GenerateOptions,
    Model,
    OllamaConfig,
    SharedOllamaClient,
)
from utils import check_service_health, ensure_service_running, get_ollama_base_url


def example_basic_usage():
    """Basic usage example."""
    print("Example 1: Basic Usage")
    print("-" * 40)

    # Ensure service is running
    ensure_service_running()

    # Create client with default config
    client = SharedOllamaClient()

    # List available models
    models = client.list_models()
    print(f"Available models: {[m['name'] for m in models]}")

    # Generate text
    response = client.generate("Explain quantum computing in one sentence.")
    print(f"\nResponse: {response.text}")


def example_custom_config():
    """Example with custom configuration."""
    print("\nExample 2: Custom Configuration")
    print("-" * 40)

    # Create custom config with qwen2.5:14b
    config = OllamaConfig(
        base_url=get_ollama_base_url(), default_model=Model.QWEN25_14B, timeout=120
    )

    client = SharedOllamaClient(config)
    response = client.generate("What is machine learning?")
    print(f"Response: {response.text[:100]}...")


def example_using_qwen7b():
    """Example using qwen2.5:7b model."""
    print("\nExample 2b: Using qwen2.5:7b Model")
    print("-" * 40)

    # Use qwen2.5:7b for efficient text generation
    config = OllamaConfig(
        base_url=get_ollama_base_url(), default_model=Model.QWEN25_7B, timeout=120
    )

    client = SharedOllamaClient(config)
    response = client.generate("Explain Python in one sentence.")
    print(f"Response: {response.text}")


def example_using_granite():
    """Example using Granite 4.0 H Tiny model."""
    print("\nExample 2c: Using Granite 4.0 H Tiny Model")
    print("-" * 40)

    # Use Granite 4.0 for RAG and function calling
    config = OllamaConfig(
        base_url=get_ollama_base_url(),
        default_model=Model.GRANITE_4_H_TINY,
        timeout=120,
    )

    client = SharedOllamaClient(config)
    response = client.generate("Explain RAG in one sentence.")
    print(f"Response: {response.text}")


def example_chat_format():
    """Example using chat format."""
    print("\nExample 3: Chat Format")
    print("-" * 40)

    client = SharedOllamaClient()

    messages = [
        {"role": "user", "content": "Hello! Can you help me?"},
    ]

    response = client.chat(messages)
    print(f"Assistant: {response['message']['content']}")


def example_with_options():
    """Example with generation options."""
    print("\nExample 4: Generation Options")
    print("-" * 40)

    client = SharedOllamaClient()

    options = GenerateOptions(temperature=0.7, top_p=0.9, max_tokens=100)

    response = client.generate("Write a haiku about programming.", options=options)
    print(f"Response: {response.text}")


def example_error_handling():
    """Example with error handling."""
    print("\nExample 5: Error Handling")
    print("-" * 40)

    is_healthy, error = check_service_health()
    if not is_healthy:
        print(f"Service is not available: {error}")
        print("Start the service with: ./scripts/setup_launchd.sh")
        return

    try:
        client = SharedOllamaClient()
        response = client.generate("Test prompt")
        print(f"Success: {response.text[:50]}...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Shared Ollama Service - Quick Start Examples")
    print("=" * 50)

    try:
        example_basic_usage()
        example_custom_config()
        example_using_qwen7b()
        example_using_granite()
        example_chat_format()
        example_with_options()
        example_error_handling()

        print("\n" + "=" * 50)
        print("✓ All examples completed!")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
