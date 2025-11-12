#!/usr/bin/env python3
"""Test script for streaming methods in AsyncSharedOllamaClient."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from shared_ollama.client.async_client import AsyncOllamaConfig, AsyncSharedOllamaClient


async def test_generate_stream():
    """Test generate_stream method."""
    print("Testing generate_stream()...")
    print("-" * 50)

    config = AsyncOllamaConfig(
        base_url="http://localhost:11434",
        default_model="qwen2.5:7b",
    )

    async with AsyncSharedOllamaClient(config) as client:
        prompt = "Write a haiku about coding."

        print(f"Prompt: {prompt}")
        print("\nStreaming response:")

        full_text = ""
        chunk_count = 0

        async for chunk in client.generate_stream(prompt=prompt):
            chunk_count += 1
            text = chunk.get("chunk", "")
            done = chunk.get("done", False)

            if text:
                print(text, end="", flush=True)
                full_text += text

            if done:
                print("\n\n" + "=" * 50)
                print("Final chunk metrics:")
                print(f"  Request ID: {chunk.get('request_id')}")
                print(f"  Model: {chunk.get('model')}")
                print(f"  Latency: {chunk.get('latency_ms')}ms")
                print(f"  Model Load: {chunk.get('model_load_ms')}ms")
                print(f"  Warm Start: {chunk.get('model_warm_start')}")
                print(f"  Total Duration: {chunk.get('total_duration_ms')}ms")
                print(f"  Prompt Tokens: {chunk.get('prompt_eval_count')}")
                print(f"  Generated Tokens: {chunk.get('generation_eval_count')}")
                print(f"  Total Chunks: {chunk_count}")
                print("=" * 50)

        print(f"\n✓ generate_stream() completed successfully")
        print(f"Total text length: {len(full_text)} characters\n")

    return True


async def test_chat_stream():
    """Test chat_stream method."""
    print("\nTesting chat_stream()...")
    print("-" * 50)

    config = AsyncOllamaConfig(
        base_url="http://localhost:11434",
        default_model="qwen2.5:7b",
    )

    async with AsyncSharedOllamaClient(config) as client:
        messages = [
            {"role": "user", "content": "Tell me a short joke about programming."}
        ]

        print(f"Messages: {messages}")
        print("\nStreaming response:")

        full_text = ""
        chunk_count = 0

        async for chunk in client.chat_stream(messages=messages):
            chunk_count += 1
            text = chunk.get("chunk", "")
            done = chunk.get("done", False)

            if text:
                print(text, end="", flush=True)
                full_text += text

            if done:
                print("\n\n" + "=" * 50)
                print("Final chunk metrics:")
                print(f"  Request ID: {chunk.get('request_id')}")
                print(f"  Model: {chunk.get('model')}")
                print(f"  Role: {chunk.get('role')}")
                print(f"  Latency: {chunk.get('latency_ms')}ms")
                print(f"  Model Load: {chunk.get('model_load_ms')}ms")
                print(f"  Warm Start: {chunk.get('model_warm_start')}")
                print(f"  Total Duration: {chunk.get('total_duration_ms')}ms")
                print(f"  Prompt Tokens: {chunk.get('prompt_eval_count')}")
                print(f"  Generated Tokens: {chunk.get('generation_eval_count')}")
                print(f"  Total Chunks: {chunk_count}")
                print("=" * 50)

        print(f"\n✓ chat_stream() completed successfully")
        print(f"Total text length: {len(full_text)} characters\n")

    return True


async def main():
    """Run all streaming tests."""
    print("\n" + "=" * 50)
    print("STREAMING CLIENT TESTS")
    print("=" * 50 + "\n")

    try:
        # Test generate streaming
        result1 = await test_generate_stream()

        # Test chat streaming
        result2 = await test_chat_stream()

        if result1 and result2:
            print("\n" + "=" * 50)
            print("✓ ALL STREAMING TESTS PASSED")
            print("=" * 50 + "\n")
            return 0
        else:
            print("\n" + "=" * 50)
            print("✗ SOME TESTS FAILED")
            print("=" * 50 + "\n")
            return 1

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
