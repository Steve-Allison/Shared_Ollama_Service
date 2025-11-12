#!/usr/bin/env python3
"""Test script for streaming REST API endpoints."""

import requests
import json
import sys


def test_generate_streaming():
    """Test /api/v1/generate with streaming."""
    print("Testing /api/v1/generate with streaming...")
    print("-" * 50)

    url = "http://localhost:8000/api/v1/generate"
    payload = {
        "prompt": "Write a haiku about Python programming.",
        "model": "qwen2.5:7b",
        "stream": True,
    }

    print(f"Request: POST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("\nStreaming response:")

    full_text = ""
    chunk_count = 0

    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode('utf-8')

                # Parse SSE format (data: {...})
                if line_str.startswith('data: '):
                    json_str = line_str[6:]  # Remove "data: " prefix
                    try:
                        chunk = json.loads(json_str)
                        chunk_count += 1

                        text = chunk.get('chunk', '')
                        done = chunk.get('done', False)

                        if text:
                            print(text, end='', flush=True)
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
                            print(f"  Total Chunks: {chunk_count}")
                            print("=" * 50)

                    except json.JSONDecodeError as e:
                        print(f"\nError parsing chunk: {e}")
                        print(f"Raw line: {line_str}")

        print(f"\n✓ Streaming generate test passed")
        print(f"Total text length: {len(full_text)} characters\n")
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chat_streaming():
    """Test /api/v1/chat with streaming."""
    print("\nTesting /api/v1/chat with streaming...")
    print("-" * 50)

    url = "http://localhost:8000/api/v1/chat"
    payload = {
        "messages": [
            {"role": "user", "content": "Tell me a very short joke about AI."}
        ],
        "model": "qwen2.5:7b",
        "stream": True,
    }

    print(f"Request: POST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("\nStreaming response:")

    full_text = ""
    chunk_count = 0

    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode('utf-8')

                # Parse SSE format (data: {...})
                if line_str.startswith('data: '):
                    json_str = line_str[6:]  # Remove "data: " prefix
                    try:
                        chunk = json.loads(json_str)
                        chunk_count += 1

                        text = chunk.get('chunk', '')
                        done = chunk.get('done', False)

                        if text:
                            print(text, end='', flush=True)
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
                            print(f"  Total Chunks: {chunk_count}")
                            print("=" * 50)

                    except json.JSONDecodeError as e:
                        print(f"\nError parsing chunk: {e}")
                        print(f"Raw line: {line_str}")

        print(f"\n✓ Streaming chat test passed")
        print(f"Total text length: {len(full_text)} characters\n")
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all streaming API tests."""
    print("\n" + "=" * 50)
    print("STREAMING REST API TESTS")
    print("=" * 50 + "\n")

    result1 = test_generate_streaming()
    result2 = test_chat_streaming()

    if result1 and result2:
        print("\n" + "=" * 50)
        print("✓ ALL STREAMING API TESTS PASSED")
        print("=" * 50 + "\n")
        return 0
    else:
        print("\n" + "=" * 50)
        print("✗ SOME TESTS FAILED")
        print("=" * 50 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
