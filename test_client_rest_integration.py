"""
Test the client code integration with the REST API.
This verifies that OllamaClient and AsyncOllamaClient work correctly via REST.
"""

import asyncio

from shared_ollama import AsyncSharedOllamaClient, SharedOllamaClient
from shared_ollama.client import AsyncOllamaConfig, OllamaConfig


def test_sync_client_rest():
    """Test synchronous client via REST API."""
    print("\n" + "=" * 60)
    print("Testing Synchronous Client (REST API)")
    print("=" * 60)

    # Create client configured for REST API
    config = OllamaConfig()
    client = SharedOllamaClient(config=config)

    # Test 1: Generate
    print("\n1. Testing generate()...")
    try:
        result = client.generate(
            prompt="Say hello in one sentence",
            model="granite4:tiny-h",
        )
        print(f"✅ Generate succeeded")
        print(f"   Response: {result.text[:100]}...")
        print(f"   Model: {result.model}")
    except Exception as e:
        print(f"❌ Generate failed: {e}")
        return False

    # Test 2: Chat
    print("\n2. Testing chat()...")
    try:
        result = client.chat(
            messages=[{"role": "user", "content": "Say hi"}],
            model="granite4:tiny-h",
        )
        print(f"✅ Chat succeeded")
        print(f"   Response: {result.get('message', {}).get('content', '')[:100]}...")
        print(f"   Model: {result.get('model')}")
    except Exception as e:
        print(f"❌ Chat failed: {e}")
        return False

    # Test 3: List models
    print("\n3. Testing list_models()...")
    try:
        models = client.list_models()
        print(f"✅ List models succeeded")
        print(f"   Found {len(models)} models")
        if models:
            print(f"   First model: {models[0].get('name')}")
    except Exception as e:
        print(f"❌ List models failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ ALL SYNC CLIENT TESTS PASSED")
    print("=" * 60)
    return True


async def test_async_client_rest():
    """Test asynchronous client via REST API."""
    print("\n" + "=" * 60)
    print("Testing Asynchronous Client (REST API)")
    print("=" * 60)

    # Create async client configured for REST API
    config = AsyncOllamaConfig()
    client = AsyncSharedOllamaClient(config=config)

    # Test 1: Generate
    print("\n1. Testing generate()...")
    try:
        result = await client.generate(
            prompt="Say hello in one sentence",
            model="granite4:tiny-h",
        )
        print(f"✅ Generate succeeded")
        print(f"   Response: {result.text[:100]}...")
        print(f"   Model: {result.model}")
    except Exception as e:
        print(f"❌ Generate failed: {e}")
        return False

    # Test 2: Chat
    print("\n2. Testing chat()...")
    try:
        result = await client.chat(
            messages=[{"role": "user", "content": "Say hi"}],
            model="granite4:tiny-h",
        )
        print(f"✅ Chat succeeded")
        print(f"   Response: {result.get('message', {}).get('content', '')[:100]}...")
        print(f"   Model: {result.get('model')}")
    except Exception as e:
        print(f"❌ Chat failed: {e}")
        await client.close()
        return False

    # Test 3: List models
    print("\n3. Testing list_models()...")
    try:
        models = await client.list_models()
        print(f"✅ List models succeeded")
        print(f"   Found {len(models)} models")
        if models:
            print(f"   First model: {models[0].get('name')}")
    except Exception as e:
        print(f"❌ List models failed: {e}")
        await client.close()
        return False

    await client.close()

    print("\n" + "=" * 60)
    print("✅ ALL ASYNC CLIENT TESTS PASSED")
    print("=" * 60)
    return True


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("REST API CLIENT INTEGRATION TESTS")
    print("=" * 60)

    # Test sync client
    sync_result = test_sync_client_rest()

    # Test async client
    async_result = asyncio.run(test_async_client_rest())

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Sync Client:  {'✅ PASS' if sync_result else '❌ FAIL'}")
    print(f"Async Client: {'✅ PASS' if async_result else '❌ FAIL'}")

    if sync_result and async_result:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("=" * 60)
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
