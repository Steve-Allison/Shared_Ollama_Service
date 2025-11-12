"""
Quick test to verify request body parsing works correctly with slowapi.
Run with: python test_body_parsing.py
"""

import httpx
import asyncio


async def test_generate_endpoint():
    """Test that the /generate endpoint accepts JSON body correctly."""
    url = "http://localhost:8000/api/v1/generate"

    # Test data - minimal valid request
    payload = {
        "prompt": "Say hello!",
        "model": "granite4:tiny-h"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=payload)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

            if response.status_code == 200:
                print("✅ SUCCESS: Body parsed correctly!")
                return True
            elif response.status_code == 422:
                print("❌ FAILED: Validation error (likely treating body as query param)")
                print(f"Error details: {response.json()}")
                return False
            else:
                print(f"⚠️  Unexpected status: {response.status_code}")
                return False

        except httpx.ConnectError:
            print("❌ Cannot connect to server. Is it running on port 8000?")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


async def test_chat_endpoint():
    """Test that the /chat endpoint accepts JSON body correctly."""
    url = "http://localhost:8000/api/v1/chat"

    # Test data - minimal valid request
    payload = {
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "model": "granite4:tiny-h"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=payload)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

            if response.status_code == 200:
                print("✅ SUCCESS: Body parsed correctly!")
                return True
            elif response.status_code == 422:
                print("❌ FAILED: Validation error (likely treating body as query param)")
                print(f"Error details: {response.json()}")
                return False
            else:
                print(f"⚠️  Unexpected status: {response.status_code}")
                return False

        except httpx.ConnectError:
            print("❌ Cannot connect to server. Is it running on port 8000?")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


async def main():
    print("=" * 60)
    print("Testing FastAPI Body Parsing with slowapi")
    print("=" * 60)

    print("\n1. Testing /api/v1/generate endpoint...")
    print("-" * 60)
    gen_result = await test_generate_endpoint()

    print("\n2. Testing /api/v1/chat endpoint...")
    print("-" * 60)
    chat_result = await test_chat_endpoint()

    print("\n" + "=" * 60)
    if gen_result and chat_result:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
