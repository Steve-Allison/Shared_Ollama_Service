"""
Simulate external caller behavior to diagnose "no response" issue.
"""

import requests
import json


def test_as_external_caller():
    """Test API as an external caller would."""
    base_url = "http://localhost:8000/api/v1"

    print("=" * 70)
    print("TESTING AS EXTERNAL CALLER")
    print("=" * 70)

    # Test 1: Generate endpoint
    print("\n1. Testing /generate endpoint...")
    print("-" * 70)

    try:
        response = requests.post(
            f"{base_url}/generate",
            headers={
                "Content-Type": "application/json",
                "X-Project-Name": "Knowledge_Machine"
            },
            json={
                "prompt": "Say hello",
                "model": "granite4:tiny-h"
            },
            timeout=30
        )

        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Raw Response: {response.text[:200]}...")

        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ SUCCESS")
            print(f"   Response has 'text' field: {'text' in data}")
            print(f"   Text content: {data.get('text', '')[:100]}...")
            print(f"   Model: {data.get('model')}")
            print(f"   Request ID: {data.get('request_id')}")
        else:
            print(f"\n❌ FAILED: {response.status_code}")
            print(f"   Error: {response.text}")

    except Exception as e:
        print(f"❌ EXCEPTION: {e}")

    # Test 2: Chat endpoint
    print("\n2. Testing /chat endpoint...")
    print("-" * 70)

    try:
        response = requests.post(
            f"{base_url}/chat",
            headers={
                "Content-Type": "application/json",
                "X-Project-Name": "Knowledge_Machine"
            },
            json={
                "messages": [
                    {"role": "user", "content": "Say hello"}
                ],
                "model": "granite4:tiny-h"
            },
            timeout=30
        )

        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Raw Response: {response.text[:200]}...")

        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ SUCCESS")
            print(f"   Response has 'message' field: {'message' in data}")
            if 'message' in data:
                msg = data['message']
                print(f"   Message role: {msg.get('role')}")
                print(f"   Message content: {msg.get('content', '')[:100]}...")
            print(f"   Model: {data.get('model')}")
            print(f"   Request ID: {data.get('request_id')}")
        else:
            print(f"\n❌ FAILED: {response.status_code}")
            print(f"   Error: {response.text}")

    except Exception as e:
        print(f"❌ EXCEPTION: {e}")

    # Test 3: Check response format compatibility
    print("\n3. Checking response format...")
    print("-" * 70)

    # Generate response format
    gen_response = requests.post(
        f"{base_url}/generate",
        json={"prompt": "Hi", "model": "granite4:tiny-h"},
        timeout=30
    ).json()

    print("Generate response keys:", list(gen_response.keys()))
    print(f"  - text: {type(gen_response.get('text'))}")
    print(f"  - model: {type(gen_response.get('model'))}")
    print(f"  - request_id: {type(gen_response.get('request_id'))}")

    # Chat response format
    chat_response = requests.post(
        f"{base_url}/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "model": "granite4:tiny-h"},
        timeout=30
    ).json()

    print("\nChat response keys:", list(chat_response.keys()))
    print(f"  - message: {type(chat_response.get('message'))}")
    print(f"  - message.content: {type(chat_response.get('message', {}).get('content'))}")
    print(f"  - model: {type(chat_response.get('model'))}")
    print(f"  - request_id: {type(chat_response.get('request_id'))}")

    print("\n" + "=" * 70)
    print("EXTERNAL CALLER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_as_external_caller()
