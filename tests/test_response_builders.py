from __future__ import annotations

from shared_ollama.api.models import RequestContext
from shared_ollama.api.response_builders import (
    build_openai_chat_response,
    build_openai_stream_chunk,
)


def _ctx() -> RequestContext:
    return RequestContext(request_id="req-123", client_ip="127.0.0.1", user_agent="pytest")


def test_build_openai_chat_response_preserves_tool_calls() -> None:
    result = {
        "message": {
            "role": "assistant",
            "content": "Here is the answer",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"foo":"bar"}'},
                }
            ],
            "refusal": None,
            "annotations": [],
        },
        "model": "qwen",
        "prompt_eval_count": 10,
        "eval_count": 5,
        "finish_reason": "stop",
    }

    response = build_openai_chat_response(result, _ctx())

    choice = response["choices"][0]
    assert choice["message"]["tool_calls"] == result["message"]["tool_calls"]
    assert choice["message"]["refusal"] is None
    assert choice["message"]["annotations"] == []
    assert response["usage"]["total_tokens"] == 15


def test_build_openai_stream_chunk_formats_intermediate_and_final_chunks() -> None:
    chunk = {
        "chunk": "Hello",
        "role": "assistant",
        "done": False,
        "model": "qwen",
    }
    chunk_payload = build_openai_stream_chunk(chunk, _ctx(), created_ts=1700000000, include_role=True)

    assert chunk_payload["object"] == "chat.completion.chunk"
    delta = chunk_payload["choices"][0]["delta"]
    assert delta["role"] == "assistant"
    assert delta["content"] == "Hello"
    assert chunk_payload["choices"][0]["finish_reason"] is None

    final_chunk = {
        "chunk": "",
        "role": "assistant",
        "done": True,
        "model": "qwen",
        "prompt_eval_count": 10,
        "generation_eval_count": 2,
    }
    final_payload = build_openai_stream_chunk(
        final_chunk, _ctx(), created_ts=1700000000, include_role=False
    )

    assert final_payload["choices"][0]["delta"] == {}
    assert final_payload["choices"][0]["finish_reason"] == "stop"
    assert final_payload["usage"]["prompt_tokens"] == 10
    assert final_payload["usage"]["completion_tokens"] == 2
    assert final_payload["usage"]["total_tokens"] == 12

