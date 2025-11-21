"""
Comprehensive behavioral tests for API mappers.

Tests focus on real mapping behavior, format resolution, edge cases, and error handling.
No mocks - tests use real Pydantic models and domain entities.
"""

import pytest

from shared_ollama.api.mappers import (
    api_to_domain_chat_request,
    api_to_domain_generation_request,
    api_to_domain_tool,
    api_to_domain_tool_call,
    api_to_domain_vlm_request,
    api_to_domain_vlm_request_openai,
    domain_to_api_model_info,
    domain_to_api_tool_call,
)
from shared_ollama.api.models import (
    ChatMessage as APIChatMessage,
    ChatMessageOpenAI as APIMessageOpenAI,
    ChatRequest as APIChatRequest,
    GenerateRequest as APIGenerateRequest,
    ImageContentPart,
    ModelInfo as APIModelInfo,
    ResponseFormat,
    TextContentPart,
    Tool as APITool,
    ToolCall as APIToolCall,
    ToolCallFunction as APIToolCallFunction,
    ToolFunction as APIToolFunction,
    VLMMessage as APIMessage,
    VLMRequest as APIVLMRequest,
    VLMRequestOpenAI,
)
from shared_ollama.domain.entities import (
    ChatRequest,
    GenerationRequest,
    ModelInfo,
    Tool,
    ToolCall,
    ToolCallFunction,
    VLMRequest,
)


class TestToolMappers:
    """Behavioral tests for tool calling mappers."""

    def test_api_to_domain_tool_converts_correctly(self):
        """Test that api_to_domain_tool converts API tool to domain tool."""
        api_tool = APITool(
            type="function",
            function=APIToolFunction(
                name="get_weather",
                description="Get weather for location",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            ),
        )

        domain_tool = api_to_domain_tool(api_tool)

        assert isinstance(domain_tool, Tool)
        assert domain_tool.type == "function"
        assert domain_tool.function.name == "get_weather"
        assert domain_tool.function.description == "Get weather for location"
        assert domain_tool.function.parameters is not None

    def test_api_to_domain_tool_call_converts_correctly(self):
        """Test that api_to_domain_tool_call converts API tool call to domain."""
        api_tool_call = APIToolCall(
            id="call_123",
            type="function",
            function=APIToolCallFunction(
                name="get_weather",
                arguments='{"location": "San Francisco"}',
            ),
        )

        domain_tool_call = api_to_domain_tool_call(api_tool_call)

        assert isinstance(domain_tool_call, ToolCall)
        assert domain_tool_call.id == "call_123"
        assert domain_tool_call.type == "function"
        assert domain_tool_call.function.name == "get_weather"
        assert domain_tool_call.function.arguments == '{"location": "San Francisco"}'

    def test_domain_to_api_tool_call_converts_correctly(self):
        """Test that domain_to_api_tool_call converts domain tool call to API."""
        domain_tool_call = ToolCall(
            id="call_456",
            type="function",
            function=ToolCallFunction(
                name="calculate",
                arguments='{"x": 10, "y": 20}',
            ),
        )

        api_tool_call = domain_to_api_tool_call(domain_tool_call)

        assert isinstance(api_tool_call, APIToolCall)
        assert api_tool_call.id == "call_456"
        assert api_tool_call.type == "function"
        assert api_tool_call.function.name == "calculate"
        assert api_tool_call.function.arguments == '{"x": 10, "y": 20}'


class TestResponseFormatResolution:
    """Behavioral tests for response format resolution."""

    def test_resolve_format_returns_direct_format_when_no_response_format(self):
        """Test that _resolve_response_format returns direct format when response_format is None."""
        from shared_ollama.api.mappers import _resolve_response_format

        result = _resolve_response_format(direct_format="json", response_format=None)
        assert result == "json"

    def test_resolve_format_returns_json_for_json_object(self):
        """Test that _resolve_response_format returns 'json' for json_object type."""
        from shared_ollama.api.mappers import _resolve_response_format

        response_format = ResponseFormat(type="json_object")
        result = _resolve_response_format(direct_format=None, response_format=response_format)
        assert result == "json"

    def test_resolve_format_returns_schema_for_json_schema(self):
        """Test that _resolve_response_format returns schema for json_schema type."""
        from shared_ollama.api.mappers import _resolve_response_format

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        response_format = ResponseFormat(type="json_schema", json_schema=schema)
        result = _resolve_response_format(direct_format=None, response_format=response_format)
        assert result == schema

    def test_resolve_format_handles_nested_schema(self):
        """Test that _resolve_response_format handles nested schema structure."""
        from shared_ollama.api.mappers import _resolve_response_format

        nested_schema = {"name": "Person", "schema": {"type": "object", "properties": {"name": {"type": "string"}}}}
        response_format = ResponseFormat(type="json_schema", json_schema=nested_schema)
        result = _resolve_response_format(direct_format=None, response_format=response_format)
        # Should extract nested schema
        assert isinstance(result, dict)
        assert "type" in result

    def test_resolve_format_raises_error_for_missing_schema(self):
        """Test that _resolve_response_format raises error when schema missing."""
        from pydantic import ValidationError

        # Pydantic V2 validates during construction, so ValidationError is raised
        # when trying to create ResponseFormat with type='json_schema' but json_schema=None
        with pytest.raises(ValidationError, match="json_schema"):
            ResponseFormat(type="json_schema", json_schema=None)

    def test_resolve_format_returns_none_for_text_type(self):
        """Test that _resolve_response_format returns None for text type."""
        from shared_ollama.api.mappers import _resolve_response_format

        response_format = ResponseFormat(type="text")
        result = _resolve_response_format(direct_format=None, response_format=response_format)
        assert result is None


class TestGenerationRequestMapper:
    """Behavioral tests for generation request mapping."""

    def test_api_to_domain_generation_request_converts_basic_request(self):
        """Test that api_to_domain_generation_request converts basic request."""
        api_req = APIGenerateRequest(prompt="Hello, world!", model="qwen3-vl:8b-instruct-q4_K_M")

        domain_req = api_to_domain_generation_request(api_req)

        assert isinstance(domain_req, GenerationRequest)
        assert domain_req.prompt.value == "Hello, world!"
        assert domain_req.model is not None
        assert domain_req.model.value == "qwen3-vl:8b-instruct-q4_K_M"
        assert domain_req.system is None
        assert domain_req.options is None

    def test_api_to_domain_generation_request_includes_system_message(self):
        """Test that mapper includes system message when provided."""
        api_req = APIGenerateRequest(
            prompt="Test",
            model="qwen3-vl:8b-instruct-q4_K_M",
            system="You are helpful",
        )

        domain_req = api_to_domain_generation_request(api_req)

        assert domain_req.system is not None
        assert domain_req.system.value == "You are helpful"

    def test_api_to_domain_generation_request_creates_options_when_provided(self):
        """Test that mapper creates GenerationOptions when any option is provided."""
        api_req = APIGenerateRequest(
            prompt="Test",
            model="qwen3-vl:8b-instruct-q4_K_M",
            temperature=0.7,
            top_p=0.95,
            max_tokens=100,
        )

        domain_req = api_to_domain_generation_request(api_req)

        assert domain_req.options is not None
        assert domain_req.options.temperature == 0.7
        assert domain_req.options.top_p == 0.95
        assert domain_req.options.max_tokens == 100

    def test_api_to_domain_generation_request_handles_none_options(self):
        """Test that mapper handles None option values correctly."""
        api_req = APIGenerateRequest(
            prompt="Test",
            model="qwen3-vl:8b-instruct-q4_K_M",
            temperature=None,
            top_p=None,
            max_tokens=None,
        )

        domain_req = api_to_domain_generation_request(api_req)

        # Should not create options if all are None
        assert domain_req.options is None

    def test_api_to_domain_generation_request_uses_defaults_for_partial_options(self):
        """Test that mapper uses defaults for partial option values."""
        api_req = APIGenerateRequest(
            prompt="Test",
            model="qwen3-vl:8b-instruct-q4_K_M",
            temperature=0.7,
            # top_p and top_k not provided
        )

        domain_req = api_to_domain_generation_request(api_req)

        assert domain_req.options is not None
        assert domain_req.options.temperature == 0.7
        assert domain_req.options.top_p == 0.9  # Default
        assert domain_req.options.top_k == 40  # Default

    def test_api_to_domain_generation_request_includes_tools(self):
        """Test that mapper includes tools when provided."""
        api_req = APIGenerateRequest(
            prompt="Test",
            model="qwen3-vl:8b-instruct-q4_K_M",
            tools=[
                APITool(
                    type="function",
                    function=APIToolFunction(
                        name="test_func",
                        description="Test",
                        parameters={},
                    ),
                )
            ],
        )

        domain_req = api_to_domain_generation_request(api_req)

        assert domain_req.tools is not None
        assert len(domain_req.tools) == 1
        assert domain_req.tools[0].function.name == "test_func"

    def test_api_to_domain_generation_request_handles_format_json(self):
        """Test that mapper handles format='json' correctly."""
        api_req = APIGenerateRequest(
            prompt="Test",
            model="qwen3-vl:8b-instruct-q4_K_M",
            format="json",
        )

        domain_req = api_to_domain_generation_request(api_req)

        assert domain_req.format == "json"

    def test_api_to_domain_generation_request_handles_response_format_json_object(self):
        """Test that mapper handles response_format with json_object type."""
        api_req = APIGenerateRequest(
            prompt="Test",
            model="qwen3-vl:8b-instruct-q4_K_M",
            response_format=ResponseFormat(type="json_object"),
        )

        domain_req = api_to_domain_generation_request(api_req)

        assert domain_req.format == "json"

    def test_api_to_domain_generation_request_handles_response_format_json_schema(self):
        """Test that mapper handles response_format with json_schema type."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        api_req = APIGenerateRequest(
            prompt="Test",
            model="qwen3-vl:8b-instruct-q4_K_M",
            response_format=ResponseFormat(type="json_schema", json_schema=schema),
        )

        domain_req = api_to_domain_generation_request(api_req)

        assert isinstance(domain_req.format, dict)
        assert domain_req.format == schema

    def test_api_to_domain_generation_request_handles_none_model(self):
        """Test that mapper handles None model correctly."""
        api_req = APIGenerateRequest(prompt="Test", model=None)

        domain_req = api_to_domain_generation_request(api_req)

        assert domain_req.model is None


class TestChatRequestMapper:
    """Behavioral tests for chat request mapping."""

    def test_api_to_domain_chat_request_converts_basic_request(self):
        """Test that api_to_domain_chat_request converts basic request."""
        api_req = APIChatRequest(
            messages=[APIChatMessage(role="user", content="Hello!")],
            model="qwen3-vl:8b-instruct-q4_K_M",
        )

        domain_req = api_to_domain_chat_request(api_req)

        assert isinstance(domain_req, ChatRequest)
        assert len(domain_req.messages) == 1
        assert domain_req.messages[0].role == "user"
        assert domain_req.messages[0].content == "Hello!"
        assert domain_req.model is not None
        assert domain_req.model.value == "qwen3-vl:8b-instruct-q4_K_M"

    def test_api_to_domain_chat_request_handles_multiple_messages(self):
        """Test that mapper handles multiple messages correctly."""
        api_req = APIChatRequest(
            messages=[
                APIChatMessage(role="system", content="You are helpful"),
                APIChatMessage(role="user", content="What is 2+2?"),
                APIChatMessage(role="assistant", content="4"),
            ],
            model="qwen3-vl:8b-instruct-q4_K_M",
        )

        domain_req = api_to_domain_chat_request(api_req)

        assert len(domain_req.messages) == 3
        assert domain_req.messages[0].role == "system"
        assert domain_req.messages[1].role == "user"
        assert domain_req.messages[2].role == "assistant"

    def test_api_to_domain_chat_request_includes_tool_calls(self):
        """Test that mapper includes tool calls when present."""
        api_req = APIChatRequest(
            messages=[
                APIChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        APIToolCall(
                            id="call_1",
                            type="function",
                            function=APIToolCallFunction(name="get_weather", arguments='{"location": "SF"}'),
                        )
                    ],
                )
            ],
            model="qwen3-vl:8b-instruct-q4_K_M",
        )

        domain_req = api_to_domain_chat_request(api_req)

        assert domain_req.messages[0].tool_calls is not None
        assert len(domain_req.messages[0].tool_calls) == 1
        assert domain_req.messages[0].tool_calls[0].id == "call_1"

    def test_api_to_domain_chat_request_includes_tool_call_id(self):
        """Test that mapper includes tool_call_id when present."""
        api_req = APIChatRequest(
            messages=[
                APIChatMessage(
                    role="tool",
                    content="Weather is sunny",
                    tool_call_id="call_1",
                )
            ],
            model="qwen3-vl:8b-instruct-q4_K_M",
        )

        domain_req = api_to_domain_chat_request(api_req)

        assert domain_req.messages[0].tool_call_id == "call_1"

    def test_api_to_domain_chat_request_handles_empty_messages(self):
        """Test that mapper rejects empty messages list (validation error)."""
        from pydantic import ValidationError

        # Chat requests must have at least one message
        with pytest.raises(ValidationError, match="at least 1 item"):
            APIChatRequest(messages=[], model="qwen3-vl:8b-instruct-q4_K_M")

    def test_api_to_domain_chat_request_handles_none_model(self):
        """Test that mapper handles None model correctly."""
        api_req = APIChatRequest(messages=[APIChatMessage(role="user", content="Test")], model=None)

        domain_req = api_to_domain_chat_request(api_req)

        assert domain_req.model is None


class TestVLMRequestMapper:
    """Behavioral tests for VLM request mapping."""

    def test_api_to_domain_vlm_request_converts_basic_request(self):
        """Test that api_to_domain_vlm_request converts basic request."""
        api_req = APIVLMRequest(
            messages=[APIMessage(role="user", content="What's in this image?")],
            images=["data:image/jpeg;base64,/9j/4AAQSkZJRg=="],
            model="qwen3-vl:8b-instruct-q4_K_M",
        )

        domain_req = api_to_domain_vlm_request(api_req)

        assert isinstance(domain_req, VLMRequest)
        assert len(domain_req.messages) == 1
        assert domain_req.messages[0].images is not None
        assert len(domain_req.messages[0].images) == 1
        assert domain_req.model is not None
        assert domain_req.model.value == "qwen3-vl:8b-instruct-q4_K_M"

    def test_api_to_domain_vlm_request_handles_multiple_images(self):
        """Test that mapper handles multiple images correctly."""
        api_req = APIVLMRequest(
            messages=[APIMessage(role="user", content="Compare these images")],
            images=[
                "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
            ],
            model="qwen3-vl:8b-instruct-q4_K_M",
        )

        domain_req = api_to_domain_vlm_request(api_req)

        assert domain_req.messages[0].images is not None
        assert len(domain_req.messages[0].images) == 2

    def test_api_to_domain_vlm_request_openai_converts_multimodal_messages(self):
        """Test that api_to_domain_vlm_request_openai converts OpenAI format."""
        api_req = VLMRequestOpenAI(
            messages=[
                APIMessageOpenAI(
                    role="user",
                    content=[
                        TextContentPart(type="text", text="What's in this image?"),
                        ImageContentPart(
                            type="image_url",
                            image_url={"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="},
                        ),
                    ],
                )
            ],
            model="qwen3-vl:8b-instruct-q4_K_M",
        )

        domain_req = api_to_domain_vlm_request_openai(api_req)

        assert isinstance(domain_req, VLMRequest)
        assert len(domain_req.messages) == 1
        assert domain_req.messages[0].images is not None
        assert len(domain_req.messages[0].images) == 1
        assert "What's in this image?" in domain_req.messages[0].content

    def test_api_to_domain_vlm_request_openai_handles_text_only_messages(self):
        """Test that mapper rejects text-only messages (VLM requires images)."""
        from pydantic import ValidationError

        # VLM requests must contain at least one image
        with pytest.raises(ValidationError, match="VLM requests must contain at least one image"):
            VLMRequestOpenAI(
                messages=[
                    APIMessageOpenAI(
                        role="user",
                        content=[TextContentPart(type="text", text="Hello!")],
                    )
                ],
                model="qwen3-vl:8b-instruct-q4_K_M",
            )

    def test_api_to_domain_vlm_request_openai_handles_image_only_messages(self):
        """Test that mapper handles image-only messages (adds default prompt)."""
        api_req = VLMRequestOpenAI(
            messages=[
                APIMessageOpenAI(
                    role="user",
                    content=[
                        ImageContentPart(
                            type="image_url",
                            image_url={"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="},
                        )
                    ],
                )
            ],
            model="qwen3-vl:8b-instruct-q4_K_M",
        )

        domain_req = api_to_domain_vlm_request_openai(api_req)

        assert len(domain_req.messages) == 1
        assert domain_req.messages[0].images is not None
        assert len(domain_req.messages[0].images) == 1
        # Should add default prompt for image-only message
        assert len(domain_req.messages[0].content) > 0

    def test_api_to_domain_vlm_request_handles_image_compression(self):
        """Test that mapper includes image compression settings."""
        api_req = APIVLMRequest(
            messages=[APIMessage(role="user", content="Test")],
            images=["data:image/jpeg;base64,/9j/4AAQSkZJRg=="],
            model="qwen3-vl:8b-instruct-q4_K_M",
            image_compression=True,
            max_dimension=1024,
        )

        domain_req = api_to_domain_vlm_request(api_req)

        assert domain_req.image_compression is True
        assert domain_req.max_dimension == 1024


class TestModelInfoMapper:
    """Behavioral tests for model info mapping."""

    def test_domain_to_api_model_info_converts_correctly(self):
        """Test that domain_to_api_model_info converts domain model to API model."""
        domain_model = ModelInfo(
            name="qwen3-vl:8b-instruct-q4_K_M",
            size=5969245856,
            modified_at="2025-11-03T17:24:58.744838946Z",
        )

        api_model = domain_to_api_model_info(domain_model)

        assert isinstance(api_model, APIModelInfo)
        assert api_model.name == "qwen3-vl:8b-instruct-q4_K_M"
        assert api_model.size == 5969245856
        assert api_model.modified_at == "2025-11-03T17:24:58.744838946Z"

    def test_domain_to_api_model_info_handles_none_values(self):
        """Test that mapper handles None values correctly."""
        domain_model = ModelInfo(name="test-model", size=None, modified_at=None)

        api_model = domain_to_api_model_info(domain_model)

        assert api_model.name == "test-model"
        assert api_model.size is None
        assert api_model.modified_at is None


class TestMapperEdgeCases:
    """Edge case and error handling tests for mappers."""

    def test_generation_request_mapper_handles_empty_prompt(self):
        """Test that mapper rejects empty prompt (validation error)."""
        from pydantic import ValidationError

        # Empty prompt should be rejected by Pydantic validation
        with pytest.raises(ValidationError, match="at least 1 character"):
            APIGenerateRequest(prompt="", model="qwen3-vl:8b-instruct-q4_K_M")

    def test_chat_request_mapper_handles_very_long_content(self):
        """Test that mapper handles very long message content."""
        long_content = "A" * 10000
        api_req = APIChatRequest(
            messages=[APIChatMessage(role="user", content=long_content)],
            model="qwen3-vl:8b-instruct-q4_K_M",
        )

        domain_req = api_to_domain_chat_request(api_req)

        assert len(domain_req.messages[0].content) == 10000

    def test_vlm_request_mapper_handles_many_images(self):
        """Test that mapper handles many images."""
        images = [f"data:image/jpeg;base64,image_{i}" for i in range(10)]
        api_req = APIVLMRequest(
            messages=[APIMessage(role="user", content="Analyze these")],
            images=images,
            model="qwen3-vl:8b-instruct-q4_K_M",
        )

        domain_req = api_to_domain_vlm_request(api_req)

        assert domain_req.messages[0].images is not None
        assert len(domain_req.messages[0].images) == 10

    @pytest.mark.parametrize(
        "format_value,response_format_type,expected",
        [
            ("json", None, "json"),
            (None, "json_object", "json"),
            (None, "json_schema", dict),
            (None, "text", None),
            ("json", "json_object", "json"),  # direct format takes precedence
        ],
    )
    def test_format_resolution_various_combinations(self, format_value, response_format_type, expected):
        """Test format resolution with various combinations."""
        from shared_ollama.api.mappers import _resolve_response_format

        response_format = None
        if response_format_type:
            if response_format_type == "json_schema":
                response_format = ResponseFormat(
                    type="json_schema",
                    json_schema={"type": "object", "properties": {"name": {"type": "string"}}},
                )
            else:
                response_format = ResponseFormat(type=response_format_type)

        result = _resolve_response_format(direct_format=format_value, response_format=response_format)

        if expected == dict:
            assert isinstance(result, dict)
        else:
            assert result == expected
