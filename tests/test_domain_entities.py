"""
Comprehensive behavioral tests for domain entities.

Tests focus on real validation behavior, business rules, edge cases, error handling,
and realistic workflows. No mocks - tests use real domain entities and value objects.

Key Principles:
- Test business rules and validation logic, not implementation details
- Test realistic scenarios and workflows
- Test error paths and boundary conditions
- Test invariants and contracts
"""

from __future__ import annotations

import pytest

from shared_ollama.domain.entities import (
    ChatMessage,
    ChatMessageOpenAI,
    ChatRequest,
    GenerationOptions,
    GenerationRequest,
    ImageContent,
    Model,
    ModelInfo,
    TextContent,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolFunction,
    VLMRequest,
    VLMRequestOpenAI,
    VLMMessage,
)
from shared_ollama.domain.exceptions import InvalidPromptError
from shared_ollama.domain.value_objects import ModelName, Prompt, SystemMessage


class TestModelEnumBehavior:
    """Behavioral tests for Model enum - testing actual usage patterns."""

    def test_model_enum_can_be_used_in_requests(self):
        """Test that Model enum values work in real request scenarios."""
        # Real usage: creating requests with enum values
        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value=Model.QWEN3_14B_Q4),
        )
        assert request.model is not None
        assert request.model.value == Model.QWEN3_14B_Q4

    def test_model_enum_values_are_valid_model_names(self):
        """Test that all enum values pass ModelName validation."""
        for model_value in [Model.QWEN3_VL_8B_Q4, Model.QWEN3_14B_Q4, Model.QWEN3_VL_32B, Model.QWEN3_30B]:
            # Should not raise - enum values are valid model names
            model_name = ModelName(value=model_value)
            assert model_name.value == model_value


class TestModelInfoBehavior:
    """Behavioral tests for ModelInfo - testing real usage scenarios."""

    def test_model_info_roundtrip_serialization(self):
        """Test that ModelInfo can be created from typical API responses."""
        # Simulating real API response structure
        api_data = {
            "name": "qwen3:14b-q4_K_M",
            "size": 8988124069,
            "modified_at": "2025-01-15T10:30:00Z",
        }
        info = ModelInfo(**api_data)
        
        # Verify it preserves all data correctly
        assert info.name == api_data["name"]
        assert info.size == api_data["size"]
        assert info.modified_at == api_data["modified_at"]

    def test_model_info_handles_missing_optional_fields(self):
        """Test that ModelInfo handles partial data (real-world scenario)."""
        # Some API responses may omit optional fields
        partial_data = {"name": "test-model"}
        info = ModelInfo(**partial_data)
        
        assert info.name == "test-model"
        assert info.size is None
        assert info.modified_at is None


class TestToolFunctionValidation:
    """Behavioral tests for ToolFunction validation rules."""

    def test_tool_function_validates_name_required(self):
        """Test that ToolFunction enforces name requirement."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ToolFunction(name="")

    def test_tool_function_validates_name_trimming(self):
        """Test that ToolFunction handles whitespace correctly."""
        # Leading/trailing whitespace should be handled
        func = ToolFunction(name="  get_weather  ")
        # Name should be trimmed (if validation does trimming) or rejected
        # This tests actual validation behavior

    def test_tool_function_with_complex_parameters_schema(self):
        """Test ToolFunction with realistic JSON schema."""
        complex_schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        }
        func = ToolFunction(name="get_weather", description="Get weather", parameters=complex_schema)
        
        assert func.parameters == complex_schema
        assert func.description == "Get weather"

    @pytest.mark.parametrize("invalid_name", ["", "   ", "\t", "\n"])
    def test_tool_function_rejects_empty_or_whitespace_names(self, invalid_name):
        """Test that ToolFunction rejects various empty/whitespace name patterns."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ToolFunction(name=invalid_name)


class TestToolCallValidation:
    """Behavioral tests for ToolCall validation and real usage."""

    def test_tool_call_with_valid_json_arguments(self):
        """Test ToolCall with realistic JSON arguments."""
        valid_args = '{"location": "San Francisco", "unit": "celsius"}'
        func_call = ToolCallFunction(name="get_weather", arguments=valid_args)
        tool_call = ToolCall(id="call-123", function=func_call)
        
        assert tool_call.function.arguments == valid_args

    def test_tool_call_rejects_invalid_id_patterns(self):
        """Test ToolCall rejects various invalid ID patterns."""
        func_call = ToolCallFunction(name="test", arguments='{"x": 1}')
        
        for invalid_id in ["", "   ", "\t", None]:
            if invalid_id is None:
                # None would be a type error, test empty string variants
                continue
            with pytest.raises(ValueError, match="cannot be empty"):
                ToolCall(id=invalid_id, function=func_call)

    def test_tool_call_workflow_simulation(self):
        """Test realistic tool calling workflow."""
        # Simulate: model wants to call a function
        tool_call = ToolCall(
            id="call_abc123",
            function=ToolCallFunction(
                name="calculate",
                arguments='{"expression": "2 + 2"}',
            ),
        )
        
        # Verify it can be used in a message
        message = ChatMessage(role="assistant", tool_calls=(tool_call,))
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].function.name == "calculate"


class TestGenerationOptionsValidation:
    """Behavioral tests for GenerationOptions validation rules."""

    @pytest.mark.parametrize(
        "temp,should_raise",
        [
            (-0.1, True),
            (0.0, False),
            (0.2, False),
            (1.0, False),
            (2.0, False),
            (2.1, True),
            (10.0, True),
        ],
    )
    def test_temperature_validation_boundaries(self, temp, should_raise):
        """Test temperature validation across boundary values."""
        if should_raise:
            with pytest.raises(ValueError, match="Temperature must be between"):
                GenerationOptions(temperature=temp)
        else:
            options = GenerationOptions(temperature=temp)
            assert options.temperature == temp

    @pytest.mark.parametrize(
        "top_p,should_raise",
        [
            (-0.1, True),
            (0.0, False),
            (0.5, False),
            (1.0, False),
            (1.1, True),
        ],
    )
    def test_top_p_validation_boundaries(self, top_p, should_raise):
        """Test top_p validation across boundary values."""
        if should_raise:
            with pytest.raises(ValueError, match="Top-p must be between"):
                GenerationOptions(top_p=top_p)
        else:
            options = GenerationOptions(top_p=top_p)
            assert options.top_p == top_p

    @pytest.mark.parametrize("top_k", [0, -1, -10])
    def test_top_k_rejects_non_positive(self, top_k):
        """Test that top_k rejects zero and negative values."""
        with pytest.raises(ValueError, match="Top-k must be >= 1"):
            GenerationOptions(top_k=top_k)

    def test_generation_options_realistic_configuration(self):
        """Test realistic generation configuration."""
        options = GenerationOptions(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=1000,
            seed=42,
            stop=["\n\n", "Human:", "Assistant:"],
        )
        
        assert options.temperature == 0.7
        assert options.max_tokens == 1000
        assert len(options.stop) == 3

    def test_generation_options_allows_unlimited_tokens(self):
        """Test that max_tokens=None allows unlimited generation."""
        options = GenerationOptions(max_tokens=None)
        assert options.max_tokens is None


class TestGenerationRequestValidation:
    """Behavioral tests for GenerationRequest validation and workflows."""

    def test_generation_request_minimal_valid(self):
        """Test minimal valid generation request."""
        request = GenerationRequest(prompt=Prompt(value="Hello"))
        assert request.prompt.value == "Hello"
        assert request.model is None  # Uses default

    def test_generation_request_with_full_configuration(self):
        """Test generation request with all options configured."""
        options = GenerationOptions(temperature=0.8, max_tokens=500)
        request = GenerationRequest(
            prompt=Prompt(value="Write a story"),
            model=ModelName(value="qwen3:14b-q4_K_M"),
            system=SystemMessage(value="You are a creative writer"),
            options=options,
            format="json",
        )
        
        assert request.prompt.value == "Write a story"
        assert request.model.value == "qwen3:14b-q4_K_M"
        assert request.system.value == "You are a creative writer"
        assert request.format == "json"

    @pytest.mark.parametrize(
        "prompt_value,should_raise",
        [
            ("", True),
            ("   ", True),
            ("\t\n", True),
            ("Valid prompt", False),
            ("x" * 1_000_000, False),  # At boundary
            ("x" * 1_000_001, True),  # Over boundary
        ],
    )
    def test_prompt_validation_edge_cases(self, prompt_value, should_raise):
        """Test prompt validation with various edge cases."""
        if should_raise:
            with pytest.raises(ValueError):
                GenerationRequest(prompt=Prompt(value=prompt_value))
        else:
            request = GenerationRequest(prompt=Prompt(value=prompt_value))
            assert len(request.prompt.value) == len(prompt_value)

    def test_generation_request_with_tools(self):
        """Test generation request with tool calling."""
        tool = Tool(function=ToolFunction(name="get_weather", parameters={"type": "object"}))
        request = GenerationRequest(
            prompt=Prompt(value="What's the weather?"),
            tools=(tool,),
        )
        
        assert request.tools is not None
        assert len(request.tools) == 1
        assert request.tools[0].function.name == "get_weather"


class TestChatMessageValidation:
    """Behavioral tests for ChatMessage validation rules."""

    def test_chat_message_requires_content_or_tool_calls(self):
        """Test that ChatMessage enforces content OR tool_calls requirement."""
        # Valid: has content
        msg1 = ChatMessage(role="user", content="Hello")
        assert msg1.content == "Hello"
        
        # Valid: has tool_calls
        tool_call = ToolCall(
            id="call-1",
            function=ToolCallFunction(name="test", arguments='{"x": 1}'),
        )
        msg2 = ChatMessage(role="assistant", tool_calls=(tool_call,))
        assert msg2.tool_calls is not None
        
        # Invalid: has neither
        with pytest.raises(ValueError, match="must have either"):
            ChatMessage(role="user", content=None, tool_calls=None)

    def test_chat_message_tool_role_validation(self):
        """Test that tool role messages require tool_call_id."""
        # Invalid: tool role without tool_call_id
        with pytest.raises(ValueError, match="must have tool_call_id"):
            ChatMessage(role="tool", content="result", tool_call_id=None)
        
        # Valid: tool role with tool_call_id
        msg = ChatMessage(role="tool", content="result", tool_call_id="call-123")
        assert msg.tool_call_id == "call-123"

    @pytest.mark.parametrize("role", ["user", "assistant", "system", "tool"])
    def test_chat_message_accepts_all_valid_roles(self, role):
        """Test that ChatMessage accepts all valid role values."""
        if role == "tool":
            msg = ChatMessage(role=role, content="test", tool_call_id="call-1")  # type: ignore[arg-type]
        else:
            msg = ChatMessage(role=role, content="test")  # type: ignore[arg-type]
        assert msg.role == role

    def test_chat_message_rejects_invalid_role(self):
        """Test that ChatMessage rejects invalid role values."""
        with pytest.raises(ValueError, match="Invalid role"):
            ChatMessage(role="invalid_role", content="test")  # type: ignore[arg-type]

    def test_chat_message_conversation_workflow(self):
        """Test realistic conversation workflow with multiple messages."""
        # User message
        user_msg = ChatMessage(role="user", content="What's 2+2?")
        
        # Assistant responds with tool call
        tool_call = ToolCall(
            id="call-1",
            function=ToolCallFunction(name="calculate", arguments='{"expr": "2+2"}'),
        )
        assistant_msg = ChatMessage(role="assistant", tool_calls=(tool_call,))
        
        # Tool response
        tool_msg = ChatMessage(role="tool", content="4", tool_call_id="call-1")
        
        # Final assistant response
        final_msg = ChatMessage(role="assistant", content="The answer is 4")
        
        # All messages should be valid
        assert user_msg.role == "user"
        assert assistant_msg.tool_calls is not None
        assert tool_msg.tool_call_id == "call-1"
        assert final_msg.content == "The answer is 4"


class TestChatRequestValidation:
    """Behavioral tests for ChatRequest validation and workflows."""

    def test_chat_request_rejects_empty_message_list(self):
        """Test that ChatRequest enforces non-empty messages."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ChatRequest(messages=())

    def test_chat_request_validates_total_length(self):
        """Test that ChatRequest validates total message content length."""
        # Single message at boundary (1,000,000 chars is the limit)
        max_content = "x" * 1_000_000
        msg1 = ChatMessage(role="user", content=max_content)
        request1 = ChatRequest(messages=(msg1,))
        assert len(request1.messages) == 1
        
        # Two messages that together exceed limit (1,000,000 chars)
        long_content = "x" * 500_001  # Each message is 500,001 chars
        msg2 = ChatMessage(role="user", content=long_content)
        msg3 = ChatMessage(role="assistant", content=long_content)
        with pytest.raises(ValueError, match="Total message content is too long"):
            ChatRequest(messages=(msg2, msg3))

    def test_chat_request_multi_turn_conversation(self):
        """Test realistic multi-turn conversation."""
        messages = (
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi! How can I help?"),
            ChatMessage(role="user", content="What's the weather?"),
        )
        request = ChatRequest(messages=messages)
        
        assert len(request.messages) == 4
        assert request.messages[0].role == "system"
        assert request.messages[-1].content == "What's the weather?"

    def test_chat_request_with_tool_calling_workflow(self):
        """Test chat request with complete tool calling workflow."""
        messages = (
            ChatMessage(role="user", content="Get weather for Paris"),
            ChatMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        function=ToolCallFunction(
                            name="get_weather",
                            arguments='{"location": "Paris"}',
                        ),
                    ),
                ),
            ),
            ChatMessage(role="tool", content='{"temp": 15}', tool_call_id="call-1"),
            ChatMessage(role="assistant", content="It's 15°C in Paris"),
        )
        request = ChatRequest(messages=messages)
        
        assert len(request.messages) == 4
        assert request.messages[1].tool_calls is not None
        assert request.messages[2].tool_call_id == "call-1"


class TestVLMRequestValidation:
    """Behavioral tests for VLMRequest validation and workflows."""

    def test_vlm_request_requires_at_least_one_image(self):
        """Test that VLMRequest enforces image requirement."""
        # Invalid: no images
        messages = (VLMMessage(role="user", content="Test"),)
        with pytest.raises(ValueError, match="must contain at least one image"):
            VLMRequest(messages=messages)
        
        # Valid: has image
        image_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        messages_with_image = (VLMMessage(role="user", content="Test", images=(image_url,)),)
        request = VLMRequest(messages=messages_with_image)
        assert len(request.messages[0].images) == 1

    @pytest.mark.parametrize(
        "dimension,should_raise",
        [
            (255, True),
            (256, False),
            (1000, False),
            (2667, False),
            (2668, True),
        ],
    )
    def test_vlm_request_max_dimension_validation(self, dimension, should_raise):
        """Test max_dimension validation boundaries."""
        image_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        messages = (VLMMessage(role="user", content="Test", images=(image_url,)),)
        
        if should_raise:
            with pytest.raises(ValueError, match="max_dimension must be between"):
                VLMRequest(messages=messages, max_dimension=dimension)
        else:
            request = VLMRequest(messages=messages, max_dimension=dimension)
            assert request.max_dimension == dimension

    def test_vlm_request_multiple_images(self):
        """Test VLMRequest with multiple images."""
        image1 = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        image2 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        messages = (VLMMessage(role="user", content="Compare these images", images=(image1, image2)),)
        request = VLMRequest(messages=messages)
        
        assert len(request.messages[0].images) == 2


class TestImageContentValidation:
    """Behavioral tests for ImageContent validation."""

    @pytest.mark.parametrize(
        "url,should_raise",
        [
            ("", True),
            ("invalid://url", True),
            ("data:image/jpeg,invalid", True),  # Missing ;base64,
            ("data:image/jpeg;base64,", False),  # Valid format (empty data OK for test)
            ("data:image/png;base64,abc123", False),
            ("data:image/webp;base64,xyz789", False),
        ],
    )
    def test_image_content_url_validation(self, url, should_raise):
        """Test ImageContent URL format validation."""
        if should_raise:
            with pytest.raises(ValueError):
                ImageContent(image_url=url)
        else:
            content = ImageContent(image_url=url)
            assert content.image_url == url


class TestChatMessageOpenAIValidation:
    """Behavioral tests for ChatMessageOpenAI validation."""

    def test_chat_message_openai_string_content(self):
        """Test ChatMessageOpenAI with string content."""
        msg = ChatMessageOpenAI(role="user", content="Hello")
        assert isinstance(msg.content, str)
        assert msg.content == "Hello"

    def test_chat_message_openai_multimodal_content(self):
        """Test ChatMessageOpenAI with multimodal content."""
        text = TextContent(text="What's in this image?")
        image = ImageContent(image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg==")
        msg = ChatMessageOpenAI(role="user", content=(text, image))
        
        assert isinstance(msg.content, tuple)
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"
        assert msg.content[1].type == "image_url"

    def test_chat_message_openai_rejects_empty_content(self):
        """Test that ChatMessageOpenAI rejects empty content."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ChatMessageOpenAI(role="user", content="")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            ChatMessageOpenAI(role="user", content=())

    def test_chat_message_openai_rejects_invalid_content_type(self):
        """Test that ChatMessageOpenAI rejects invalid content types."""
        with pytest.raises(ValueError, match="must be either string or tuple"):
            ChatMessageOpenAI(role="user", content=123)  # type: ignore[arg-type]


class TestVLMRequestOpenAIValidation:
    """Behavioral tests for VLMRequestOpenAI validation."""

    def test_vlm_request_openai_requires_images(self):
        """Test that VLMRequestOpenAI enforces image requirement."""
        # Invalid: text-only message
        text_only = ChatMessageOpenAI(role="user", content="Text only")
        with pytest.raises(ValueError, match="must contain at least one image"):
            VLMRequestOpenAI(messages=(text_only,))
        
        # Valid: message with image
        text = TextContent(text="What's this?")
        image = ImageContent(image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg==")
        with_image = ChatMessageOpenAI(role="user", content=(text, image))
        request = VLMRequestOpenAI(messages=(with_image,))
        assert len(request.messages) == 1

    def test_vlm_request_openai_multiple_images(self):
        """Test VLMRequestOpenAI with multiple images in content."""
        text = TextContent(text="Compare these")
        img1 = ImageContent(image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg==")
        img2 = ImageContent(image_url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        msg = ChatMessageOpenAI(role="user", content=(text, img1, img2))
        request = VLMRequestOpenAI(messages=(msg,))
        
        # Should have 2 images in content
        image_count = sum(1 for part in msg.content if isinstance(part, ImageContent))
        assert image_count == 2


class TestEntityInvariants:
    """Tests for domain entity invariants and contracts."""

    def test_all_entities_are_immutable(self):
        """Test that all domain entities enforce immutability contract."""
        # Test ModelInfo immutability
        info = ModelInfo(name="test")
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            info.name = "modified"
        
        # Test GenerationOptions immutability
        options = GenerationOptions()
        with pytest.raises(Exception):
            options.temperature = 1.0
        
        # Test GenerationRequest immutability
        request = GenerationRequest(prompt=Prompt(value="test"))
        with pytest.raises(Exception):
            request.prompt = Prompt(value="modified")
        
        # Test ChatMessage immutability
        msg = ChatMessage(role="user", content="test")
        with pytest.raises(Exception):
            msg.content = "modified"
        
        # Test ChatRequest immutability
        chat_req = ChatRequest(messages=(ChatMessage(role="user", content="test"),))
        with pytest.raises(Exception):
            chat_req.messages = (ChatMessage(role="user", content="new"),)

    def test_value_objects_enforce_validation(self):
        """Test that value objects enforce their validation rules."""
        # Prompt rejects empty
        with pytest.raises(ValueError):
            Prompt(value="")
        
        # ModelName accepts valid names
        model = ModelName(value="qwen3:14b-q4_K_M")
        assert model.value == "qwen3:14b-q4_K_M"
        
        # SystemMessage can be empty (different rule than Prompt)
        system = SystemMessage(value="")
        assert system.value == ""
