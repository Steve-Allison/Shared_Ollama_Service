"""
Comprehensive behavioral tests for domain entities.

Tests focus on real validation behavior, edge cases, error handling, and business rules.
No mocks - tests use real domain entities and value objects.
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


class TestModelEnum:
    """Behavioral tests for Model enum."""

    def test_model_enum_values(self):
        """Test that Model enum has expected values."""
        assert Model.QWEN3_VL_8B_Q4 == "qwen3-vl:8b-instruct-q4_K_M"
        assert Model.QWEN3_14B_Q4 == "qwen3:14b-q4_K_M"
        assert Model.QWEN3_VL_32B == "qwen3-vl:32b"
        assert Model.QWEN3_30B == "qwen3:30b"

    def test_model_enum_is_string_enum(self):
        """Test that Model enum values are strings."""
        assert isinstance(Model.QWEN3_VL_8B_Q4, str)
        assert isinstance(Model.QWEN3_14B_Q4, str)


class TestModelInfo:
    """Behavioral tests for ModelInfo entity."""

    def test_model_info_creation(self):
        """Test that ModelInfo can be created with required fields."""
        info = ModelInfo(name="test-model")

        assert info.name == "test-model"
        assert info.size is None
        assert info.modified_at is None

    def test_model_info_with_optional_fields(self):
        """Test that ModelInfo can include optional fields."""
        info = ModelInfo(
            name="test-model",
            size=1024,
            modified_at="2025-01-01T00:00:00Z",
        )

        assert info.name == "test-model"
        assert info.size == 1024
        assert info.modified_at == "2025-01-01T00:00:00Z"

    def test_model_info_is_immutable(self):
        """Test that ModelInfo is immutable (frozen dataclass)."""
        info = ModelInfo(name="test-model")
        with pytest.raises(Exception):
            info.name = "new-name"


class TestToolFunction:
    """Behavioral tests for ToolFunction entity."""

    def test_tool_function_creation(self):
        """Test that ToolFunction can be created with name."""
        func = ToolFunction(name="test_function")

        assert func.name == "test_function"
        assert func.description is None
        assert func.parameters is None

    def test_tool_function_with_description(self):
        """Test that ToolFunction can include description."""
        func = ToolFunction(name="test_function", description="Test function")

        assert func.name == "test_function"
        assert func.description == "Test function"

    def test_tool_function_with_parameters(self):
        """Test that ToolFunction can include parameters schema."""
        params = {"type": "object", "properties": {"x": {"type": "number"}}}
        func = ToolFunction(name="test_function", parameters=params)

        assert func.parameters == params

    def test_tool_function_rejects_empty_name(self):
        """Test that ToolFunction rejects empty name."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ToolFunction(name="")

    def test_tool_function_rejects_whitespace_only_name(self):
        """Test that ToolFunction rejects whitespace-only name."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ToolFunction(name="   ")

    def test_tool_function_is_immutable(self):
        """Test that ToolFunction is immutable."""
        func = ToolFunction(name="test")
        with pytest.raises(Exception):
            func.name = "new"


class TestTool:
    """Behavioral tests for Tool entity."""

    def test_tool_creation(self):
        """Test that Tool can be created with function."""
        func = ToolFunction(name="test_function")
        tool = Tool(function=func)

        assert tool.function == func
        assert tool.type == "function"

    def test_tool_is_immutable(self):
        """Test that Tool is immutable."""
        func = ToolFunction(name="test")
        tool = Tool(function=func)
        with pytest.raises(Exception):
            tool.type = "invalid"


class TestToolCallFunction:
    """Behavioral tests for ToolCallFunction entity."""

    def test_tool_call_function_creation(self):
        """Test that ToolCallFunction can be created."""
        func_call = ToolCallFunction(name="test_function", arguments='{"x": 1}')

        assert func_call.name == "test_function"
        assert func_call.arguments == '{"x": 1}'

    def test_tool_call_function_rejects_empty_name(self):
        """Test that ToolCallFunction rejects empty name."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ToolCallFunction(name="", arguments='{"x": 1}')

    def test_tool_call_function_rejects_whitespace_only_name(self):
        """Test that ToolCallFunction rejects whitespace-only name."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ToolCallFunction(name="   ", arguments='{"x": 1}')

    def test_tool_call_function_rejects_empty_arguments(self):
        """Test that ToolCallFunction rejects empty arguments."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ToolCallFunction(name="test", arguments="")

    def test_tool_call_function_is_immutable(self):
        """Test that ToolCallFunction is immutable."""
        func_call = ToolCallFunction(name="test", arguments='{"x": 1}')
        with pytest.raises(Exception):
            func_call.name = "new"


class TestToolCall:
    """Behavioral tests for ToolCall entity."""

    def test_tool_call_creation(self):
        """Test that ToolCall can be created."""
        func_call = ToolCallFunction(name="test", arguments='{"x": 1}')
        tool_call = ToolCall(id="call-123", function=func_call)

        assert tool_call.id == "call-123"
        assert tool_call.function == func_call
        assert tool_call.type == "function"

    def test_tool_call_rejects_empty_id(self):
        """Test that ToolCall rejects empty ID."""
        func_call = ToolCallFunction(name="test", arguments='{"x": 1}')
        with pytest.raises(ValueError, match="cannot be empty"):
            ToolCall(id="", function=func_call)

    def test_tool_call_rejects_whitespace_only_id(self):
        """Test that ToolCall rejects whitespace-only ID."""
        func_call = ToolCallFunction(name="test", arguments='{"x": 1}')
        with pytest.raises(ValueError, match="cannot be empty"):
            ToolCall(id="   ", function=func_call)

    def test_tool_call_is_immutable(self):
        """Test that ToolCall is immutable."""
        func_call = ToolCallFunction(name="test", arguments='{"x": 1}')
        tool_call = ToolCall(id="call-123", function=func_call)
        with pytest.raises(Exception):
            tool_call.id = "new"


class TestGenerationOptions:
    """Behavioral tests for GenerationOptions entity."""

    def test_generation_options_defaults(self):
        """Test that GenerationOptions has sensible defaults."""
        options = GenerationOptions()

        assert options.temperature == 0.2
        assert options.top_p == 0.9
        assert options.top_k == 40
        assert options.repeat_penalty == 1.1
        assert options.max_tokens is None
        assert options.seed is None
        assert options.stop is None

    def test_generation_options_custom_values(self):
        """Test that GenerationOptions accepts custom values."""
        options = GenerationOptions(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            max_tokens=100,
            seed=42,
            stop=["\n"],
        )

        assert options.temperature == 0.7
        assert options.top_p == 0.95
        assert options.top_k == 50
        assert options.max_tokens == 100
        assert options.seed == 42
        assert options.stop == ["\n"]

    def test_generation_options_validates_temperature_min(self):
        """Test that GenerationOptions validates temperature minimum."""
        with pytest.raises(ValueError, match="Temperature must be between"):
            GenerationOptions(temperature=-0.1)

    def test_generation_options_validates_temperature_max(self):
        """Test that GenerationOptions validates temperature maximum."""
        with pytest.raises(ValueError, match="Temperature must be between"):
            GenerationOptions(temperature=2.1)

    def test_generation_options_validates_temperature_boundaries(self):
        """Test that GenerationOptions accepts temperature at boundaries."""
        # Should not raise
        GenerationOptions(temperature=0.0)
        GenerationOptions(temperature=2.0)

    def test_generation_options_validates_top_p_min(self):
        """Test that GenerationOptions validates top_p minimum."""
        with pytest.raises(ValueError, match="Top-p must be between"):
            GenerationOptions(top_p=-0.1)

    def test_generation_options_validates_top_p_max(self):
        """Test that GenerationOptions validates top_p maximum."""
        with pytest.raises(ValueError, match="Top-p must be between"):
            GenerationOptions(top_p=1.1)

    def test_generation_options_validates_top_p_boundaries(self):
        """Test that GenerationOptions accepts top_p at boundaries."""
        GenerationOptions(top_p=0.0)
        GenerationOptions(top_p=1.0)

    def test_generation_options_validates_top_k_min(self):
        """Test that GenerationOptions validates top_k minimum."""
        with pytest.raises(ValueError, match="Top-k must be >= 1"):
            GenerationOptions(top_k=0)

    def test_generation_options_validates_top_k_boundary(self):
        """Test that GenerationOptions accepts top_k at minimum boundary."""
        GenerationOptions(top_k=1)  # Should not raise

    def test_generation_options_validates_max_tokens_min(self):
        """Test that GenerationOptions validates max_tokens minimum when not None."""
        with pytest.raises(ValueError, match="Max tokens must be >= 1"):
            GenerationOptions(max_tokens=0)

    def test_generation_options_allows_none_max_tokens(self):
        """Test that GenerationOptions allows None max_tokens."""
        options = GenerationOptions(max_tokens=None)
        assert options.max_tokens is None

    def test_generation_options_is_immutable(self):
        """Test that GenerationOptions is immutable."""
        options = GenerationOptions()
        with pytest.raises(Exception):
            options.temperature = 1.0


class TestGenerationRequest:
    """Behavioral tests for GenerationRequest entity."""

    def test_generation_request_creation(self):
        """Test that GenerationRequest can be created with prompt."""
        request = GenerationRequest(prompt=Prompt(value="Hello"))

        assert request.prompt.value == "Hello"
        assert request.model is None
        assert request.system is None
        assert request.options is None
        assert request.format is None
        assert request.tools is None

    def test_generation_request_with_model(self):
        """Test that GenerationRequest can include model."""
        request = GenerationRequest(
            prompt=Prompt(value="Hello"),
            model=ModelName(value="qwen3:14b-q4_K_M"),
        )

        assert request.model is not None
        assert request.model.value == "qwen3:14b-q4_K_M"

    def test_generation_request_with_system(self):
        """Test that GenerationRequest can include system message."""
        request = GenerationRequest(
            prompt=Prompt(value="Hello"),
            system=SystemMessage(value="You are helpful"),
        )

        assert request.system is not None
        assert request.system.value == "You are helpful"

    def test_generation_request_with_options(self):
        """Test that GenerationRequest can include options."""
        options = GenerationOptions(temperature=0.7)
        request = GenerationRequest(prompt=Prompt(value="Hello"), options=options)

        assert request.options == options

    def test_generation_request_rejects_empty_prompt(self):
        """Test that GenerationRequest rejects empty prompt."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GenerationRequest(prompt=Prompt(value=""))

    def test_generation_request_rejects_whitespace_only_prompt(self):
        """Test that GenerationRequest rejects whitespace-only prompt."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GenerationRequest(prompt=Prompt(value="   "))

    def test_generation_request_rejects_too_long_prompt(self):
        """Test that GenerationRequest rejects prompt exceeding max length."""
        long_prompt = "x" * (1_000_001)  # Exceeds PROMPT_MAX_LENGTH
        with pytest.raises(ValueError, match="too long"):
            GenerationRequest(prompt=Prompt(value=long_prompt))

    def test_generation_request_accepts_max_length_prompt(self):
        """Test that GenerationRequest accepts prompt at max length."""
        max_prompt = "x" * 1_000_000  # Exactly PROMPT_MAX_LENGTH
        request = GenerationRequest(prompt=Prompt(value=max_prompt))
        assert len(request.prompt.value) == 1_000_000

    def test_generation_request_is_immutable(self):
        """Test that GenerationRequest is immutable."""
        request = GenerationRequest(prompt=Prompt(value="Hello"))
        with pytest.raises(Exception):
            request.prompt = Prompt(value="New")


class TestChatMessage:
    """Behavioral tests for ChatMessage entity."""

    def test_chat_message_creation_with_content(self):
        """Test that ChatMessage can be created with content."""
        msg = ChatMessage(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_chat_message_creation_with_tool_calls(self):
        """Test that ChatMessage can be created with tool_calls."""
        func_call = ToolCallFunction(name="test", arguments='{"x": 1}')
        tool_call = ToolCall(id="call-1", function=func_call)
        msg = ChatMessage(role="assistant", tool_calls=(tool_call,))

        assert msg.role == "assistant"
        assert msg.content is None
        assert msg.tool_calls == (tool_call,)

    def test_chat_message_tool_role_requires_tool_call_id(self):
        """Test that tool role messages require tool_call_id."""
        with pytest.raises(ValueError, match="must have tool_call_id"):
            ChatMessage(role="tool", content="result", tool_call_id=None)

    def test_chat_message_tool_role_with_tool_call_id(self):
        """Test that tool role messages work with tool_call_id."""
        msg = ChatMessage(role="tool", content="result", tool_call_id="call-1")
        assert msg.tool_call_id == "call-1"

    def test_chat_message_rejects_invalid_role(self):
        """Test that ChatMessage rejects invalid role."""
        with pytest.raises(ValueError, match="Invalid role"):
            ChatMessage(role="invalid", content="test")  # type: ignore[arg-type]

    def test_chat_message_rejects_no_content_no_tool_calls(self):
        """Test that ChatMessage requires either content or tool_calls."""
        with pytest.raises(ValueError, match="must have either"):
            ChatMessage(role="user", content=None, tool_calls=None)

    def test_chat_message_validates_all_valid_roles(self):
        """Test that ChatMessage accepts all valid roles."""
        for role in ["user", "assistant", "system", "tool"]:
            if role == "tool":
                msg = ChatMessage(role=role, content="test", tool_call_id="call-1")  # type: ignore[arg-type]
            else:
                msg = ChatMessage(role=role, content="test")  # type: ignore[arg-type]
            assert msg.role == role

    def test_chat_message_is_immutable(self):
        """Test that ChatMessage is immutable."""
        msg = ChatMessage(role="user", content="Hello")
        with pytest.raises(Exception):
            msg.content = "New"


class TestChatRequest:
    """Behavioral tests for ChatRequest entity."""

    def test_chat_request_creation(self):
        """Test that ChatRequest can be created."""
        messages = (ChatMessage(role="user", content="Hello"),)
        request = ChatRequest(messages=messages)

        assert len(request.messages) == 1
        assert request.model is None
        assert request.options is None

    def test_chat_request_rejects_empty_messages(self):
        """Test that ChatRequest rejects empty messages."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ChatRequest(messages=())

    def test_chat_request_validates_total_message_length(self):
        """Test that ChatRequest validates total message character length."""
        # Create messages that exceed MAX_TOTAL_MESSAGE_CHARS
        long_content = "x" * (1_000_001)
        messages = (ChatMessage(role="user", content=long_content),)
        with pytest.raises(ValueError, match="Total message content is too long"):
            ChatRequest(messages=messages)

    def test_chat_request_is_immutable(self):
        """Test that ChatRequest is immutable."""
        messages = (ChatMessage(role="user", content="Hello"),)
        request = ChatRequest(messages=messages)
        with pytest.raises(Exception):
            request.messages = (ChatMessage(role="user", content="New"),)


class TestVLMMessage:
    """Behavioral tests for VLMMessage entity."""

    def test_vlm_message_creation(self):
        """Test that VLMMessage can be created."""
        msg = VLMMessage(role="user", content="What's in this image?")

        assert msg.role == "user"
        assert msg.content == "What's in this image?"

    def test_vlm_message_rejects_invalid_role(self):
        """Test that VLMMessage rejects invalid role."""
        with pytest.raises(ValueError, match="Invalid role"):
            VLMMessage(role="invalid", content="test")  # type: ignore[arg-type]

    def test_vlm_message_is_immutable(self):
        """Test that VLMMessage is immutable."""
        msg = VLMMessage(role="user", content="Hello")
        with pytest.raises(Exception):
            msg.content = "New"


class TestVLMRequest:
    """Behavioral tests for VLMRequest entity."""

    def test_vlm_request_creation(self):
        """Test that VLMRequest can be created."""
        image_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        messages = (VLMMessage(role="user", content="What's in this image?", images=(image_url,)),)
        request = VLMRequest(messages=messages)

        assert len(request.messages) == 1
        assert request.messages[0].images is not None
        assert len(request.messages[0].images) == 1

    def test_vlm_request_rejects_empty_messages(self):
        """Test that VLMRequest rejects empty messages."""
        with pytest.raises(ValueError, match="cannot be empty"):
            VLMRequest(messages=())

    def test_vlm_request_rejects_empty_images(self):
        """Test that VLMRequest rejects empty images."""
        messages = (VLMMessage(role="user", content="Test"),)
        with pytest.raises(ValueError, match="must contain at least one image"):
            VLMRequest(messages=messages)

    def test_vlm_request_validates_max_dimension_min(self):
        """Test that VLMRequest validates max_dimension minimum."""
        image_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        messages = (VLMMessage(role="user", content="Test", images=(image_url,)),)
        with pytest.raises(ValueError, match="max_dimension must be between"):
            VLMRequest(messages=messages, max_dimension=255)

    def test_vlm_request_validates_max_dimension_max(self):
        """Test that VLMRequest validates max_dimension maximum."""
        image_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        messages = (VLMMessage(role="user", content="Test", images=(image_url,)),)
        with pytest.raises(ValueError, match="max_dimension must be between"):
            VLMRequest(messages=messages, max_dimension=2668)

    def test_vlm_request_validates_max_dimension_boundaries(self):
        """Test that VLMRequest accepts max_dimension at boundaries."""
        image_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        messages = (VLMMessage(role="user", content="Test", images=(image_url,)),)
        # Should not raise
        VLMRequest(messages=messages, max_dimension=256)
        VLMRequest(messages=messages, max_dimension=2667)

    def test_vlm_request_is_immutable(self):
        """Test that VLMRequest is immutable."""
        image_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        messages = (VLMMessage(role="user", content="Test", images=(image_url,)),)
        request = VLMRequest(messages=messages)
        with pytest.raises(Exception):
            request.messages = (VLMMessage(role="user", content="New"),)


class TestImageContent:
    """Behavioral tests for ImageContent entity."""

    def test_image_content_creation(self):
        """Test that ImageContent can be created."""
        content = ImageContent(image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg==")

        assert content.type == "image_url"
        assert content.image_url == "data:image/jpeg;base64,/9j/4AAQSkZJRg=="

    def test_image_content_rejects_empty_url(self):
        """Test that ImageContent rejects empty URL."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ImageContent(image_url="")

    def test_image_content_rejects_invalid_prefix(self):
        """Test that ImageContent rejects URL without data:image/ prefix."""
        with pytest.raises(ValueError, match="must start with 'data:image/'"):
            ImageContent(image_url="invalid://image")

    def test_image_content_rejects_missing_base64_separator(self):
        """Test that ImageContent rejects URL without ;base64, separator."""
        with pytest.raises(ValueError, match="must contain ';base64,'"):
            ImageContent(image_url="data:image/jpeg,invalid")

    def test_image_content_is_immutable(self):
        """Test that ImageContent is immutable."""
        content = ImageContent(image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg==")
        with pytest.raises(Exception):
            content.image_url = "new"


class TestTextContent:
    """Behavioral tests for TextContent entity."""

    def test_text_content_creation(self):
        """Test that TextContent can be created."""
        content = TextContent(text="Hello")

        assert content.type == "text"
        assert content.text == "Hello"

    def test_text_content_rejects_empty_text(self):
        """Test that TextContent rejects empty text."""
        with pytest.raises(ValueError, match="cannot be empty"):
            TextContent(text="")

    def test_text_content_rejects_whitespace_only_text(self):
        """Test that TextContent rejects whitespace-only text."""
        with pytest.raises(ValueError, match="cannot be empty"):
            TextContent(text="   ")

    def test_text_content_is_immutable(self):
        """Test that TextContent is immutable."""
        content = TextContent(text="Hello")
        with pytest.raises(Exception):
            content.text = "New"


class TestChatMessageOpenAI:
    """Behavioral tests for ChatMessageOpenAI entity."""

    def test_chat_message_openai_string_content(self):
        """Test that ChatMessageOpenAI can have string content."""
        msg = ChatMessageOpenAI(role="user", content="Hello")

        assert msg.role == "user"
        assert isinstance(msg.content, str)
        assert msg.content == "Hello"

    def test_chat_message_openai_multimodal_content(self):
        """Test that ChatMessageOpenAI can have multimodal content."""
        text_part = TextContent(text="What's in this image?")
        image_part = ImageContent(image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg==")
        msg = ChatMessageOpenAI(role="user", content=(text_part, image_part))

        assert msg.role == "user"
        assert isinstance(msg.content, tuple)
        assert len(msg.content) == 2

    def test_chat_message_openai_rejects_empty_string_content(self):
        """Test that ChatMessageOpenAI rejects empty string content."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ChatMessageOpenAI(role="user", content="")

    def test_chat_message_openai_rejects_empty_tuple_content(self):
        """Test that ChatMessageOpenAI rejects empty tuple content."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ChatMessageOpenAI(role="user", content=())

    def test_chat_message_openai_rejects_invalid_role(self):
        """Test that ChatMessageOpenAI rejects invalid role."""
        with pytest.raises(ValueError, match="Invalid role"):
            ChatMessageOpenAI(role="tool", content="test")  # type: ignore[arg-type]

    def test_chat_message_openai_rejects_invalid_content_type(self):
        """Test that ChatMessageOpenAI rejects invalid content type."""
        with pytest.raises(ValueError, match="must be either string or tuple"):
            ChatMessageOpenAI(role="user", content=123)  # type: ignore[arg-type]

    def test_chat_message_openai_is_immutable(self):
        """Test that ChatMessageOpenAI is immutable."""
        msg = ChatMessageOpenAI(role="user", content="Hello")
        with pytest.raises(Exception):
            msg.content = "New"


class TestVLMRequestOpenAI:
    """Behavioral tests for VLMRequestOpenAI entity."""

    def test_vlm_request_openai_creation(self):
        """Test that VLMRequestOpenAI can be created."""
        text_part = TextContent(text="What's in this image?")
        image_part = ImageContent(image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg==")
        msg = ChatMessageOpenAI(role="user", content=(text_part, image_part))
        request = VLMRequestOpenAI(messages=(msg,))

        assert len(request.messages) == 1

    def test_vlm_request_openai_rejects_empty_messages(self):
        """Test that VLMRequestOpenAI rejects empty messages."""
        with pytest.raises(ValueError, match="cannot be empty"):
            VLMRequestOpenAI(messages=())

    def test_vlm_request_openai_rejects_no_images(self):
        """Test that VLMRequestOpenAI rejects messages without images."""
        msg = ChatMessageOpenAI(role="user", content="Text only")
        with pytest.raises(ValueError, match="must contain at least one image"):
            VLMRequestOpenAI(messages=(msg,))

    def test_vlm_request_openai_validates_max_dimension(self):
        """Test that VLMRequestOpenAI validates max_dimension."""
        text_part = TextContent(text="Test")
        image_part = ImageContent(image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg==")
        msg = ChatMessageOpenAI(role="user", content=(text_part, image_part))
        with pytest.raises(ValueError, match="max_dimension must be between"):
            VLMRequestOpenAI(messages=(msg,), max_dimension=255)

    def test_vlm_request_openai_is_immutable(self):
        """Test that VLMRequestOpenAI is immutable."""
        text_part = TextContent(text="Test")
        image_part = ImageContent(image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg==")
        msg = ChatMessageOpenAI(role="user", content=(text_part, image_part))
        request = VLMRequestOpenAI(messages=(msg,))
        with pytest.raises(Exception):
            request.messages = (msg, msg)

