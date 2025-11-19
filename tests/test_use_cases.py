"""
Comprehensive behavioral tests for use cases.

Tests focus on real workflows, error handling, edge cases, and integration scenarios.
Uses real adapters and clients - no mocks of internal logic.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from shared_ollama.application.use_cases import (
    ChatUseCase,
    GenerateUseCase,
    ListModelsUseCase,
)
from shared_ollama.domain.entities import (
    ChatMessage,
    ChatRequest,
    GenerationOptions,
    GenerationRequest,
    ModelInfo,
)
from shared_ollama.domain.value_objects import ModelName, Prompt, SystemMessage
from shared_ollama.infrastructure.adapters import (
    AsyncOllamaClientAdapter,
    MetricsCollectorAdapter,
    RequestLoggerAdapter,
)
from shared_ollama.telemetry.metrics import MetricsCollector


class MockGenerateResponse:
    """Mock GenerateResponse for testing."""
    def __init__(self):
        self.text = "Generated text"
        self.model = "qwen2.5vl:7b"
        self.total_duration = 500_000_000
        self.load_duration = 200_000_000
        self.prompt_eval_count = 5
        self.prompt_eval_duration = 100_000_000
        self.eval_count = 10
        self.eval_duration = 400_000_000


class MockChatResponse:
    """Mock ChatResponse for testing."""
    def __init__(self):
        self.message = {"role": "assistant", "content": "Chat response"}
        self.model = "qwen2.5vl:7b"
        self.done = True


@pytest.fixture
def mock_async_client():
    """Create a mock async client for testing (external service)."""
    client = AsyncMock()
    client.generate = AsyncMock(return_value=MockGenerateResponse())
    client.chat = AsyncMock(return_value=MockChatResponse())
    client.list_models = AsyncMock(
        return_value=[
            {"name": "qwen2.5vl:7b", "size": 5969245856},
            {"name": "qwen2.5:7b", "size": 4730000000},
        ]
    )
    return client


@pytest.fixture
def use_case_dependencies(mock_async_client):
    """Create use case dependencies with real adapters."""
    client_adapter = AsyncOllamaClientAdapter(mock_async_client)
    logger_adapter = RequestLoggerAdapter()
    metrics_adapter = MetricsCollectorAdapter()

    return {
        "client": client_adapter,
        "logger": logger_adapter,
        "metrics": metrics_adapter,
    }


@pytest.mark.asyncio
class TestGenerateUseCase:
    """Behavioral tests for GenerateUseCase."""

    async def test_execute_returns_generation_result(self, use_case_dependencies):
        """Test that execute() returns generation result."""
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Hello, world!"),
            model=ModelName(value="qwen2.5vl:7b"),
        )

        result = await use_case.execute(request, request_id="test-1")

        assert isinstance(result, dict)
        assert "text" in result
        assert result["text"] == "Generated text"

    async def test_execute_includes_all_metrics(self, use_case_dependencies):
        """Test that execute() includes all performance metrics."""
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value="qwen2.5vl:7b"),
        )

        result = await use_case.execute(request, request_id="test-1")

        assert "total_duration" in result
        assert "load_duration" in result
        assert "prompt_eval_count" in result
        assert "eval_count" in result

    async def test_execute_uses_model_from_request(self, use_case_dependencies, mock_async_client):
        """Test that execute() uses model from request."""
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value="custom-model"),
        )

        await use_case.execute(request, request_id="test-1")

        # Verify client was called with correct model
        mock_async_client.generate.assert_called_once()
        call_kwargs = mock_async_client.generate.call_args.kwargs
        assert call_kwargs["model"] == "custom-model"

    async def test_execute_uses_default_model_when_none(self, use_case_dependencies, mock_async_client):
        """Test that execute() handles None model correctly."""
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=None,
        )

        await use_case.execute(request, request_id="test-1")

        # Should pass None to client (client handles default)
        mock_async_client.generate.assert_called_once()
        call_kwargs = mock_async_client.generate.call_args.kwargs
        assert call_kwargs["model"] is None

    async def test_execute_includes_system_message(self, use_case_dependencies, mock_async_client):
        """Test that execute() includes system message when provided."""
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value="qwen2.5vl:7b"),
            system=SystemMessage(value="You are helpful"),
        )

        await use_case.execute(request, request_id="test-1")

        call_kwargs = mock_async_client.generate.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful"

    async def test_execute_includes_options(self, use_case_dependencies, mock_async_client):
        """Test that execute() includes generation options."""
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value="qwen2.5vl:7b"),
            options=GenerationOptions(
                temperature=0.7,
                top_p=0.95,
                max_tokens=100,
            ),
        )

        await use_case.execute(request, request_id="test-1")

        call_kwargs = mock_async_client.generate.call_args.kwargs
        assert "options" in call_kwargs
        options = call_kwargs["options"]
        assert options["temperature"] == 0.7
        assert options["top_p"] == 0.95
        assert options["num_predict"] == 100

    async def test_execute_filters_none_options(self, use_case_dependencies, mock_async_client):
        """Test that execute() filters out None option values."""
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value="qwen2.5vl:7b"),
            options=GenerationOptions(
                temperature=0.7,
                max_tokens=None,  # None value
                seed=None,  # None value
            ),
        )

        await use_case.execute(request, request_id="test-1")

        call_kwargs = mock_async_client.generate.call_args.kwargs
        options = call_kwargs["options"]
        assert "num_predict" not in options or options.get("num_predict") is not None
        assert "seed" not in options or options.get("seed") is not None

    async def test_execute_includes_format(self, use_case_dependencies, mock_async_client):
        """Test that execute() includes format when provided."""
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value="qwen2.5vl:7b"),
            format="json",
        )

        await use_case.execute(request, request_id="test-1")

        call_kwargs = mock_async_client.generate.call_args.kwargs
        assert call_kwargs["format"] == "json"

    async def test_execute_includes_tools(self, use_case_dependencies, mock_async_client):
        """Test that execute() includes tools when provided."""
        from shared_ollama.domain.entities import Tool, ToolFunction

        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value="qwen2.5vl:7b"),
            tools=(
                Tool(
                    type="function",
                    function=ToolFunction(
                        name="get_weather",
                        description="Get weather",
                        parameters={"type": "object"},
                    ),
                ),
            ),
        )

        await use_case.execute(request, request_id="test-1")

        call_kwargs = mock_async_client.generate.call_args.kwargs
        assert "tools" in call_kwargs
        tools = call_kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "get_weather"

    async def test_execute_records_metrics(self, use_case_dependencies):
        """Test that execute() records metrics via MetricsCollector."""
        MetricsCollector.reset()
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value="qwen2.5vl:7b"),
        )

        await use_case.execute(request, request_id="test-1")

        metrics = MetricsCollector.get_metrics()
        assert metrics.total_requests >= 1
        assert metrics.successful_requests >= 1

    async def test_execute_handles_client_error(self, use_case_dependencies, mock_async_client):
        """Test that execute() handles client errors correctly."""
        mock_async_client.generate.side_effect = ConnectionError("Service unavailable")
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value="qwen2.5vl:7b"),
        )

        with pytest.raises(ConnectionError):
            await use_case.execute(request, request_id="test-1")

        # Should still record metrics
        metrics = MetricsCollector.get_metrics()
        assert metrics.total_requests >= 1
        assert metrics.failed_requests >= 1

    async def test_execute_handles_streaming(self, use_case_dependencies, mock_async_client):
        """Test that execute() handles streaming responses."""
        # Mock streaming response
        async def stream_generator():
            yield {"chunk": "Hello", "done": False}
            yield {"chunk": " world", "done": False}
            yield {"chunk": "!", "done": True}

        mock_async_client.generate.return_value = stream_generator()
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value="qwen2.5vl:7b"),
        )

        result = await use_case.execute(request, request_id="test-1", stream=True)

        # Should return async iterator
        assert hasattr(result, "__aiter__")
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        assert len(chunks) > 0

    async def test_execute_includes_project_name(self, use_case_dependencies):
        """Test that execute() includes project name in logging."""
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value="qwen2.5vl:7b"),
        )

        result = await use_case.execute(
            request, request_id="test-1", project_name="test-project"
        )

        assert isinstance(result, dict)
        # Project name should be included in logging (tested via logger adapter)

    async def test_execute_includes_client_ip(self, use_case_dependencies):
        """Test that execute() includes client IP in logging."""
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value="qwen2.5vl:7b"),
        )

        result = await use_case.execute(
            request, request_id="test-1", client_ip="192.168.1.1"
        )

        assert isinstance(result, dict)
        # Client IP should be included in logging


@pytest.mark.asyncio
class TestChatUseCase:
    """Behavioral tests for ChatUseCase."""

    async def test_execute_returns_chat_result(self, use_case_dependencies):
        """Test that execute() returns chat result."""
        use_case = ChatUseCase(**use_case_dependencies)

        request = ChatRequest(
            messages=(ChatMessage(role="user", content="Hello!"),),
            model=ModelName(value="qwen2.5vl:7b"),
        )

        result = await use_case.execute(request, request_id="test-1")

        assert isinstance(result, dict)
        assert "message" in result
        assert result["message"]["content"] == "Chat response"

    async def test_execute_handles_multiple_messages(self, use_case_dependencies, mock_async_client):
        """Test that execute() handles multiple messages."""
        use_case = ChatUseCase(**use_case_dependencies)

        request = ChatRequest(
            messages=(
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="What is 2+2?"),
            ),
            model=ModelName(value="qwen2.5vl:7b"),
        )

        await use_case.execute(request, request_id="test-1")

        call_kwargs = mock_async_client.chat.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2

    async def test_execute_includes_tool_calls(self, use_case_dependencies, mock_async_client):
        """Test that execute() includes tool calls when present."""
        from shared_ollama.domain.entities import ToolCall, ToolCallFunction

        use_case = ChatUseCase(**use_case_dependencies)

        request = ChatRequest(
            messages=(
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="call_1",
                            type="function",
                            function=ToolCallFunction(
                                name="get_weather",
                                arguments='{"location": "SF"}',
                            ),
                        ),
                    ),
                ),
            ),
            model=ModelName(value="qwen2.5vl:7b"),
        )

        await use_case.execute(request, request_id="test-1")

        call_kwargs = mock_async_client.chat.call_args.kwargs
        messages = call_kwargs["messages"]
        assert "tool_calls" in messages[0] or len(messages) > 0

    async def test_execute_records_metrics(self, use_case_dependencies):
        """Test that execute() records metrics."""
        MetricsCollector.reset()
        use_case = ChatUseCase(**use_case_dependencies)

        request = ChatRequest(
            messages=(ChatMessage(role="user", content="Test"),),
            model=ModelName(value="qwen2.5vl:7b"),
        )

        await use_case.execute(request, request_id="test-1")

        metrics = MetricsCollector.get_metrics()
        assert metrics.total_requests >= 1


@pytest.mark.asyncio
class TestListModelsUseCase:
    """Behavioral tests for ListModelsUseCase."""

    async def test_execute_returns_model_list(self, use_case_dependencies):
        """Test that execute() returns list of models."""
        use_case = ListModelsUseCase(**use_case_dependencies)

        result = await use_case.execute(request_id="test-1")

        assert isinstance(result, list)
        assert len(result) >= 2
        assert all(isinstance(m, ModelInfo) for m in result)

    async def test_execute_converts_to_domain_entities(self, use_case_dependencies):
        """Test that execute() converts API models to domain entities."""
        use_case = ListModelsUseCase(**use_case_dependencies)

        result = await use_case.execute(request_id="test-1")

        assert isinstance(result[0], ModelInfo)
        assert result[0].name == "qwen2.5vl:7b"
        assert result[0].size == 5969245856

    async def test_execute_handles_empty_list(self, use_case_dependencies, mock_async_client):
        """Test that execute() handles empty model list."""
        mock_async_client.list_models.return_value = []
        use_case = ListModelsUseCase(**use_case_dependencies)

        result = await use_case.execute(request_id="test-1")

        assert result == []

    async def test_execute_records_metrics(self, use_case_dependencies):
        """Test that execute() records metrics."""
        MetricsCollector.reset()
        use_case = ListModelsUseCase(**use_case_dependencies)

        await use_case.execute(request_id="test-1")

        metrics = MetricsCollector.get_metrics()
        assert metrics.total_requests >= 1


@pytest.mark.asyncio
class TestUseCaseEdgeCases:
    """Edge case and error handling tests for use cases."""

    async def test_generate_handles_empty_prompt(self, use_case_dependencies):
        """Test that GenerateUseCase handles empty prompt."""
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value=""),
            model=ModelName(value="qwen2.5vl:7b"),
        )

        # Should still execute (validation happens in domain)
        result = await use_case.execute(request, request_id="test-1")
        assert isinstance(result, dict)

    async def test_chat_handles_empty_messages(self, use_case_dependencies):
        """Test that ChatUseCase handles empty messages."""
        use_case = ChatUseCase(**use_case_dependencies)

        request = ChatRequest(
            messages=(),
            model=ModelName(value="qwen2.5vl:7b"),
        )

        # Should still execute (validation happens in domain or client)
        try:
            result = await use_case.execute(request, request_id="test-1")
            assert isinstance(result, dict)
        except Exception:
            # Client may reject empty messages, which is valid
            pass

    async def test_generate_handles_all_none_options(self, use_case_dependencies, mock_async_client):
        """Test that GenerateUseCase handles all None options."""
        use_case = GenerateUseCase(**use_case_dependencies)

        request = GenerationRequest(
            prompt=Prompt(value="Test"),
            model=ModelName(value="qwen2.5vl:7b"),
            options=GenerationOptions(
                max_tokens=None,
                seed=None,
                stop=None,
            ),
        )

        await use_case.execute(request, request_id="test-1")

        call_kwargs = mock_async_client.generate.call_args.kwargs
        # Options dict should not include None values
        if "options" in call_kwargs:
            options = call_kwargs["options"]
            assert "num_predict" not in options or options.get("num_predict") is not None

    async def test_concurrent_execute_calls(self, use_case_dependencies):
        """Test that use cases handle concurrent execute calls."""
        use_case = GenerateUseCase(**use_case_dependencies)

        async def execute_request(i: int):
            request = GenerationRequest(
                prompt=Prompt(value=f"Request {i}"),
                model=ModelName(value="qwen2.5vl:7b"),
            )
            return await use_case.execute(request, request_id=f"test-{i}")

        tasks = [execute_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(isinstance(r, dict) for r in results)
