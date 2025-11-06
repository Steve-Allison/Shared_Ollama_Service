"""Type stubs for shared_ollama_client module."""

from enum import StrEnum
from typing import Any

class Model(StrEnum):
    """Available Ollama models."""

    QWEN25_VL_7B: str
    QWEN25_7B: str
    QWEN25_14B: str
    GRANITE_4_H_TINY: str

class OllamaConfig:
    """Configuration for Ollama client."""

    base_url: str
    default_model: str
    timeout: int
    verbose: bool

    def __init__(
        self,
        base_url: str = ...,
        default_model: str = ...,
        timeout: int = ...,
        verbose: bool = ...,
    ) -> None: ...

class GenerateOptions:
    """Options for text generation."""

    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    max_tokens: int | None
    seed: int | None
    stop: list[str] | None

    def __init__(
        self,
        temperature: float = ...,
        top_p: float = ...,
        top_k: int = ...,
        repeat_penalty: float = ...,
        max_tokens: int | None = ...,
        seed: int | None = ...,
        stop: list[str] | None = ...,
    ) -> None: ...

class GenerateResponse:
    """Response from Ollama generate API."""

    text: str
    model: str
    context: list[int] | None
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int

    def __init__(
        self,
        text: str,
        model: str,
        context: list[int] | None = ...,
        total_duration: int = ...,
        load_duration: int = ...,
        prompt_eval_count: int = ...,
        prompt_eval_duration: int = ...,
        eval_count: int = ...,
        eval_duration: int = ...,
    ) -> None: ...

class SharedOllamaClient:
    """Unified Ollama client for all projects."""

    def __init__(
        self,
        config: OllamaConfig | None = ...,
        verify_on_init: bool = ...,
    ) -> None: ...
    def list_models(self) -> list[dict[str, Any]]: ...
    def generate(
        self,
        prompt: str,
        model: str | None = ...,
        system: str | None = ...,
        options: GenerateOptions | None = ...,
        stream: bool = ...,
    ) -> GenerateResponse: ...
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = ...,
        stream: bool = ...,
    ) -> dict[str, Any]: ...
    def pull_model(self, model: str) -> dict[str, Any]: ...
    def health_check(self) -> bool: ...
    def get_model_info(self, model: str) -> dict[str, Any] | None: ...

def create_client(base_url: str = ...) -> SharedOllamaClient: ...
def quick_generate(prompt: str, model: str | None = ...) -> str: ...
