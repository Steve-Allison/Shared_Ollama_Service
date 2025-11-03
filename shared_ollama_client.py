"""
Unified Ollama Client Library
=============================

A common Ollama client that all projects can use to interact with the shared Ollama service.

Usage:
    from shared_ollama_client import SharedOllamaClient
    
    client = SharedOllamaClient()
    response = client.generate("Hello, world!")
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import requests
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model(str, Enum):
    """Available Ollama models."""
    LLAVA_13B = "llava:13b"       # Primary: 13B params, vision model
    QWEN25_14B = "qwen2.5:14b"    # Secondary: 14.8B params


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    base_url: str = "http://localhost:11434"
    default_model: str = Model.LLAVA_13B.value
    timeout: int = 60
    verbose: bool = False


@dataclass
class GenerateOptions:
    """Options for text generation."""
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None


@dataclass
class GenerateResponse:
    """Response from Ollama generate API."""
    text: str
    model: str
    context: Optional[List[int]] = None
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


class SharedOllamaClient:
    """
    Unified Ollama client for all projects.
    
    This client provides a common interface to the shared Ollama service
    running on port 11434. It supports all standard Ollama operations.
    
    Example:
        >>> client = SharedOllamaClient()
        >>> response = client.generate("Hello!")
        >>> print(response.text)
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        """
        Initialize the Ollama client.
        
        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        self.config = config or OllamaConfig()
        self.session = requests.Session()
        self._verify_connection()
    
    def _verify_connection(self) -> None:
        """Verify connection to Ollama service."""
        try:
            response = self.session.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            logger.info("Connected to Ollama service")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Make sure the service is running."
            )
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model information dictionaries
        """
        response = self.session.get(
            f"{self.config.base_url}/api/tags",
            timeout=self.config.timeout
        )
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        options: Optional[GenerateOptions] = None,
        stream: bool = False
    ) -> GenerateResponse:
        """
        Generate text using Ollama.
        
        Args:
            prompt: The prompt to generate from
            model: Model to use (uses default if not provided)
            system: Optional system prompt
            options: Generation options
            stream: Whether to stream the response
            
        Returns:
            GenerateResponse with generated text and metadata
            
        Example:
            >>> client = SharedOllamaClient()
            >>> response = client.generate(
            ...     "Why is the sky blue?",
            ...     model=Model.LLAVA_13B
            ... )
            >>> print(response.text)
        """
        model = model or self.config.default_model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        if system:
            payload["system"] = system
        
        if options:
            payload["options"] = {
                "temperature": options.temperature,
                "top_p": options.top_p,
                "top_k": options.top_k,
                "repeat_penalty": options.repeat_penalty,
            }
            
            if options.max_tokens:
                payload["options"]["num_predict"] = options.max_tokens
            
            if options.seed:
                payload["options"]["seed"] = options.seed
            
            if options.stop:
                payload["options"]["stop"] = options.stop
        
        logger.info(f"Generating with model {model}")
        
        response = self.session.post(
            f"{self.config.base_url}/api/generate",
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        
        return GenerateResponse(
            text=data.get("response", ""),
            model=data.get("model", model),
            context=data.get("context"),
            total_duration=data.get("total_duration", 0),
            load_duration=data.get("load_duration", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            prompt_eval_duration=data.get("prompt_eval_duration", 0),
            eval_count=data.get("eval_count", 0),
            eval_duration=data.get("eval_duration", 0)
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Chat with the model using chat format.
        
        Args:
            messages: List of message dicts with "role" and "content"
            model: Model to use (uses default if not provided)
            stream: Whether to stream the response
            
        Returns:
            Chat response dictionary
            
        Example:
            >>> messages = [
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> response = client.chat(messages)
            >>> print(response["message"]["content"])
        """
        model = model or self.config.default_model
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        response = self.session.post(
            f"{self.config.base_url}/api/chat",
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        return response.json()
    
    def pull_model(self, model: str) -> Dict[str, Any]:
        """
        Pull a model from Ollama.
        
        Args:
            model: Model name to pull
            
        Returns:
            Status dictionary
        """
        payload = {"name": model}
        
        logger.info(f"Pulling model {model}")
        
        response = self.session.post(
            f"{self.config.base_url}/api/pull",
            json=payload,
            timeout=300  # Pulling can take a while
        )
        
        return response.json()
    
    def health_check(self) -> bool:
        """
        Perform health check on Ollama service.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Model information dictionary or None if not found
        """
        models = self.list_models()
        for m in models:
            if m.get("name") == model:
                return m
        return None


# Convenience functions for easy usage
def create_client(base_url: str = "http://localhost:11434") -> SharedOllamaClient:
    """Create a shared Ollama client with default config."""
    return SharedOllamaClient(OllamaConfig(base_url=base_url))


def quick_generate(prompt: str, model: Optional[str] = None) -> str:
    """
    Quick generate function for simple use cases.
    
    Args:
        prompt: The prompt to generate from
        model: Model to use (optional)
        
    Returns:
        Generated text
        
    Example:
        >>> text = quick_generate("Hello!")
    """
    client = SharedOllamaClient()
    response = client.generate(prompt, model=model)
    return response.text


if __name__ == "__main__":
    # Example usage
    print("Shared Ollama Client Test")
    print("=" * 40)
    
    try:
        client = SharedOllamaClient()
        
        # List available models
        print("\nAvailable models:")
        models = client.list_models()
        for model in models:
            print(f"  - {model['name']}")
        
        # Test generation
        print("\nTesting generation...")
        response = quick_generate("Say hello in one sentence.")
        print(f"Response: {response}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
