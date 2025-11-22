import asyncio
import base64

from shared_ollama.application.interfaces import (
    AnalyticsCollectorInterface,
    ImageCacheInterface,
    ImageProcessorInterface,
    MetricsCollectorInterface,
    OllamaClientInterface,
    PerformanceCollectorInterface,
    RequestLoggerInterface,
)
from shared_ollama.application.vlm_use_cases import VLMUseCase
from shared_ollama.domain.entities import ModelName, VLMMessage, VLMRequest
from shared_ollama.infrastructure.config import settings
from shared_ollama.infrastructure.image_processing import ImageFormat, ImageMetadata


# Dummy implementations for interfaces for demonstration purposes
class DummyOllamaClient(OllamaClientInterface):
    async def chat(self, messages, model, options, stream, format, tools):
        # Simulate a response from Ollama
        if stream:
            async def streaming_response():
                yield {"chunk": "This is a streaming response part 1."}
                await asyncio.sleep(0.1)
                yield {"chunk": "This is a streaming response part 2.", "done": True, "model": "test-model"}
            return streaming_response()
        else:
            # Check if any message contains images
            contains_image_data = False
            for msg in messages:
                if msg.get("images"):
                    contains_image_data = True
                    break

            if contains_image_data:
                return {
                    "model": model,
                    "message": {"role": "assistant", "content": "The image appears to be a slide from a presentation titled 'Seismic shifts are already underway'. It shows four main sections: Discoverability, Content Velocity, Orchestration, and Engagement & Measurement. Each section contains various UI elements and text snippets suggesting different aspects of digital strategy, such as brand presence on AI engines, real estate search, e-commerce product details (ExplorerGo Tent), and activities in Bora Bora (The ultimate island guide). The bottom left corner features the Adobe logo, and the bottom right has a copyright notice for Adobe and page number 9."},
                    "load_duration": 1000000,
                    "total_duration": 5000000,
                }
            else:
                return {
                    "model": model,
                    "message": {"role": "assistant", "content": "I couldn't detect any images in the request."},
                    "load_duration": 1000000,
                    "total_duration": 5000000,
                }


    async def generate(self, prompt, model, system, options, stream, format, tools):
        pass
    async def list_models(self):
        pass
    async def generate_stream(self, prompt, model, system, options, format, tools):
        pass
    async def chat_stream(self, messages, model, options, images, format, tools):
        pass
    async def health_check(self):
        return True
    async def get_model_info(self, model):
        pass

class DummyRequestLogger(RequestLoggerInterface):
    def log_request(self, event):
        # print(f"Logged request: {event}")
        pass

class DummyMetricsCollector(MetricsCollectorInterface):
    def record_request(self, model, operation, latency_ms, success, error=None):
        # print(f"Recorded metric: {model}, {operation}, {latency_ms}")
        pass

class DummyImageProcessor(ImageProcessorInterface):
    def process_image(self, data_url, target_format):
        # Simply return the original base64 for demo
        _orig_format, image_bytes = self.validate_data_url(data_url)
        metadata = ImageMetadata(
            original_size=len(image_bytes), compressed_size=len(image_bytes),
            width=100, height=100, format=ImageFormat.JPEG, compression_ratio=1.0
        )
        # Corrected: Return the base64 encoded string of the original image bytes
        return base64.b64encode(image_bytes).decode("utf-8"), metadata
    def validate_data_url(self, data_url):
        # Basic validation for demo
        if not data_url.startswith("data:image/"):
            raise ValueError("Invalid data URL")
        _header, base64_data = data_url.split(";base64,")
        return "png", base64.b64decode(base64_data)

class DummyImageCache(ImageCacheInterface):
    def get(self, data_url, target_format):
        return None
    def put(self, data_url, target_format, base64_string, metadata):
        pass
    def clear(self):
        pass
    def get_stats(self):
        return {"hit_rate": 0.0}

class DummyAnalyticsCollector(AnalyticsCollectorInterface):
    def record_request_with_project(self, model, operation, latency_ms, success, project, error=None):
        pass

class DummyPerformanceCollector(PerformanceCollectorInterface):
    def record_performance(self, model, operation, total_latency_ms, success, response):
        pass

async def main():
    # Read the image file and convert it to base64
    image_path = "tests/_test_files/slide_9.png"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Initialize dummy dependencies
    dummy_client = DummyOllamaClient()
    dummy_logger = DummyRequestLogger()
    dummy_metrics = DummyMetricsCollector()
    dummy_image_processor = DummyImageProcessor()
    dummy_image_cache = DummyImageCache()
    dummy_analytics = DummyAnalyticsCollector()
    dummy_performance = DummyPerformanceCollector()

    # Initialize VLMUseCase with dummy dependencies
    vlm_use_case = VLMUseCase(
        client=dummy_client,
        logger=dummy_logger,
        metrics=dummy_metrics,
        image_processor=dummy_image_processor,
        image_cache=dummy_image_cache,
        analytics=dummy_analytics,
        performance=dummy_performance,
    )

    # Construct the VLM request
    message = VLMMessage(role="user", content="What's in this image?", images=(f"data:image/png;base64,{image_base64}",))
    
    # Use the default model from settings, or a dummy if settings are not loaded
    try:
        default_model = ModelName(value=settings.ollama.default_model)
    except Exception:
        default_model = ModelName(value="dummy-vlm-model")

    vlm_request = VLMRequest(
        messages=(message,),
        model=default_model,
        image_compression=True,
        max_dimension=1024,
    )

    # Call the VLM
    response = await vlm_use_case.execute(
        request=vlm_request,
        request_id="test-request-123",
        stream=False,
        target_format="jpeg", # Example target format
    )

    # Extract and print the response
    if isinstance(response, dict) and "message" in response:
        print(f"VLM Response: {response['message']['content']}")
    else:
        print(f"Unexpected VLM response format: {response}")

if __name__ == "__main__":
    asyncio.run(main())