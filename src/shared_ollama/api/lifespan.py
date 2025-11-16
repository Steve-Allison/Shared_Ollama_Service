"""Application lifespan management.

Handles startup and shutdown of the Shared Ollama Service API, including
Ollama manager initialization, client setup, queue configuration, and
image processing infrastructure.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from shared_ollama.api.dependencies import set_dependencies
from shared_ollama.client import AsyncOllamaConfig, AsyncSharedOllamaClient
from shared_ollama.core.ollama_manager import initialize_ollama_manager
from shared_ollama.core.queue import RequestQueue
from shared_ollama.core.utils import get_project_root
from shared_ollama.infrastructure.adapters import (
    AsyncOllamaClientAdapter,
    MetricsCollectorAdapter,
    RequestLoggerAdapter,
)
from shared_ollama.infrastructure.image_cache import ImageCache
from shared_ollama.infrastructure.image_processing import ImageProcessor

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan_context(app: FastAPI):
    """Manage application lifespan.

    Handles:
    - Ollama service initialization and startup
    - Async client creation and connection verification
    - Infrastructure adapter setup (client, logger, metrics)
    - Request queue initialization (separate chat and VLM queues)
    - Image processing infrastructure (processor and cache)
    - Dependency injection setup

    Yields control to the application, then handles shutdown.

    Raises:
        RuntimeError: If Ollama service fails to start.
    """
    # Debug: Log that lifespan is starting
    print("LIFESPAN: Starting Shared Ollama Service API", flush=True)
    logger.info("LIFESPAN: Starting Shared Ollama Service API")

    # Initialize and start Ollama manager (manages Ollama process internally)
    logger.info("LIFESPAN: Initializing Ollama manager")
    try:
        project_root = get_project_root()
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        ollama_manager = initialize_ollama_manager(
            base_url="http://localhost:11434",
            log_dir=log_dir,
            auto_detect_optimizations=True,
        )

        logger.info("LIFESPAN: Starting Ollama service (managed internally)")
        ollama_started = await ollama_manager.start(wait_for_ready=True, max_wait_time=30)
        if not ollama_started:
            logger.error("LIFESPAN: Failed to start Ollama service")
            raise RuntimeError("Failed to start Ollama service. Check logs for details.")
        logger.info("LIFESPAN: Ollama service started successfully")
        print("LIFESPAN: Ollama service started", flush=True)
    except Exception as exc:
        logger.error("LIFESPAN: Failed to start Ollama service: %s", exc, exc_info=True)
        print(f"LIFESPAN ERROR: Failed to start Ollama: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        raise

    # Initialize async client (connects to the managed Ollama service)
    client: AsyncSharedOllamaClient | None = None
    try:
        config = AsyncOllamaConfig()
        logger.info("LIFESPAN: Creating AsyncSharedOllamaClient")
        # Don't verify on init - we'll do it manually to ensure it completes
        client = AsyncSharedOllamaClient(config=config, verify_on_init=False)
        logger.info("LIFESPAN: Client created, ensuring initialization")
        # Ensure client is initialized and verified (async)
        await client._ensure_client()
        logger.info("LIFESPAN: Client ensured, verifying connection")
        await client._verify_connection()
        logger.info("LIFESPAN: Ollama async client initialized successfully")
        print("LIFESPAN: Client initialized successfully", flush=True)
    except Exception as exc:
        logger.error("LIFESPAN: Failed to initialize Ollama async client: %s", exc, exc_info=True)
        print(f"LIFESPAN ERROR: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        client = None
        # Don't raise - allow server to start but client will be None
        # This way we can see the error in logs

    # Initialize infrastructure adapters
    if client:
        client_adapter = AsyncOllamaClientAdapter(client)
        logger_adapter = RequestLoggerAdapter()
        metrics_adapter = MetricsCollectorAdapter()
    else:
        client_adapter = None
        logger_adapter = None
        metrics_adapter = None

    # Initialize separate request queues for chat and VLM
    logger.info("LIFESPAN: Initializing separate request queues")
    chat_queue = RequestQueue(max_concurrent=6, max_queue_size=50, default_timeout=60.0)
    vlm_queue = RequestQueue(max_concurrent=3, max_queue_size=20, default_timeout=120.0)
    logger.info("LIFESPAN: Chat queue initialized (max_concurrent=6, max_queue_size=50)")
    logger.info("LIFESPAN: VLM queue initialized (max_concurrent=3, max_queue_size=20)")
    print("LIFESPAN: Separate queues initialized (chat + VLM)", flush=True)

    # Initialize image processing infrastructure
    logger.info("LIFESPAN: Initializing image processing infrastructure")
    image_processor = ImageProcessor(
        max_dimension=1024,
        jpeg_quality=85,
        png_compression=6,
        max_size_bytes=10 * 1024 * 1024,  # 10MB max
    )
    image_cache = ImageCache(max_size=100, ttl_seconds=3600.0)
    logger.info("LIFESPAN: Image processor and cache initialized")
    print("LIFESPAN: Image processing ready", flush=True)

    # Set dependencies for dependency injection
    if client_adapter and logger_adapter and metrics_adapter:
        set_dependencies(
            client_adapter,
            logger_adapter,
            metrics_adapter,
            chat_queue,
            vlm_queue,
            image_processor,
            image_cache,
        )
        logger.info("LIFESPAN: Dependencies initialized for dependency injection")

    yield

    # Shutdown
    logger.info("Shutting down Shared Ollama Service API")
    if client:
        try:
            # Close the async client properly (we're in an async context)
            await client.close()
        except Exception as exc:
            logger.warning("Error closing async client: %s", exc)

    # Stop Ollama service (managed internally)
    try:
        from shared_ollama.core.ollama_manager import get_ollama_manager

        logger.info("LIFESPAN: Stopping Ollama service")
        ollama_manager = get_ollama_manager()
        await ollama_manager.stop(timeout=10)
        logger.info("LIFESPAN: Ollama service stopped")
        print("LIFESPAN: Ollama service stopped", flush=True)
    except Exception as exc:
        logger.warning("Error stopping Ollama service: %s", exc)
