"""Application lifespan management.

This module handles startup and shutdown of the Shared Ollama Service API,
including Ollama manager initialization, client setup, queue configuration,
and image processing infrastructure.

Lifespan Responsibilities:
    - Startup:
        1. Load and validate model configuration
        2. Initialize and start Ollama manager (manages Ollama process)
        3. Create and verify async Ollama client connection
        4. Initialize infrastructure adapters (client, logger, metrics)
        5. Create request queues (chat and VLM, separate concurrency limits)
        6. Initialize image processing infrastructure (processor and cache)
        7. Set up dependency injection (store adapters for FastAPI Depends)
        8. Pre-warm models (optional, for faster first requests)
    - Runtime:
        - Application serves requests using initialized dependencies
    - Shutdown:
        1. Stop Ollama manager (shuts down managed Ollama process)
        2. Clean up client connections
        3. Close queues and cleanup resources

Error Handling:
    - Startup failures are logged but don't prevent server from starting
    - Ollama startup failures allow server to start (will retry on first request)
    - Client initialization failures are logged with warnings
    - All errors include detailed logging for debugging

Design Principles:
    - Graceful Degradation: Server can start even if some components fail
    - Retry Logic: Failed components can be retried on first use
    - Comprehensive Logging: All startup steps are logged for observability
    - Resource Cleanup: Proper shutdown ensures no resource leaks
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from shared_ollama.api.dependencies import set_dependencies
from shared_ollama.client import AsyncOllamaConfig, AsyncSharedOllamaClient
from shared_ollama.core.ollama_manager import initialize_ollama_manager
from shared_ollama.core.queue import RequestQueue
from shared_ollama.core.utils import (
    get_default_text_model,
    get_default_vlm_model,
    get_project_root,
)
from shared_ollama.infrastructure.adapters import (
    AnalyticsCollectorAdapter,
    AsyncOllamaClientAdapter,
    ImageCacheAdapter,
    ImageProcessorAdapter,
    MetricsCollectorAdapter,
    PerformanceCollectorAdapter,
    RequestLoggerAdapter,
)
from shared_ollama.infrastructure.config import settings
from shared_ollama.infrastructure.image_cache import ImageCache
from shared_ollama.infrastructure.image_processing import ImageProcessor

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan_context(app: FastAPI):
    """Manage application lifespan (startup and shutdown).

    Async context manager that handles complete application lifecycle:
    startup initialization, runtime operation, and shutdown cleanup.

    Startup Sequence:
        1. Load model configuration (defaults from config files)
        2. Initialize Ollama manager and start Ollama service
        3. Create async Ollama client and verify connection
        4. Initialize infrastructure adapters (client, logger, metrics, analytics, performance)
        5. Create request queues (chat and VLM with separate concurrency limits)
        6. Initialize image processing (processor and cache with config-based settings)
        7. Set up dependency injection (store adapters for FastAPI Depends)
        8. Pre-warm models (optional, for faster first requests)

    Shutdown Sequence:
        1. Stop Ollama manager (shuts down managed Ollama process)
        2. Clean up client connections
        3. Close queues and cleanup resources

    Args:
        app: FastAPI application instance. Used for storing state if needed.

    Yields:
        None. Control is yielded to the application for request handling.

    Raises:
        RuntimeError: If critical startup failures occur (rare, most errors
            are logged but don't prevent startup).

    Note:
        Startup failures are logged but generally don't prevent the server
        from starting. Ollama startup failures allow the server to start
        and will retry on the first request. All errors include detailed
        logging for debugging.
    """
    logger.info("LIFESPAN: Starting Shared Ollama Service API")

    # Load and validate model configuration (config-driven defaults)
    try:
        default_vlm = get_default_vlm_model()
        default_text = get_default_text_model()
        logger.info(
            "LIFESPAN: Model configuration loaded - VLM: %s, Text: %s",
            default_vlm,
            default_text,
        )
    except Exception as exc:
        logger.warning("LIFESPAN: Failed to load model configuration: %s", exc, exc_info=True)
        # Don't fail startup - will use fallback defaults

    # Initialize and start Ollama manager (manages Ollama process internally)
    logger.info("LIFESPAN: Initializing Ollama manager")
    try:
        project_root = get_project_root()
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        ollama_manager = initialize_ollama_manager(
            base_url=settings.ollama.url,
            log_dir=log_dir,
            auto_detect_optimizations=settings.ollama_manager.auto_detect_optimizations,
            force_manage=settings.ollama_manager.force_manage,
        )

        logger.info("LIFESPAN: Starting Ollama service (managed internally)")
        ollama_started = await ollama_manager.start(
            wait_for_ready=settings.ollama_manager.wait_for_ready,
            max_wait_time=settings.ollama_manager.max_wait_time,
        )
        if not ollama_started:
            logger.error("LIFESPAN: Failed to start Ollama service")
            # Don't raise - log the error and allow server to start
            # The server can still run even if Ollama fails to start initially
            # (it will be retried on first request or can be started manually)
            logger.warning(
                "LIFESPAN: Continuing despite Ollama startup failure - will retry on first request"
            )
        else:
            logger.info("LIFESPAN: Ollama service started successfully")

            # Proactively warm up models
            logger.info("LIFESPAN: Initiating model pre-warming")
            await ollama_manager.warmup_models()
    except Exception as exc:
        logger.exception("LIFESPAN: Failed to start Ollama service: %s", exc)
        # Don't raise - log the error and allow server to start
        # The server can still run even if Ollama fails to start initially
        # (it will be retried on first request or can be started manually)
        logger.warning(
            "LIFESPAN: Continuing despite Ollama startup failure - will retry on first request"
        )

    # Initialize async client (connects to the managed Ollama service)
    client: AsyncSharedOllamaClient | None = None
    try:
        # type: ignore needed because pyright doesn't recognize dataclass field defaults
        config = AsyncOllamaConfig(  # type: ignore[call-arg]
            base_url=settings.ollama.url,
            timeout=settings.client.timeout,
            health_check_timeout=settings.client.health_check_timeout,
            max_connections=settings.client.max_connections,
            max_keepalive_connections=settings.client.max_keepalive_connections,
            max_concurrent_requests=settings.client.max_concurrent_requests,
            max_retries=settings.client.max_retries,
            retry_delay=settings.client.retry_delay,
            verbose=settings.client.verbose,
        )
        logger.info("LIFESPAN: Creating AsyncSharedOllamaClient")
        # Don't verify on init - we'll do it manually to ensure it completes
        client = AsyncSharedOllamaClient(config=config, verify_on_init=False)
        logger.info("LIFESPAN: Client created, ensuring initialization")
        await client.initialize()
        logger.info("LIFESPAN: Ollama async client initialized successfully")
    except Exception as exc:
        logger.exception("LIFESPAN: Failed to initialize Ollama async client: %s", exc)
        client = None
        # Don't raise - allow server to start but client will be None
        # This way we can see the error in logs
        # The client will be retried on first request
        logger.warning(
            "LIFESPAN: Continuing despite client initialization failure - will retry on first request"
        )

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
    chat_queue = RequestQueue(
        max_concurrent=settings.queue.chat_max_concurrent,
        max_queue_size=settings.queue.chat_max_queue_size,
        default_timeout=settings.queue.chat_default_timeout,
    )
    vlm_queue = RequestQueue(
        max_concurrent=settings.queue.vlm_max_concurrent,
        max_queue_size=settings.queue.vlm_max_queue_size,
        default_timeout=settings.queue.vlm_default_timeout,
    )
    logger.info(
        "LIFESPAN: Chat queue initialized (max_concurrent=%d, max_queue_size=%d)",
        settings.queue.chat_max_concurrent,
        settings.queue.chat_max_queue_size,
    )
    logger.info(
        "LIFESPAN: VLM queue initialized (max_concurrent=%d, max_queue_size=%d)",
        settings.queue.vlm_max_concurrent,
        settings.queue.vlm_max_queue_size,
    )

    # Initialize image processing infrastructure
    logger.info("LIFESPAN: Initializing image processing infrastructure")
    image_processor = ImageProcessor(
        max_dimension=settings.image.max_dimension,
        jpeg_quality=settings.image.jpeg_quality,
        png_compression=settings.image.png_compression,
        max_size_bytes=settings.image.max_size_bytes,
    )
    image_cache = ImageCache(
        max_size=settings.image_cache.max_size,
        ttl_seconds=settings.image_cache.ttl_seconds,
    )
    logger.info("LIFESPAN: Image processor and cache initialized")

    # Create adapters for image processing
    image_processor_adapter = ImageProcessorAdapter(image_processor)
    image_cache_adapter = ImageCacheAdapter(image_cache)

    # Create optional analytics and performance adapters
    analytics_adapter = AnalyticsCollectorAdapter()
    performance_adapter = PerformanceCollectorAdapter()

    # Set dependencies for dependency injection
    if client_adapter and logger_adapter and metrics_adapter:
        set_dependencies(
            client_adapter,
            logger_adapter,
            metrics_adapter,
            chat_queue,
            vlm_queue,
            image_processor_adapter,
            image_cache_adapter,
            analytics_adapter=analytics_adapter,
            performance_adapter=performance_adapter,
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
        await ollama_manager.stop(timeout=settings.ollama_manager.shutdown_timeout)
        logger.info("LIFESPAN: Ollama service stopped")
    except Exception as exc:
        logger.warning("Error stopping Ollama service: %s", exc)
