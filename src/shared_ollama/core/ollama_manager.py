"""Ollama Process Manager.

Manages the Ollama service process lifecycle internally within the REST API.
Handles starting, stopping, and health checking of the Ollama backend.

This module provides process management for the Ollama service, including:
    - Automatic system optimization detection (Metal GPU, CPU cores, memory)
    - Graceful process lifecycle management with timeouts
    - Health checking and readiness verification
    - Log file management for stdout/stderr

Key behaviors:
    - Auto-detects system capabilities and applies optimizations
    - Uses subprocess management with proper signal handling
    - Implements async patterns for non-blocking operations
    - Thread-safe status checking via HTTP health endpoints
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import platform
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias

import psutil

logger = logging.getLogger(__name__)

# Type aliases for better type safety
ProcessEnv: TypeAlias = dict[str, str]
OptimizationConfig: TypeAlias = dict[str, str]


class OllamaManager:
    """Manages the Ollama service process lifecycle.

    This class handles the complete lifecycle of the Ollama subprocess, including
    starting, stopping, and monitoring. It automatically detects system capabilities
    and applies optimizations for better performance.

    Attributes:
        base_url: Base URL for Ollama service (default: "http://localhost:11434").
        process: Managed asyncio.subprocess.Process instance, or None if not started.
        log_dir: Directory path for Ollama log files.
        auto_detect_optimizations: Whether to automatically detect and apply
            system-specific optimizations.

    Thread safety:
        Methods are not thread-safe. Use from a single async context or
        protect with locks if accessing from multiple threads.

    Lifecycle:
        - Initialize with __init__()
        - Start process with start()
        - Check status with is_running() or get_status()
        - Stop process with stop()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        log_dir: Path | None = None,
        auto_detect_optimizations: bool = True,
        force_manage: bool = True,
    ) -> None:
        """Initialize the Ollama manager.

        Args:
            base_url: Base URL for Ollama service. Trailing slashes are removed.
            log_dir: Directory for Ollama logs. If None, defaults to project logs/
                directory.
            auto_detect_optimizations: If True, automatically detect and apply
                system-specific optimizations (GPU, CPU cores, memory limits).
            force_manage: If True, stop external Ollama instances (Homebrew/launchd)
                before starting our managed instance. If False, skip starting if
                Ollama is already running.
        """
        self.base_url = base_url.rstrip("/")
        self.process: asyncio.subprocess.Process | None = None
        self.log_dir = log_dir or self._get_default_log_dir()
        self.auto_detect_optimizations = auto_detect_optimizations
        self.force_manage = force_manage

    def _get_default_log_dir(self) -> Path:
        """Get default log directory path.

        Creates the logs directory if it doesn't exist.

        Returns:
            Path to logs directory.
        """
        from shared_ollama.core.utils import get_project_root

        log_dir = get_project_root() / "logs"
        log_dir.mkdir(exist_ok=True)
        return log_dir

    @functools.cached_property
    def ollama_executable(self) -> str | None:
        """Find and cache the Ollama executable path.

        Uses functools.cached_property for automatic caching (Python 3.8+).
        Result is cached since the executable location doesn't change at runtime.

        Returns:
            Absolute path to ollama executable, or None if not found in PATH.
        """
        ollama_path = shutil.which("ollama")
        if ollama_path:
            return ollama_path

        logger.warning("Ollama executable not found in PATH")
        return None

    async def _detect_system_optimizations(self) -> OptimizationConfig:
        """Detect system-specific optimizations for Ollama.

        Analyzes the system architecture and platform to determine optimal
        environment variables for Ollama performance. Detects:
            - Apple Silicon (Metal GPU acceleration)
            - CPU core count
            - Memory limits (via helper script if available)

        Returns:
            Dictionary of environment variable names to values. Always includes
            OLLAMA_HOST and OLLAMA_KEEP_ALIVE defaults.

        Side effects:
            May execute async subprocess to calculate memory limits if helper script
            exists. Logs debug messages for system detection.
        """
        optimizations: OptimizationConfig = {}
        arch = platform.machine()
        system = platform.system()

        match (system, arch):
            case ("Darwin", "arm64"):  # Apple Silicon
                optimizations["OLLAMA_METAL"] = "1"
                optimizations["OLLAMA_NUM_GPU"] = "-1"  # Use all GPU cores

                cpu_cores = os.cpu_count() or 10
                optimizations["OLLAMA_NUM_THREAD"] = str(cpu_cores)

            case ("Darwin", _):  # Intel Mac
                optimizations["OLLAMA_METAL"] = "0"
                optimizations["OLLAMA_NUM_GPU"] = "0"

            case _:
                logger.debug("System %s/%s - using default optimizations", system, arch)

        defaults: OptimizationConfig = {
            "OLLAMA_HOST": "0.0.0.0:11434",
            "OLLAMA_KEEP_ALIVE": "30m",
        }
        optimizations = defaults | optimizations

        # Auto-calculate memory limit if script exists (async subprocess)
        try:
            from shared_ollama.core.utils import get_project_root

            calc_script = get_project_root() / "scripts" / "calculate_memory_limit.sh"
            if calc_script.exists():
                process = await asyncio.create_subprocess_exec(
                    "bash",
                    str(calc_script),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5.0)
                    output = stdout.decode("utf-8") if stdout else ""
                    for line in output.splitlines():
                        if line.startswith("OLLAMA_MAX_RAM="):
                            optimizations["OLLAMA_MAX_RAM"] = line.split("=", 1)[1].strip()
                            break
                except asyncio.TimeoutError:
                    logger.debug("Memory limit calculation timed out")
                    process.kill()
                    await process.wait()
        except Exception as exc:
            logger.debug("Could not auto-calculate memory limit: %s", exc)

        return optimizations

    async def _stop_external_ollama(self) -> None:
        """Stop external Ollama instances (Homebrew/launchd).

        Attempts to stop Ollama services managed by external systems using
        async subprocess for non-blocking operations:
        - Homebrew services (brew services stop ollama)
        - Launchd services (launchctl unload)
        - Direct processes (kill by PID using psutil)

        Side effects:
            - Stops Homebrew service if running
            - Unloads launchd service if found
            - Kills remaining ollama serve processes
            - Waits briefly for processes to terminate
        """
        logger.info("Stopping external Ollama instances...")

        # Stop Homebrew service (async subprocess)
        if shutil.which("brew"):
            try:
                process = await asyncio.create_subprocess_exec(
                    "brew",
                    "services",
                    "list",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5.0)
                    output = stdout.decode("utf-8") if stdout else ""
                    if "ollama" in output and "started" in output:
                        logger.info("Stopping Homebrew Ollama service...")
                        stop_process = await asyncio.create_subprocess_exec(
                            "brew",
                            "services",
                            "stop",
                            "ollama",
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                        )
                        await asyncio.wait_for(stop_process.wait(), timeout=10.0)
                        await asyncio.sleep(2)  # Wait for service to stop
                except asyncio.TimeoutError:
                    logger.debug("Homebrew service check timed out")
                    process.kill()
                    await process.wait()
            except Exception as exc:
                logger.debug("Could not stop Homebrew service: %s", exc)

        # Stop launchd service (async subprocess)
        launchd_plist = Path.home() / "Library/LaunchAgents/com.ollama.service.plist"
        if launchd_plist.exists():
            try:
                logger.info("Unloading launchd Ollama service...")
                process = await asyncio.create_subprocess_exec(
                    "launchctl",
                    "unload",
                    str(launchd_plist),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(process.wait(), timeout=10.0)
                await asyncio.sleep(1)
            except Exception as exc:
                logger.debug("Could not unload launchd service: %s", exc)

        # Kill any remaining ollama serve processes using psutil (more reliable)
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = proc.info.get("cmdline", [])
                    if cmdline and "ollama" in cmdline and "serve" in cmdline:
                        logger.info("Killing Ollama process PID %s", proc.info["pid"])
                        proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            await asyncio.sleep(2)  # Wait for processes to terminate
        except Exception as exc:
            logger.debug("Could not find/kill Ollama processes: %s", exc)

        # Verify Ollama is stopped
        match self._check_ollama_running(timeout=1):
            case True:
                logger.warning("External Ollama instance may still be running")
            case False:
                logger.info("External Ollama instances stopped successfully")

    def _check_ollama_running(self, timeout: int = 2) -> bool:
        """Check if Ollama service is currently running.

        Performs a lightweight HTTP health check to determine if the service
        is responding.

        This method delegates to the infrastructure layer for HTTP operations,
        keeping the core module framework-agnostic.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            True if service responds with HTTP 200, False otherwise.

        Side effects:
            Makes an HTTP GET request to /api/tags endpoint (via infrastructure layer).
        """
        from shared_ollama.infrastructure.health_checker import check_ollama_health_simple

        return check_ollama_health_simple(base_url=self.base_url, timeout=timeout)

    async def start(
        self,
        wait_for_ready: bool = True,
        max_wait_time: int = 30,
    ) -> bool:
        """Start the Ollama service process.

        Starts the Ollama subprocess with optimized environment variables.
        Optionally waits for the service to become ready before returning.

        Args:
            wait_for_ready: If True, wait for service to respond to health checks
                before returning. If False, return immediately after starting process.
            max_wait_time: Maximum seconds to wait for service readiness.
                Only used if wait_for_ready is True.

        Returns:
            True if process started successfully (and is ready if wait_for_ready=True),
            False otherwise.

        Side effects:
            - Creates subprocess running 'ollama serve'
            - Creates log files in log_dir (ollama.log, ollama.error.log)
            - Updates environment variables for optimization
            - May stop process if readiness check fails

        Raises:
            No exceptions raised, but logs errors and returns False on failure.
        """
        # Check if Ollama is already running
        if self._check_ollama_running():
            if self.force_manage:
                logger.info(
                    "Ollama service is already running externally. Stopping external instance to manage our own..."
                )
                await self._stop_external_ollama()
            else:
                logger.info("Ollama service is already running at %s", self.base_url)
                return True

        ollama_path = self.ollama_executable
        if not ollama_path:
            logger.error("Ollama executable not found. Please install Ollama.")
            return False

        # Prepare environment with optimizations
        env: ProcessEnv = os.environ.copy()
        if self.auto_detect_optimizations:
            optimizations = await self._detect_system_optimizations()
            env.update(optimizations)
            logger.info("Applied system optimizations: %s", optimizations)

        log_file = self.log_dir / "ollama.log"
        error_log_file = self.log_dir / "ollama.error.log"

        try:
            logger.info("Starting Ollama service process...")
            # Open log files for async subprocess (keep open for subprocess lifetime)
            # Note: Files will be closed when process terminates
            log_f = log_file.open("a")
            err_f = error_log_file.open("a")
            
            # Use asyncio.create_subprocess_exec for non-blocking async subprocess
            # Note: File handles are passed to subprocess and will remain open
            self.process = await asyncio.create_subprocess_exec(
                ollama_path,
                "serve",
                stdout=log_f,
                stderr=err_f,
                env=env,
                start_new_session=True,  # Create new process group
            )

            logger.info("Ollama process started with PID %s", self.process.pid)

            if wait_for_ready:
                ready = await self._wait_for_ready(max_wait_time)
                if not ready:
                    logger.error(
                        "Ollama service did not become ready within %s seconds",
                        max_wait_time,
                    )
                    await self.stop()
                    return False

            return True

        except Exception as exc:
            logger.exception("Failed to start Ollama process: %s", exc)
            self.process = None
            return False

    async def _wait_for_ready(self, max_wait_time: int = 30) -> bool:
        """Wait for Ollama service to become ready.

        Polls the health endpoint at 1-second intervals until the service
        responds or timeout is reached.

        Args:
            max_wait_time: Maximum seconds to wait.

        Returns:
            True if service becomes ready within timeout, False otherwise.

        Side effects:
            Makes periodic HTTP requests to health endpoint.
        """
        start_time = time.monotonic()
        check_interval = 1.0

        while (elapsed := time.monotonic() - start_time) < max_wait_time:
            if self._check_ollama_running(timeout=1):
                logger.info("Ollama service is ready (took %.1fs)", elapsed)
                return True
            await asyncio.sleep(check_interval)

        return False

    async def stop(self, timeout: int = 10) -> bool:
        """Stop the Ollama service process.

        Attempts graceful shutdown (SIGTERM) first, then force kills (SIGKILL)
        if graceful shutdown times out.

        Args:
            timeout: Maximum seconds to wait for graceful shutdown before
                force killing.

        Returns:
            True if process stopped successfully, False if process was not
            managed by this instance.

        Side effects:
            - Sends SIGTERM to process
            - May send SIGKILL if graceful shutdown fails
            - Sets self.process to None after stopping
        """
        match self.process:
            case None:
                match self._check_ollama_running():
                    case False:
                        logger.info("Ollama service is not running")
                        return True
                    case True:
                        logger.warning(
                            "Ollama process not managed by this manager, but service is running"
                        )
                        return False
            case process:
                try:
                    logger.info("Stopping Ollama service process (PID %s)...", process.pid)

                    process.terminate()

                    try:
                        # Use asyncio.wait_for for timeout with async wait
                        await asyncio.wait_for(process.wait(), timeout=timeout)
                        logger.info("Ollama process stopped gracefully")
                        self.process = None
                        return True
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Ollama process did not stop gracefully, forcing termination..."
                        )
                        process.kill()
                        await process.wait()
                        logger.info("Ollama process force-stopped")
                        self.process = None
                        return True

                except Exception as exc:
                    logger.exception("Error stopping Ollama process: %s", exc)
                    if self.process:
                        try:
                            self.process.kill()
                            await self.process.wait()
                        except Exception:
                            pass
                        finally:
                            self.process = None
                    return False

    def is_running(self) -> bool:
        """Check if Ollama service is currently running.

        Checks both the managed process state and the service health endpoint.
        Updates self.process to None if the managed process has terminated.
        Uses psutil for reliable cross-platform process checking.

        Returns:
            True if service is running and healthy, False otherwise.

        Side effects:
            - May update self.process to None if process terminated
            - Makes HTTP request to health endpoint if process check is inconclusive
        """
        match self.process:
            case None:
                pass
            case process:
                # Use psutil for reliable process checking
                try:
                    if process.returncode is not None:
                        # Process has terminated
                        self.process = None
                        return False
                    # Check if process is actually running using psutil
                    if psutil.pid_exists(process.pid):
                        proc = psutil.Process(process.pid)
                        if proc.is_running():
                            return True
                        else:
                            # Process exists but not running
                            self.process = None
                            return False
                    else:
                        # PID doesn't exist
                        self.process = None
                        return False
                except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                    logger.debug("Process check failed: %s", exc)
                    self.process = None
                    return False

        return self._check_ollama_running()

    def get_status(self) -> dict[str, Any]:
        """Get current status of Ollama service.

        Returns:
            Dictionary with keys:
                - running: bool - Whether service is running
                - base_url: str - Service base URL
                - managed: bool - Whether process is managed by this instance
                - pid: int (optional) - Process ID if managed
                - returncode: int | None (optional) - Process return code if managed
        """
        status: dict[str, Any] = {
            "running": self.is_running(),
            "base_url": self.base_url,
            "managed": self.process is not None,
        }

        if self.process:
            status["pid"] = self.process.pid
            status["returncode"] = self.process.returncode
            # Add process info using psutil if available
            try:
                if psutil.pid_exists(self.process.pid):
                    proc = psutil.Process(self.process.pid)
                    status["cpu_percent"] = proc.cpu_percent(interval=0.1)
                    status["memory_mb"] = proc.memory_info().rss / 1024 / 1024
                    status["status"] = proc.status()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return status


# Global instance (will be initialized by server)
_ollama_manager: OllamaManager | None = None


def get_ollama_manager() -> OllamaManager:
    """Get the global Ollama manager instance.

    Returns:
        The initialized OllamaManager instance.

    Raises:
        RuntimeError: If manager has not been initialized via
            initialize_ollama_manager().
    """
    if _ollama_manager is None:
        raise RuntimeError("Ollama manager not initialized")
    return _ollama_manager


def initialize_ollama_manager(
    base_url: str = "http://localhost:11434",
    log_dir: Path | None = None,
    auto_detect_optimizations: bool = True,
    force_manage: bool = True,
) -> OllamaManager:
    """Initialize the global Ollama manager instance.

    Creates and stores a global OllamaManager instance. Should be called
    once during application startup.

    Args:
        base_url: Base URL for Ollama service.
        log_dir: Directory for Ollama logs. If None, uses default.
        auto_detect_optimizations: Whether to auto-detect system optimizations.
        force_manage: Whether to stop external Ollama instances and manage our own.

    Returns:
        The initialized OllamaManager instance.

    Side effects:
        Sets the global _ollama_manager variable.
    """
    global _ollama_manager
    _ollama_manager = OllamaManager(
        base_url=base_url,
        log_dir=log_dir,
        auto_detect_optimizations=auto_detect_optimizations,
        force_manage=force_manage,
    )
    return _ollama_manager
