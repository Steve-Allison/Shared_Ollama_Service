"""
Ollama Process Manager

Manages the Ollama service process lifecycle internally within the REST API.
Handles starting, stopping, and health checking of the Ollama backend.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


class OllamaManager:
    """Manages the Ollama service process lifecycle."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        log_dir: Path | None = None,
        auto_detect_optimizations: bool = True,
    ):
        """
        Initialize the Ollama manager.

        Args:
            base_url: Base URL for Ollama service
            log_dir: Directory for Ollama logs (defaults to project logs/)
            auto_detect_optimizations: Automatically detect and apply system optimizations
        """
        self.base_url = base_url.rstrip("/")
        self.process: subprocess.Popen[str] | None = None
        self.log_dir = log_dir or self._get_default_log_dir()
        self.auto_detect_optimizations = auto_detect_optimizations
        self._ollama_path: str | None = None

    def _get_default_log_dir(self) -> Path:
        """Get default log directory."""
        project_root = Path(__file__).resolve().parents[3]
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        return log_dir

    def _find_ollama_executable(self) -> str | None:
        """Find the Ollama executable path."""
        if self._ollama_path:
            return self._ollama_path

        # Check if ollama is in PATH
        ollama_path = shutil.which("ollama")
        if ollama_path:
            self._ollama_path = ollama_path
            return ollama_path

        logger.warning("Ollama executable not found in PATH")
        return None

    def _detect_system_optimizations(self) -> dict[str, str]:
        """
        Detect system-specific optimizations for Ollama.

        Returns:
            Dictionary of environment variables to set for optimal performance
        """
        optimizations: dict[str, str] = {}

        # Detect system architecture
        arch = platform.machine()
        system = platform.system()

        if system == "Darwin":  # macOS
            # Check if Apple Silicon
            if arch == "arm64":
                # Apple Silicon: Enable Metal/MPS GPU acceleration
                optimizations["OLLAMA_METAL"] = "1"
                optimizations["OLLAMA_NUM_GPU"] = "-1"  # Use all GPU cores

                # Auto-detect CPU cores
                try:
                    cpu_cores = os.cpu_count() or 10
                    optimizations["OLLAMA_NUM_THREAD"] = str(cpu_cores)
                except Exception:
                    optimizations["OLLAMA_NUM_THREAD"] = "10"
            else:
                # Intel Mac: CPU only
                optimizations["OLLAMA_METAL"] = "0"
                optimizations["OLLAMA_NUM_GPU"] = "0"

        # Set default host and keep-alive
        optimizations.setdefault("OLLAMA_HOST", "0.0.0.0:11434")
        optimizations.setdefault("OLLAMA_KEEP_ALIVE", "30m")

        # Auto-calculate memory limit if script exists
        try:
            project_root = Path(__file__).resolve().parents[3]
            calc_script = project_root / "scripts" / "calculate_memory_limit.sh"
            if calc_script.exists():
                result = subprocess.run(
                    ["bash", str(calc_script)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in result.stdout.splitlines():
                    if line.startswith("OLLAMA_MAX_RAM="):
                        max_ram = line.split("=", 1)[1].strip()
                        optimizations["OLLAMA_MAX_RAM"] = max_ram
                        break
        except Exception as exc:
            logger.debug("Could not auto-calculate memory limit: %s", exc)

        return optimizations

    def _check_ollama_running(self, timeout: int = 2) -> bool:
        """Check if Ollama is already running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False

    async def start(self, wait_for_ready: bool = True, max_wait_time: int = 30) -> bool:
        """
        Start the Ollama service process.

        Args:
            wait_for_ready: Wait for Ollama to be ready before returning
            max_wait_time: Maximum time to wait for Ollama to be ready (seconds)

        Returns:
            True if Ollama started successfully, False otherwise
        """
        # Check if already running
        if self._check_ollama_running():
            logger.info("Ollama service is already running at %s", self.base_url)
            return True

        # Find Ollama executable
        ollama_path = self._find_ollama_executable()
        if not ollama_path:
            logger.error("Ollama executable not found. Please install Ollama.")
            return False

        # Prepare environment with optimizations
        env = os.environ.copy()
        if self.auto_detect_optimizations:
            optimizations = self._detect_system_optimizations()
            env.update(optimizations)
            logger.info("Applied system optimizations: %s", optimizations)

        # Prepare log files
        log_file = self.log_dir / "ollama.log"
        error_log_file = self.log_dir / "ollama.error.log"

        # Start Ollama process
        try:
            logger.info("Starting Ollama service process...")
            with open(log_file, "a") as log_f, open(error_log_file, "a") as err_f:
                self.process = subprocess.Popen(
                    [ollama_path, "serve"],
                    stdout=log_f,
                    stderr=err_f,
                    env=env,
                    start_new_session=True,  # Create new process group
                )

            logger.info("Ollama process started with PID %s", self.process.pid)

            # Wait for service to be ready
            if wait_for_ready:
                ready = await self._wait_for_ready(max_wait_time)
                if not ready:
                    logger.error("Ollama service did not become ready within %s seconds", max_wait_time)
                    await self.stop()
                    return False

            return True

        except Exception as exc:
            logger.exception("Failed to start Ollama process: %s", exc)
            self.process = None
            return False

    async def _wait_for_ready(self, max_wait_time: int = 30) -> bool:
        """Wait for Ollama service to be ready."""
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            if self._check_ollama_running(timeout=1):
                logger.info("Ollama service is ready")
                return True
            await asyncio.sleep(1)

        return False

    async def stop(self, timeout: int = 10) -> bool:
        """
        Stop the Ollama service process.

        Args:
            timeout: Maximum time to wait for graceful shutdown (seconds)

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.process:
            # Check if Ollama is running (might have been started externally)
            if not self._check_ollama_running():
                logger.info("Ollama service is not running")
                return True
            logger.warning("Ollama process not managed by this manager, but service is running")
            return False

        try:
            logger.info("Stopping Ollama service process (PID %s)...", self.process.pid)

            # Try graceful shutdown (SIGTERM)
            self.process.terminate()

            # Wait for process to terminate
            try:
                self.process.wait(timeout=timeout)
                logger.info("Ollama process stopped gracefully")
                self.process = None
                return True
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning("Ollama process did not stop gracefully, forcing termination...")
                self.process.kill()
                self.process.wait()
                logger.info("Ollama process force-stopped")
                self.process = None
                return True

        except Exception as exc:
            logger.exception("Error stopping Ollama process: %s", exc)
            if self.process:
                try:
                    self.process.kill()
                except Exception:
                    pass
                self.process = None
            return False

    def is_running(self) -> bool:
        """Check if Ollama service is currently running."""
        if self.process:
            # Check if process is still alive
            if self.process.poll() is not None:
                # Process has terminated
                self.process = None
                return False

        return self._check_ollama_running()

    def get_status(self) -> dict[str, Any]:
        """
        Get current status of Ollama service.

        Returns:
            Dictionary with status information
        """
        status = {
            "running": self.is_running(),
            "base_url": self.base_url,
            "managed": self.process is not None,
        }

        if self.process:
            status["pid"] = self.process.pid
            status["returncode"] = self.process.returncode

        return status


# Global instance (will be initialized by server)
_ollama_manager: OllamaManager | None = None


def get_ollama_manager() -> OllamaManager:
    """Get the global Ollama manager instance."""
    if _ollama_manager is None:
        raise RuntimeError("Ollama manager not initialized")
    return _ollama_manager


def initialize_ollama_manager(
    base_url: str = "http://localhost:11434",
    log_dir: Path | None = None,
    auto_detect_optimizations: bool = True,
) -> OllamaManager:
    """Initialize the global Ollama manager."""
    global _ollama_manager
    _ollama_manager = OllamaManager(
        base_url=base_url,
        log_dir=log_dir,
        auto_detect_optimizations=auto_detect_optimizations,
    )
    return _ollama_manager

