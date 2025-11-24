"""Async process manager for the local Ollama service."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import psutil

from shared_ollama.core.utils import get_project_root, get_warmup_models
from shared_ollama.infrastructure.health_checker import check_ollama_health_simple

logger = logging.getLogger(__name__)

ProcessEnv = dict[str, str]

_CALC_MEMORY_SCRIPT: Final = (
    get_project_root() / "scripts" / "maintenance" / "calculate_memory_limit.sh"
)


@dataclass(slots=True, frozen=True)
class ProcessStatus:
    running: bool
    base_url: str
    managed: bool
    pid: int | None = None
    returncode: int | None = None
    cpu_percent: float | None = None
    memory_mb: float | None = None
    status: str | None = None


class OllamaManager:
    """Manage the Ollama subprocess lifecycle with optional warmup support."""

    __slots__ = (
        "base_url",
        "log_dir",
        "auto_detect_optimizations",
        "force_manage",
        "process",
        "_ollama_path",
        "_ollama_path_initialized",
    )

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        log_dir: Path | None = None,
        auto_detect_optimizations: bool = True,
        force_manage: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.log_dir = log_dir or self._default_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.auto_detect_optimizations = auto_detect_optimizations
        self.force_manage = force_manage
        self.process: asyncio.subprocess.Process | None = None
        self._ollama_path: str | None = None
        self._ollama_path_initialized = False

    @staticmethod
    def _default_log_dir() -> Path:
        return get_project_root() / "logs"

    @property
    def ollama_executable(self) -> str | None:
        if not self._ollama_path_initialized:
            path = shutil.which("ollama")
            if not path:
                logger.warning("Ollama executable not found in PATH")
            self._ollama_path = path
            self._ollama_path_initialized = True
        return self._ollama_path or None

    @ollama_executable.setter
    def ollama_executable(self, value: str | None) -> None:
        self._ollama_path = value
        self._ollama_path_initialized = True

    @ollama_executable.deleter
    def ollama_executable(self) -> None:
        self._ollama_path = None
        self._ollama_path_initialized = False

    @staticmethod
    async def _detect_system_optimizations() -> ProcessEnv:
        env: ProcessEnv = {
            "OLLAMA_HOST": "0.0.0.0:11434",
            "OLLAMA_KEEP_ALIVE": "30m",
        }

        match (platform.system(), platform.machine()):
            case ("Darwin", "arm64"):
                env.update(
                    {
                        "OLLAMA_METAL": "1",
                        "OLLAMA_NUM_GPU": "-1",
                        "OLLAMA_NUM_THREAD": str(os.cpu_count() or 8),
                    }
                )
            case ("Darwin", _):
                env.update({"OLLAMA_METAL": "0", "OLLAMA_NUM_GPU": "0"})
            case _:
                logger.debug("Using default optimization env for platform %s", platform.system())

        if _CALC_MEMORY_SCRIPT.exists():
            try:
                process = await asyncio.create_subprocess_exec(
                    "bash",
                    str(_CALC_MEMORY_SCRIPT),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                async with asyncio.timeout(5):
                    stdout, _ = await process.communicate()
                for line in (stdout or b"").decode().splitlines():
                    if line.startswith("OLLAMA_MAX_RAM="):
                        env["OLLAMA_MAX_RAM"] = line.split("=", 1)[1].strip()
                        break
            except TimeoutError:
                logger.debug("Memory limit helper timed out")
            except Exception as exc:  # pragma: no cover - best effort only
                logger.debug("Memory limit helper failed: %s", exc)

        return env

    async def _stop_external_ollama(self) -> None:
        logger.info("Stopping external Ollama instances")
        await asyncio.gather(
            self._stop_homebrew_service(),
            self._stop_launchd_service(),
            self._kill_residual_processes(),
        )
        if self._check_ollama_running(timeout=1):
            logger.warning("External Ollama instance still appears to be running")

    async def _stop_homebrew_service(self) -> None:
        if shutil.which("brew") is None:
            return
        try:
            process = await asyncio.create_subprocess_exec(
                "brew",
                "services",
                "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            async with asyncio.timeout(5):
                stdout, _ = await process.communicate()
        except Exception:
            return

        if b"ollama" not in (stdout or b"") or b"started" not in (stdout or b""):
            return

        logger.info("Stopping Homebrew-managed Ollama service")
        stop_process = await asyncio.create_subprocess_exec(
            "brew",
            "services",
            "stop",
            "ollama",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        with contextlib.suppress(Exception):
            async with asyncio.timeout(10):
                await stop_process.wait()
        await asyncio.sleep(2)

    async def _stop_launchd_service(self) -> None:
        plist = Path.home() / "Library/LaunchAgents/com.ollama.service.plist"
        if not plist.exists():
            return
        logger.info("Unloading launchd Ollama service")
        process = await asyncio.create_subprocess_exec(
            "launchctl",
            "unload",
            str(plist),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        with contextlib.suppress(Exception):
            async with asyncio.timeout(10):
                await process.wait()
        await asyncio.sleep(1)

    async def _kill_residual_processes(self) -> None:
        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmdline = proc.info.get("cmdline") or []
            if "ollama" in cmdline and "serve" in cmdline:
                logger.info("Terminating stray Ollama process PID %s", proc.pid)
                with contextlib.suppress(psutil.Error):
                    proc.terminate()
        await asyncio.sleep(2)

    def _check_ollama_running(self, timeout: int = 2) -> bool:
        return check_ollama_health_simple(base_url=self.base_url, timeout=timeout)

    async def start(self, wait_for_ready: bool = True, max_wait_time: int = 30) -> bool:
        ollama_path = self.ollama_executable
        if not ollama_path:
            logger.error("Install Ollama before attempting to start the service")
            return False

        if self._check_ollama_running():
            if self.force_manage:
                await self._stop_external_ollama()
            else:
                logger.info("External Ollama already running at %s", self.base_url)
                return True

        env = os.environ.copy()
        if self.auto_detect_optimizations:
            env.update(await self._detect_system_optimizations())

        stdout_path = self.log_dir / "ollama.log"
        stderr_path = self.log_dir / "ollama.error.log"

        with contextlib.ExitStack() as stack:
            stdout_handle = stack.enter_context(stdout_path.open("a"))
            stderr_handle = stack.enter_context(stderr_path.open("a"))
            try:
                self.process = await asyncio.create_subprocess_exec(
                    ollama_path,
                    "serve",
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=env,
                    start_new_session=True,
                )
            except Exception:
                logger.exception("Failed to start Ollama process")
                self.process = None
                return False

        logger.info("Started Ollama process PID %s", self.process.pid)

        if not wait_for_ready:
            return True

        ready = await self._wait_for_ready(max_wait_time)
        if not ready:
            logger.error("Ollama failed readiness checks within %ss", max_wait_time)
            await self.stop()
        return ready

    async def _wait_for_ready(self, max_wait_time: int) -> bool:
        try:
            async with asyncio.timeout(max_wait_time):
                while True:
                    try:
                        running = self._check_ollama_running(timeout=1)
                    except TypeError:
                        running = self._check_ollama_running()
                    if running:
                        return True
                    await asyncio.sleep(1)
        except TimeoutError:
            return False

    async def stop(self, timeout: int = 10) -> bool:
        if self.process is None:
            if self._check_ollama_running():
                logger.warning("Service running but process not managed by this instance")
                return False
            logger.info("Ollama service already stopped")
            return True

        logger.info("Stopping Ollama process PID %s", self.process.pid)
        try:
            self.process.terminate()
        except Exception as exc:
            logger.exception("Failed to terminate Ollama process: %s", exc)
            await self._force_kill_process(self.process)
            self.process = None
            return False
        try:
            async with asyncio.timeout(timeout):
                await self.process.wait()
        except TimeoutError:
            logger.warning("Force killing Ollama process PID %s", self.process.pid)
            await self._force_kill_process(self.process)
            self.process = None
            return True
        finally:
            self.process = None
        return True

    async def _force_kill_process(self, process: asyncio.subprocess.Process | None) -> None:
        if process is None:
            return
        with contextlib.suppress(Exception):
            process.kill()
            await process.wait()

    async def warmup_models(self) -> None:
        models = get_warmup_models()
        if not models:
            logger.info("No warmup models configured")
            return

        logger.info("Pre-warming models: %s", ", ".join(models))
        try:
            tg = asyncio.TaskGroup()
        except AttributeError:  # pragma: no cover - Python <3.11 fallback
            await asyncio.gather(*(self._pull_model(model) for model in models))
        else:
            async with tg:
                for model in models:
                    tg.create_task(self._pull_model(model))

    async def _pull_model(self, model_name: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                "ollama",
                "pull",
                model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            async with asyncio.timeout(600):
                _, stderr = await process.communicate()
            if process.returncode == 0:
                logger.info("Warmup pull succeeded for %s", model_name)
            else:
                logger.error(
                    "Warmup pull failed for %s (code %s): %s",
                    model_name,
                    process.returncode,
                    (stderr or b"").decode(errors="ignore"),
                )
        except TimeoutError:
            logger.error("Warmup pull timed out for %s", model_name)
        except Exception:
            logger.exception("Unexpected error while pulling %s", model_name)

    def is_running(self) -> bool:
        if self.process:
            if self.process.returncode is not None:
                self.process = None
                return False
            with contextlib.suppress(psutil.Error):
                ps_proc = psutil.Process(self.process.pid)
                if ps_proc.is_running():
                    return True
            self.process = None
            return False
        return self._check_ollama_running()

    def get_status(self) -> ProcessStatus:
        managed_process = self.process
        running = self.is_running()
        proc_ref = self.process or managed_process
        cpu, memory, proc_status = self._collect_process_metrics()
        return ProcessStatus(
            running=running,
            base_url=self.base_url,
            managed=self.process is not None,
            pid=proc_ref.pid if proc_ref else None,
            returncode=proc_ref.returncode if proc_ref else None,
            cpu_percent=cpu,
            memory_mb=memory,
            status=proc_status,
        )

    def _collect_process_metrics(self) -> tuple[float | None, float | None, str | None]:
        cpu: float | None = None
        memory: float | None = None
        status: str | None = None
        if self.process:
            with contextlib.suppress(psutil.Error):
                proc = psutil.Process(self.process.pid)
                cpu = proc.cpu_percent(interval=0.1)
                memory = proc.memory_info().rss / (1024 * 1024)
                status = proc.status()
        return cpu, memory, status


_ollama_manager: OllamaManager | None = None


def get_ollama_manager() -> OllamaManager:
    if _ollama_manager is None:
        raise RuntimeError("Ollama manager not initialized")
    return _ollama_manager


def initialize_ollama_manager(
    base_url: str = "http://localhost:11434",
    log_dir: Path | None = None,
    auto_detect_optimizations: bool = True,
    force_manage: bool = True,
) -> OllamaManager:
    global _ollama_manager
    _ollama_manager = OllamaManager(
        base_url=base_url,
        log_dir=log_dir,
        auto_detect_optimizations=auto_detect_optimizations,
        force_manage=force_manage,
    )
    return _ollama_manager
