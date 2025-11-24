from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT_DIRS = ("core", "maintenance")


def _collect_scripts() -> list[Path]:
    scripts: list[Path] = []
    for directory in SCRIPT_DIRS:
        for path in sorted((SCRIPT_ROOT / directory).glob("*.sh")):
            scripts.append(path)
    return scripts


SHELL_SCRIPTS = _collect_scripts()


@pytest.mark.parametrize("script_path", SHELL_SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_shell_scripts_have_exec_bit(script_path: Path) -> None:
    assert os.access(script_path, os.X_OK), f"{script_path} must be executable"


@pytest.mark.parametrize("script_path", SHELL_SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_shell_scripts_pass_bash_syntax_check(script_path: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

