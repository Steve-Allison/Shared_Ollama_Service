from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from shared_ollama.core import utils

SAMPLE_CONFIG = """profiles:
  - name: small
    max_ram_gb: 32
    vlm_model: qwen3-vl:8b-instruct-q4_K_M
    text_model: qwen3:14b-q4_K_M
    required_models:
      - qwen3-vl:8b-instruct-q4_K_M
      - qwen3:14b-q4_K_M
    warmup_models:
      - qwen3-vl:8b-instruct-q4_K_M
      - qwen3:14b-q4_K_M
    memory_hints:
      qwen3-vl:8b-instruct-q4_K_M: 6
      qwen3:14b-q4_K_M: 8
    largest_model_gb: 8
  - name: large
    min_ram_gb: 33
    vlm_model: qwen3-vl:32b
    text_model: qwen3:30b
    required_models:
      - qwen3-vl:32b
      - qwen3:30b
    warmup_models:
      - qwen3-vl:32b
      - qwen3:30b
    memory_hints:
      qwen3-vl:32b: 21
      qwen3:30b: 19
    largest_model_gb: 21
defaults:
  inference_buffer_gb: 4
  service_overhead_gb: 2
"""


def write_config(directory: Path, contents: str = SAMPLE_CONFIG) -> Path:
    config_path = directory / "config" / "models.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(contents, encoding="utf-8")
    return config_path


def reset_caches() -> None:
    utils._load_model_profile_defaults.cache_clear()
    utils.get_default_vlm_model.cache_clear()
    utils.get_default_text_model.cache_clear()
    utils.get_warmup_models.cache_clear()
    utils.get_allowed_models.cache_clear()


class TestModelConfig:
    def setup_method(self) -> None:
        reset_caches()

    def test_small_profile_selected_for_low_ram(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_config(root)
            with patch.object(utils, "get_project_root", return_value=root):
                with patch.object(utils, "_detect_system_info", return_value=("arm64", 32)):
                    defaults = utils._load_model_profile_defaults()

        assert defaults["vlm_model"] == "qwen3-vl:8b-instruct-q4_K_M"
        assert defaults["text_model"] == "qwen3:14b-q4_K_M"
        assert defaults["required_models"] == [
            "qwen3-vl:8b-instruct-q4_K_M",
            "qwen3:14b-q4_K_M",
        ]
        assert defaults["largest_model_gb"] == 8

    def test_large_profile_selected_for_high_ram(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_config(root)
            with patch.object(utils, "get_project_root", return_value=root):
                with patch.object(utils, "_detect_system_info", return_value=("arm64", 96)):
                    defaults = utils._load_model_profile_defaults()

        assert defaults["vlm_model"] == "qwen3-vl:32b"
        assert defaults["text_model"] == "qwen3:30b"
        assert defaults["largest_model_gb"] == 21

    def test_missing_config_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(utils, "get_project_root", return_value=root):
                with patch.object(utils, "_detect_system_info", return_value=("arm64", 32)):
                    with pytest.raises(RuntimeError, match="Model configuration file not found"):
                        utils._load_model_profile_defaults()

    def test_get_default_helpers_use_selected_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_config(root)
            with patch.object(utils, "get_project_root", return_value=root):
                with patch.object(utils, "_detect_system_info", return_value=("arm64", 64)):
                    vlm = utils.get_default_vlm_model()
                    text = utils.get_default_text_model()
                    warmup = utils.get_warmup_models()
                    allowed = utils.get_allowed_models()

        assert vlm == "qwen3-vl:32b"
        assert text == "qwen3:30b"
        assert warmup == ["qwen3-vl:32b", "qwen3:30b"]
        assert "qwen3-vl:32b" in allowed
        assert "qwen3:30b" in allowed

    def test_is_model_allowed_respects_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_config(root)
            with patch.object(utils, "get_project_root", return_value=root):
                with patch.object(utils, "_detect_system_info", return_value=("arm64", 16)):
                    allowed = utils.get_allowed_models()

        assert "qwen3-vl:8b-instruct-q4_K_M" in allowed
        assert "qwen3:30b" not in allowed

