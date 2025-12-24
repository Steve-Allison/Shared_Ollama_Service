"""Behavioral tests for fine-tuning helper script.

Tests focus on real behavior: modelfile creation, data preparation,
and model creation via API.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the fine_tune_helper module functions
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "maintenance"))


class TestCreateModelfile:
    """Tests for create_modelfile function."""

    def test_create_modelfile_with_base_model(self):
        """Test modelfile creation with base model."""
        from fine_tune_helper import create_modelfile

        modelfile = create_modelfile(base_model="qwen3:14b-q4_K_M")

        assert "FROM qwen3:14b-q4_K_M" in modelfile
        assert modelfile.startswith("FROM")

    def test_create_modelfile_with_system_prompt(self):
        """Test modelfile creation with system prompt."""
        from fine_tune_helper import create_modelfile

        modelfile = create_modelfile(
            base_model="qwen3:14b-q4_K_M",
            system_prompt="You are a helpful assistant",
        )

        assert "FROM qwen3:14b-q4_K_M" in modelfile
        assert 'SYSTEM """You are a helpful assistant"""' in modelfile

    def test_create_modelfile_with_temperature(self):
        """Test modelfile creation with temperature."""
        from fine_tune_helper import create_modelfile

        modelfile = create_modelfile(
            base_model="qwen3:14b-q4_K_M",
            temperature=0.7,
        )

        assert "PARAMETER temperature 0.7" in modelfile

    def test_create_modelfile_with_top_p(self):
        """Test modelfile creation with top_p."""
        from fine_tune_helper import create_modelfile

        modelfile = create_modelfile(
            base_model="qwen3:14b-q4_K_M",
            top_p=0.9,
        )

        assert "PARAMETER top_p 0.9" in modelfile

    def test_create_modelfile_with_all_options(self):
        """Test modelfile creation with all options."""
        from fine_tune_helper import create_modelfile

        modelfile = create_modelfile(
            base_model="qwen3:14b-q4_K_M",
            system_prompt="Custom prompt",
            temperature=0.7,
            top_p=0.9,
        )

        assert "FROM qwen3:14b-q4_K_M" in modelfile
        assert "SYSTEM" in modelfile
        assert "PARAMETER temperature 0.7" in modelfile
        assert "PARAMETER top_p 0.9" in modelfile

    def test_create_modelfile_writes_to_file(self):
        """Test modelfile creation writes to file."""
        from fine_tune_helper import create_modelfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".modelfile") as f:
            output_path = Path(f.name)

        try:
            create_modelfile(
                base_model="qwen3:14b-q4_K_M",
                output_path=output_path,
            )

            assert output_path.exists()
            content = output_path.read_text()
            assert "FROM qwen3:14b-q4_K_M" in content
        finally:
            output_path.unlink()


class TestPrepareTrainingData:
    """Tests for prepare_training_data function."""

    def test_prepare_training_data_jsonl_format(self):
        """Test data preparation for JSONL format."""
        from fine_tune_helper import prepare_training_data

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "output.jsonl"

            # Create input file
            input_path.write_text('{"prompt": "test", "response": "answer"}\n')

            prepare_training_data(
                input_path=input_path,
                output_path=output_path,
                format_type="jsonl",
            )

            assert output_path.exists()
            assert output_path.read_text() == input_path.read_text()


class TestCreateModelViaAPI:
    """Tests for create_model_via_api function."""

    @patch("fine_tune_helper.httpx.Client")
    def test_create_model_via_api_success(self, mock_client_class):
        """Test successful model creation via API."""
        from fine_tune_helper import create_model_via_api

        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".modelfile") as f:
            modelfile_path = Path(f.name)
            modelfile_path.write_text("FROM qwen3:14b-q4_K_M\nSYSTEM \"Test\"")

        try:
            create_model_via_api(
                name="test-model",
                modelfile_path=modelfile_path,
                base_url="http://localhost:11434",
            )

            assert mock_client.post.called
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/create"
            assert call_args[1]["json"]["name"] == "test-model"
        finally:
            modelfile_path.unlink()

    @patch("fine_tune_helper.httpx.Client")
    def test_create_model_via_api_handles_errors(self, mock_client_class):
        """Test create_model_via_api handles API errors."""
        from fine_tune_helper import create_model_via_api

        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("API error")
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".modelfile") as f:
            modelfile_path = Path(f.name)
            modelfile_path.write_text("FROM qwen3:14b-q4_K_M")

        try:
            with pytest.raises(Exception):
                create_model_via_api(
                    name="test-model",
                    modelfile_path=modelfile_path,
                )
        finally:
            modelfile_path.unlink()

