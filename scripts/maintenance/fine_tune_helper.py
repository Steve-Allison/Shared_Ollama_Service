#!/usr/bin/env python3
"""Fine-tuning helper script for local Ollama model customization.

This script provides utilities for fine-tuning and customizing Ollama models
locally. It helps with creating Modelfiles, preparing training data, and
managing custom models.

Usage:
    python scripts/maintenance/fine_tune_helper.py create-modelfile --base-model llama3.2 --output custom-model
    python scripts/maintenance/fine_tune_helper.py prepare-data --input data.jsonl --output training.jsonl
    python scripts/maintenance/fine_tune_helper.py create-model --name custom-model --modelfile Modelfile
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


def create_modelfile(
    base_model: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    output_path: Path | None = None,
) -> str:
    """Create a Modelfile for custom model.

    Args:
        base_model: Base model name to customize.
        system_prompt: Optional system prompt to set behavior.
        temperature: Optional temperature setting.
        top_p: Optional top_p setting.
        output_path: Optional path to save Modelfile.

    Returns:
        Modelfile content as string.
    """
    lines = [f"FROM {base_model}"]

    if system_prompt:
        lines.append(f'SYSTEM """{system_prompt}"""')

    if temperature is not None:
        lines.append(f"PARAMETER temperature {temperature}")

    if top_p is not None:
        lines.append(f"PARAMETER top_p {top_p}")

    modelfile_content = "\n".join(lines)

    if output_path:
        output_path.write_text(modelfile_content, encoding="utf-8")
        print(f"Modelfile written to {output_path}")

    return modelfile_content


def prepare_training_data(
    input_path: Path,
    output_path: Path,
    format_type: str = "jsonl",
) -> None:
    """Prepare training data for fine-tuning.

    Args:
        input_path: Path to input data file.
        output_path: Path to write prepared data.
        format_type: Data format (jsonl, alpaca, etc.).
    """
    if format_type == "jsonl":
        # Simple pass-through for now
        data = input_path.read_text(encoding="utf-8")
        output_path.write_text(data, encoding="utf-8")
        print(f"Training data prepared: {output_path}")
    else:
        print(f"Format {format_type} not yet supported")


def create_model_via_api(
    name: str,
    modelfile_path: Path,
    base_url: str = "http://localhost:11434",
) -> None:
    """Create a model via Ollama API.

    Args:
        name: Model name to create.
        modelfile_path: Path to Modelfile.
        base_url: Ollama service base URL.
    """
    modelfile_content = modelfile_path.read_text(encoding="utf-8")

    with httpx.Client(timeout=600.0) as client:
        response = client.post(
            f"{base_url}/api/create",
            json={"name": name, "modelfile": modelfile_content, "stream": False},
        )
        response.raise_for_status()
        print(f"Model '{name}' created successfully")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ollama fine-tuning helper")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # create-modelfile command
    modelfile_parser = subparsers.add_parser("create-modelfile", help="Create a Modelfile")
    modelfile_parser.add_argument("--base-model", required=True, help="Base model name")
    modelfile_parser.add_argument("--system-prompt", help="System prompt")
    modelfile_parser.add_argument("--temperature", type=float, help="Temperature setting")
    modelfile_parser.add_argument("--top-p", type=float, help="Top-p setting")
    modelfile_parser.add_argument("--output", type=Path, help="Output file path")

    # prepare-data command
    data_parser = subparsers.add_parser("prepare-data", help="Prepare training data")
    data_parser.add_argument("--input", type=Path, required=True, help="Input data file")
    data_parser.add_argument("--output", type=Path, required=True, help="Output data file")
    data_parser.add_argument("--format", default="jsonl", help="Data format")

    # create-model command
    create_parser = subparsers.add_parser("create-model", help="Create model via API")
    create_parser.add_argument("--name", required=True, help="Model name")
    create_parser.add_argument("--modelfile", type=Path, required=True, help="Modelfile path")
    create_parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama URL")

    args = parser.parse_args()

    if args.command == "create-modelfile":
        create_modelfile(
            base_model=args.base_model,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            output_path=args.output,
        )
    elif args.command == "prepare-data":
        prepare_training_data(
            input_path=args.input,
            output_path=args.output,
            format_type=args.format,
        )
    elif args.command == "create-model":
        create_model_via_api(
            name=args.name,
            modelfile_path=args.modelfile,
            base_url=args.base_url,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

