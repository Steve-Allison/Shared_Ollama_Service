#!/bin/bash

# Shared helper for loading model configuration from config/model_profiles.yaml

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PROFILE_FILE="$PROJECT_ROOT/config/model_profiles.yaml"
DETECT_SCRIPT="$PROJECT_ROOT/scripts/detect_system.sh"

load_model_config() {
    if [[ -n "${MODEL_CONFIG_LOADED:-}" ]]; then
        return
    fi

    # Allow explicit overrides via environment variables if the user sets them
    DEFAULT_VLM_MODEL="${OLLAMA_DEFAULT_VLM_MODEL:-}"
    DEFAULT_TEXT_MODEL="${OLLAMA_DEFAULT_TEXT_MODEL:-}"
    REQUIRED_MODELS_CSV="${OLLAMA_REQUIRED_MODELS:-}"
    WARMUP_MODELS_CSV="${OLLAMA_WARMUP_MODELS:-}"

    MODEL_MEMORY_HINTS_CSV="${OLLAMA_MODEL_MEMORY_HINTS:-}"
    LARGEST_MODEL_GB="${OLLAMA_LARGEST_MODEL_GB:-}"
    INFERENCE_BUFFER_GB="${OLLAMA_INFERENCE_BUFFER_GB:-}"
    SERVICE_OVERHEAD_GB="${OLLAMA_SERVICE_OVERHEAD_GB:-}"

    if [[ -z "$DEFAULT_VLM_MODEL" || -z "$DEFAULT_TEXT_MODEL" || -z "$REQUIRED_MODELS_CSV" || -z "$WARMUP_MODELS_CSV" || -z "$MODEL_MEMORY_HINTS_CSV" || -z "$LARGEST_MODEL_GB" ]]; then
        ARCH=""
        TOTAL_RAM_GB=""

        if [[ -x "$DETECT_SCRIPT" ]]; then
            SYSTEM_INFO=$(bash "$DETECT_SCRIPT" 2>/dev/null || true)
            while IFS='=' read -r key value; do
                [[ -z "$key" || "$key" =~ ^# ]] && continue
                case "$key" in
                    ARCH) ARCH="$value" ;;
                    TOTAL_RAM_GB) TOTAL_RAM_GB="$value" ;;
                esac
            done <<< "$SYSTEM_INFO"
        fi

        PROFILE_DEFAULTS=$(PROJECT_ROOT="$PROJECT_ROOT" ARCH="$ARCH" TOTAL_RAM_GB="$TOTAL_RAM_GB" python3 - <<'PY'
import json
import math
import os
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml is part of project deps
    yaml = None

project_root = Path(os.environ["PROJECT_ROOT"])
profile_path = project_root / "config" / "model_profiles.yaml"

defaults = {}
if yaml and profile_path.exists():
    with profile_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    profiles = data.get("profiles") or {}
    ram = int(os.environ.get("TOTAL_RAM_GB") or 0)
    arch = os.environ.get("ARCH")
    selected = profiles.get("default", {}) if isinstance(profiles, dict) else {}
    for profile in profiles.values():
        if not isinstance(profile, dict):
            continue
        match = profile.get("match") or {}
        min_ram = match.get("min_ram_gb", 0)
        max_ram = match.get("max_ram_gb", math.inf)
        match_arch = match.get("arch")
        if ram >= min_ram and ram <= max_ram and (match_arch is None or match_arch == arch):
            selected = profile
            break
    defaults = selected.get("defaults") or {}

print(json.dumps(defaults))
PY
)

        if [[ -n "$PROFILE_DEFAULTS" && "$PROFILE_DEFAULTS" != "null" ]]; then
            DEFAULT_VLM_MODEL="${DEFAULT_VLM_MODEL:-$(echo "$PROFILE_DEFAULTS" | jq -r '.vlm_model // empty')}"
            DEFAULT_TEXT_MODEL="${DEFAULT_TEXT_MODEL:-$(echo "$PROFILE_DEFAULTS" | jq -r '.text_model // empty')}"
            REQUIRED_MODELS_CSV="${REQUIRED_MODELS_CSV:-$(echo "$PROFILE_DEFAULTS" | jq -r '(.required_models // []) | join(",")')}"
            WARMUP_MODELS_CSV="${WARMUP_MODELS_CSV:-$(echo "$PROFILE_DEFAULTS" | jq -r '(.warmup_models // []) | join(",")')}"
            MODEL_MEMORY_HINTS_CSV="${MODEL_MEMORY_HINTS_CSV:-$(echo "$PROFILE_DEFAULTS" | jq -r '(.memory_hints // {}) | to_entries | map("\(.key):\(.value)") | join(",")')}"
            LARGEST_MODEL_GB="${LARGEST_MODEL_GB:-$(echo "$PROFILE_DEFAULTS" | jq -r '.largest_model_gb // empty')}"
            INFERENCE_BUFFER_GB="${INFERENCE_BUFFER_GB:-$(echo "$PROFILE_DEFAULTS" | jq -r '.inference_buffer_gb // empty')}"
            SERVICE_OVERHEAD_GB="${SERVICE_OVERHEAD_GB:-$(echo "$PROFILE_DEFAULTS" | jq -r '.service_overhead_gb // empty')}"
        fi
    fi

    DEFAULT_VLM_MODEL="${DEFAULT_VLM_MODEL:-qwen3-vl:32b}"
    DEFAULT_TEXT_MODEL="${DEFAULT_TEXT_MODEL:-qwen3:30b}"
    MODEL_MEMORY_HINTS_CSV="${MODEL_MEMORY_HINTS_CSV:-$DEFAULT_VLM_MODEL:21,$DEFAULT_TEXT_MODEL:19}"
    LARGEST_MODEL_GB="${LARGEST_MODEL_GB:-21}"
    INFERENCE_BUFFER_GB="${INFERENCE_BUFFER_GB:-4}"
    SERVICE_OVERHEAD_GB="${SERVICE_OVERHEAD_GB:-2}"

    if [[ -z "$REQUIRED_MODELS_CSV" ]]; then
        REQUIRED_MODELS_CSV="$DEFAULT_VLM_MODEL,$DEFAULT_TEXT_MODEL"
    fi
    IFS=',' read -r -a REQUIRED_MODELS <<< "$REQUIRED_MODELS_CSV"

    if [[ -z "$WARMUP_MODELS_CSV" ]]; then
        WARMUP_MODELS_CSV="$DEFAULT_VLM_MODEL,$DEFAULT_TEXT_MODEL"
    fi
    IFS=',' read -r -a WARMUP_MODELS <<< "$WARMUP_MODELS_CSV"

    MODEL_MEMORY_HINTS="$MODEL_MEMORY_HINTS_CSV"
    MODEL_CONFIG_LOADED=1
}

