#!/bin/bash

# Shared helper for loading model configuration from .env

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

load_model_config() {
    if [[ -n "${MODEL_CONFIG_LOADED:-}" ]]; then
        return
    fi

    if [ -f "$ENV_FILE" ]; then
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
    fi

    DEFAULT_VLM_MODEL="${OLLAMA_DEFAULT_VLM_MODEL:-qwen3-vl:8b-instruct-q4_K_M}"
    DEFAULT_TEXT_MODEL="${OLLAMA_DEFAULT_TEXT_MODEL:-qwen3:14b-q4_K_M}"

    REQUIRED_MODELS_CSV="${OLLAMA_REQUIRED_MODELS:-$DEFAULT_VLM_MODEL,$DEFAULT_TEXT_MODEL}"
    IFS=',' read -r -a REQUIRED_MODELS <<< "$REQUIRED_MODELS_CSV"

    WARMUP_MODELS_CSV="${OLLAMA_WARMUP_MODELS:-$DEFAULT_VLM_MODEL,$DEFAULT_TEXT_MODEL}"
    IFS=',' read -r -a WARMUP_MODELS <<< "$WARMUP_MODELS_CSV"

    MODEL_CONFIG_LOADED=1
}

