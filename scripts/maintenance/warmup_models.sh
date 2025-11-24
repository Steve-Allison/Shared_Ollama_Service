#!/bin/bash

# Warm-up script to preload models into memory
# This reduces first-request latency by loading models before they're needed

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/model_config.sh"
load_model_config

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
API_ENDPOINT="${OLLAMA_URL}/api"

# Models to warm up (primary models, in priority order)
MODELS=("${WARMUP_MODELS[@]}")

# Optional: keep-alive duration for warm-up (default: 30 minutes)
KEEP_ALIVE="${KEEP_ALIVE:-30m}"

echo -e "${BLUE}🔥 Warming up Ollama models${NC}"
echo "======================================"
echo ""

# Check if Ollama is running
if ! curl -f -s "${API_ENDPOINT}/tags" > /dev/null 2>&1; then
    echo -e "${RED}✗ Ollama service is not running${NC}"
    echo "Start Ollama first: ollama serve"
    exit 1
fi

echo -e "${GREEN}✓ Ollama service is running${NC}"
echo ""

# Fall back to defaults if no warmup models were provided
if [ ${#MODELS[@]} -eq 0 ]; then
    echo -e "${RED}✗ No warm-up models configured. load_model_config must define WARMUP_MODELS.${NC}"
    exit 1
fi

# Deduplicate models while preserving order
declare -A SEEN_MODELS=()
UNIQUE_MODELS=()
for model in "${MODELS[@]}"; do
    [[ -z "$model" ]] && continue
    if [[ -z "${SEEN_MODELS[$model]:-}" ]]; then
        SEEN_MODELS[$model]=1
        UNIQUE_MODELS+=("$model")
    fi
done

if [ ${#UNIQUE_MODELS[@]} -eq 0 ]; then
    echo -e "${RED}✗ Warm-up model list is empty after validation. Check configuration.${NC}"
    exit 1
fi

# Cache available models to avoid unnecessary errors
AVAILABLE_MODELS=$(curl -s "${API_ENDPOINT}/tags" | jq -r '.models[].name' 2>/dev/null || true)

warm_model() {
    local model=$1
    echo -e "${BLUE}Warming up ${model}...${NC}"

    if [ -n "$AVAILABLE_MODELS" ] && ! echo "$AVAILABLE_MODELS" | grep -qx "$model"; then
        echo -e "${YELLOW}  ⚠ Model ${model} not available on this system. Skipping.${NC}"
        echo ""
        return
    fi

    if ! RESPONSE=$(curl -s -X POST "${API_ENDPOINT}/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model}\",
            \"prompt\": \"Hi\",
            \"stream\": false,
            \"options\": {
                \"num_predict\": 1
            },
            \"keep_alive\": \"${KEEP_ALIVE}\"
        }" 2>/dev/null); then
        echo -e "${YELLOW}  ⚠ ${model} warm-up request failed${NC}"
        echo "    Verify the model is installed: ollama list"
        echo ""
        return
    fi

    if echo "$RESPONSE" | grep -q "\"response\""; then
        echo -e "${GREEN}  ✓ ${model} warmed up and loaded into memory${NC}"
        echo -e "    Keep-alive: ${KEEP_ALIVE}"
    else
        echo -e "${YELLOW}  ⚠ ${model} warm-up may have failed${NC}"
        echo "    Response:"
        echo "$RESPONSE" | sed 's/^/      /'
    fi

    echo ""
}

# Warm up each model
for model in "${UNIQUE_MODELS[@]}"; do
    warm_model "$model"
done

echo "======================================"
echo -e "${GREEN}✓ Model warm-up complete!${NC}"
echo ""
echo "Models are now loaded in memory and ready for fast inference."
echo "They will remain loaded for ${KEEP_ALIVE} (or until unloaded manually)."
echo ""
echo "To check model status:"
echo "  curl ${API_ENDPOINT}/tags"

