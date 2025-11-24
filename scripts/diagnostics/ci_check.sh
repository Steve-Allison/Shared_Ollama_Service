#!/bin/bash

# CI/CD Helper Script
# Verifies Shared Ollama Service is running before tests
# Use this in your CI/CD pipelines or test setup

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/model_config.sh"
load_model_config

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
API_ENDPOINT="${OLLAMA_URL}/api"
MAX_WAIT=${MAX_WAIT:-30}  # Maximum seconds to wait for service
RETRY_INTERVAL=${RETRY_INTERVAL:-2}  # Seconds between retries

echo -e "${BLUE}🔍 Checking Shared Ollama Service for CI/CD${NC}"
echo "=================================================="
echo ""

# Function to check service
check_service() {
    curl -f -s "${API_ENDPOINT}/tags" > /dev/null 2>&1
}

# Wait for service to be ready
echo "Waiting for service to be available..."
WAITED=0

while [ $WAITED -lt $MAX_WAIT ]; do
    if check_service; then
        echo -e "${GREEN}✓ Service is available${NC}"
        break
    fi

    if [ $WAITED -eq 0 ]; then
        echo -e "${YELLOW}⚠ Service not immediately available, waiting...${NC}"
    fi

    sleep $RETRY_INTERVAL
    WAITED=$((WAITED + RETRY_INTERVAL))
    printf "."
done

echo ""

if ! check_service; then
    echo -e "${RED}✗ Service is not available after ${MAX_WAIT}s${NC}"
    echo ""
    echo "The Shared Ollama Service must be running for tests."
    echo ""
    echo "To start the service:"
    echo "  1. ./scripts/core/start.sh (recommended - REST API manages Ollama)"
    echo "  2. ollama serve (manual start)"
    echo "  3. brew services start ollama (if using Homebrew)"
    echo ""
    echo "Or skip service check with: SKIP_OLLAMA_CHECK=true"
    exit 1
fi

# Verify required models are available
echo "Checking required models..."

MODELS_JSON=$(curl -s "${API_ENDPOINT}/tags" 2>/dev/null || echo "")
MODELS_LIST=$(echo "$MODELS_JSON" | jq -r '.models[].name' 2>/dev/null || echo "")

# Fallback to ollama list
if [ -z "$MODELS_LIST" ] || [ "$MODELS_LIST" = "" ]; then
    if command -v ollama &> /dev/null; then
        MODELS_LIST=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}' || echo "")
    fi
fi

# Function to test if a model is actually usable (not just listed)
test_model_usable() {
    local model=$1
    TEST_PROMPT="{\"model\": \"${model}\", \"prompt\": \"OK\", \"stream\": false}"
    
    TEST_RESPONSE=$(curl -s -X POST "${API_ENDPOINT}/generate" \
        -H "Content-Type: application/json" \
        -d "$TEST_PROMPT" 2>/dev/null)
    
    # Check for error field (model corrupted, missing files, etc.)
    if echo "$TEST_RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
        return 1  # Model is not usable
    # Check for successful response
    elif echo "$TEST_RESPONSE" | jq -e '.response' > /dev/null 2>&1; then
        return 0  # Model is usable
    else
        return 1  # Invalid response
    fi
}

MISSING_MODELS=()
CORRUPTED_MODELS=()
for model in "${REQUIRED_MODELS[@]}"; do
    if echo "$MODELS_LIST" | grep -q "^${model}$"; then
        # Model is listed - test if it's actually usable
        if test_model_usable "$model"; then
            echo -e "${GREEN}  ✓ ${model} (available and usable)${NC}"
        else
            echo -e "${RED}  ✗ ${model} (listed but CORRUPTED)${NC}"
            CORRUPTED_MODELS+=("$model")
        fi
    else
        echo -e "${YELLOW}  ⚠ ${model} not found${NC}"
        MISSING_MODELS+=("$model")
    fi
done

if [ ${#CORRUPTED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}✗ Some models are corrupted or missing files${NC}"
    echo "Models may appear in list but cannot generate responses."
    echo ""
    echo "To fix corrupted models:"
    for model in "${CORRUPTED_MODELS[@]}"; do
        echo "  ollama rm ${model} && ollama pull ${model}"
    done
    exit 1
fi

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠ Some models are missing${NC}"
    echo "Download with: ./scripts/preload_models.sh"
    echo ""
    echo "Continuing anyway (models will be downloaded on first use)..."
fi

echo ""
echo "=================================================="
echo -e "${GREEN}✓ Service check complete - ready for CI/CD${NC}"
echo ""
echo "Service URL: $OLLAMA_URL"

