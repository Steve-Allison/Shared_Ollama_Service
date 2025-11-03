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
    echo "  1. ./scripts/setup_launchd.sh (recommended for local dev)"
    echo "  2. ollama serve (manual start)"
    echo "  3. brew services start ollama (if using Homebrew)"
    echo ""
    echo "Or skip service check with: SKIP_OLLAMA_CHECK=true"
    exit 1
fi

# Verify required models are available
echo "Checking required models..."

REQUIRED_MODELS=("qwen2.5vl:7b" "qwen2.5:14b")
MODELS_JSON=$(curl -s "${API_ENDPOINT}/tags" 2>/dev/null || echo "")
MODELS_LIST=$(echo "$MODELS_JSON" | jq -r '.models[].name' 2>/dev/null || echo "")

# Fallback to ollama list
if [ -z "$MODELS_LIST" ] || [ "$MODELS_LIST" = "" ]; then
    if command -v ollama &> /dev/null; then
        MODELS_LIST=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}' || echo "")
    fi
fi

MISSING_MODELS=()
for model in "${REQUIRED_MODELS[@]}"; do
    if echo "$MODELS_LIST" | grep -q "^${model}$"; then
        echo -e "${GREEN}  ✓ ${model}${NC}"
    else
        echo -e "${YELLOW}  ⚠ ${model} not found${NC}"
        MISSING_MODELS+=("$model")
    fi
done

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠ Some models are missing${NC}"
    echo "Download with: ./scripts/preload_models.sh"
    echo ""
    echo "Continuing anyway (models will be downloaded on first use)..."
fi

# Quick health test
echo ""
echo "Running health test..."
TEST_RESPONSE=$(curl -s -X POST "${API_ENDPOINT}/generate" \
    -H "Content-Type: application/json" \
    -d '{"model": "qwen2.5vl:7b", "prompt": "Say OK", "stream": false}' \
    2>/dev/null || echo "")

if echo "$TEST_RESPONSE" | jq -r '.response' > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Health test passed${NC}"
else
    echo -e "${YELLOW}⚠ Health test incomplete (model may still be loading)${NC}"
fi

echo ""
echo "=================================================="
echo -e "${GREEN}✓ Service check complete - ready for CI/CD${NC}"
echo ""
echo "Service URL: $OLLAMA_URL"

