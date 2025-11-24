#!/bin/bash

# Shared Ollama Service Health Check Script
# This script verifies that Ollama is running and all required models are available

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
API_ENDPOINT="${OLLAMA_URL}/api"

# Load model configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/model_config.sh"
load_model_config

# Status tracking
CHECK_PASSED=0
CHECK_FAILED=0

echo "🔍 Shared Ollama Service Health Check"
echo "======================================"
echo "URL: $OLLAMA_URL"
echo ""

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
        ((CHECK_PASSED++))
    else
        echo -e "${RED}✗${NC} $2"
        ((CHECK_FAILED++))
    fi
}

# Check 1: Ollama service is running
echo "Checking Ollama service status..."
if curl -f -s "${API_ENDPOINT}/tags" > /dev/null 2>&1; then
    print_status 0 "Ollama service is running"
else
    print_status 1 "Ollama service is not accessible"
    echo ""
    echo "Attempting to start service with MPS/Metal optimization..."

    # Start native Ollama service
    if command -v ollama &> /dev/null; then
        echo "Starting Ollama service with optimizations..."
        export OLLAMA_METAL=1
        export OLLAMA_NUM_GPU=-1
        ollama serve > /dev/null 2>&1 &
        echo "Waiting 3 seconds for service to start..."
        sleep 3

        if curl -f -s "${API_ENDPOINT}/tags" > /dev/null 2>&1; then
            print_status 0 "Ollama service started successfully"
        else
            print_status 1 "Failed to start Ollama service"
            echo "Try: ollama serve"
            exit 1
        fi
    else
        echo "Ollama not found. Please install: https://ollama.ai/download"
        echo "Or run: ./scripts/install_native.sh"
        exit 1
    fi
fi

# Check 2: Get list of available models
echo ""
echo "Checking available models..."
MODELS_JSON=$(curl -s "${API_ENDPOINT}/tags")
MODELS_LIST=$(echo "$MODELS_JSON" | jq -r '.models[].name' 2>/dev/null || echo "")

if [ -z "$MODELS_LIST" ]; then
    print_status 1 "No models found"
else
    echo ""
    echo "Available models:"
    echo "$MODELS_LIST" | while read -r model; do
        echo "  - $model"
    done
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

# Check 3: Verify required models exist AND are usable
echo ""
echo "Checking required models (verifying files are intact)..."
for model in "${REQUIRED_MODELS[@]}"; do
    if echo "$MODELS_LIST" | grep -q "^${model}$"; then
        # Model is listed - now test if it's actually usable
        if test_model_usable "$model"; then
            print_status 0 "Model '${model}' is available and usable"
        else
            print_status 1 "Model '${model}' is listed but CORRUPTED or missing files"
            echo "    Model appears in list but cannot generate. Try: ollama rm ${model} && ollama pull ${model}"
        fi
    else
        print_status 1 "Model '${model}' is NOT available"
        echo "    Pull with: ollama pull ${model}"
    fi
done

# Summary
echo ""
echo "======================================"
echo "Summary"
echo "======================================"
echo -e "${GREEN}Passed: ${CHECK_PASSED}${NC}"
echo -e "${RED}Failed: ${CHECK_FAILED}${NC}"

if [ $CHECK_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All checks passed! Ollama service is ready.${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some checks failed. Please review above.${NC}"
    exit 1
fi
