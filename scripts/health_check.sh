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

# Required models
REQUIRED_MODELS=("llama3.1:8b" "mistral")

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
    echo "Attempting to start service..."
    
    # Try to start docker-compose if available
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d
        echo "Waiting 10 seconds for service to start..."
        sleep 10
        
        if curl -f -s "${API_ENDPOINT}/tags" > /dev/null 2>&1; then
            print_status 0 "Ollama service started successfully"
        else
            print_status 1 "Failed to start Ollama service"
            echo ""
            echo "Manual start: docker-compose up -d"
            exit 1
        fi
    else
        echo "docker-compose not found. Please start Ollama manually."
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

# Check 3: Verify required models
echo ""
echo "Checking required models..."
for model in "${REQUIRED_MODELS[@]}"; do
    if echo "$MODELS_LIST" | grep -q "^${model}$"; then
        print_status 0 "Model '${model}' is available"
    else
        print_status 1 "Model '${model}' is NOT available"
        echo "    Pull with: docker-compose exec ollama ollama pull ${model}"
    fi
done

# Check 4: Test model generation (quick smoke test)
if echo "$MODELS_LIST" | grep -q "llama3.1:8b"; then
    echo ""
    echo "Testing model generation with llama3.1:8b..."
    TEST_PROMPT='{"model": "llama3.1:8b", "prompt": "Say hello", "stream": false}'
    
    if curl -s -X POST "${API_ENDPOINT}/generate" \
        -H "Content-Type: application/json" \
        -d "$TEST_PROMPT" | jq -r '.response' > /dev/null 2>&1; then
        print_status 0 "Model generation test passed"
    else
        print_status 1 "Model generation test failed"
    fi
fi

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
