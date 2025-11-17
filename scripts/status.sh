#!/bin/bash

# Quick status check script for Shared Ollama Service
# Shows service health, models, and basic metrics

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
API_ENDPOINT="${OLLAMA_URL}/api"

echo -e "${BLUE}📊 Shared Ollama Service Status${NC}"
echo "=================================================="
echo ""

# Check if service is running
echo -e "${BLUE}Service Status:${NC}"
if curl -f -s "${API_ENDPOINT}/tags" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Service is running${NC}"
    echo "  URL: $OLLAMA_URL"

    # Get service info
    if command -v ollama &> /dev/null; then
        VERSION=$(ollama --version 2>&1 | head -n 1 || echo "unknown")
        echo "  Version: $VERSION"
    fi
else
    echo -e "${RED}✗ Service is not running${NC}"
    echo ""
    echo "To start the service:"
    echo "  - ./scripts/setup_launchd.sh (if using launchd)"
    echo "  - brew services start ollama (if using Homebrew)"
    echo "  - ollama serve (manual start)"
    exit 1
fi

echo ""

# List models
echo -e "${BLUE}Available Models:${NC}"
MODELS_JSON=$(curl -s "${API_ENDPOINT}/tags" 2>/dev/null || echo "")
MODELS_LIST=$(echo "$MODELS_JSON" | jq -r '.models[].name' 2>/dev/null || echo "")

if [ -z "$MODELS_LIST" ] || [ "$MODELS_LIST" = "" ]; then
    # Fallback to ollama list
    if command -v ollama &> /dev/null; then
        MODELS_LIST=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}' || echo "")
    fi
fi

if [ -n "$MODELS_LIST" ]; then
    MODEL_COUNT=$(echo "$MODELS_LIST" | wc -l | tr -d ' ')
    echo "  Total: $MODEL_COUNT model(s)"
    echo "$MODELS_LIST" | while read -r model; do
        if [ -n "$model" ]; then
            echo "  - $model"
        fi
    done
else
    echo -e "${YELLOW}  No models found${NC}"
    echo "  Run: ./scripts/preload_models.sh"
fi

echo ""

# Check processes
echo -e "${BLUE}Processes:${NC}"
OLLAMA_PIDS=$(ps aux | grep -i "[o]llama" | wc -l | tr -d ' ' || echo "0")
if [ "$OLLAMA_PIDS" -gt 0 ]; then
    echo "  Running processes: $OLLAMA_PIDS"
    ps aux | grep -i "[o]llama" | grep -v grep | awk '{print "    PID " $2 ": " $11 " " $12}' | head -3
else
    echo "  No processes found"
fi

echo ""

# Memory usage (if available)
if command -v ps &> /dev/null; then
    echo -e "${BLUE}Memory Usage:${NC}"
    OLLAMA_MEM=$(ps aux | grep -i "[o]llama serve" | grep -v grep | awk '{sum+=$6} END {print sum/1024}' || echo "0")
    if [ -n "$OLLAMA_MEM" ] && [ "$OLLAMA_MEM" != "0" ]; then
        printf "  Ollama service: %.1f MB\n" "$OLLAMA_MEM"
    else
        echo "  Not available"
    fi
fi

echo ""

# Storage info
if [ -d ~/.ollama/models ]; then
    echo -e "${BLUE}Storage:${NC}"
    STORAGE_SIZE=$(du -sh ~/.ollama/models 2>/dev/null | awk '{print $1}' || echo "unknown")
    echo "  Models: ~/.ollama/models ($STORAGE_SIZE)"
fi

echo ""

# Quick health test
echo -e "${BLUE}Health Check:${NC}"
if echo "$MODELS_LIST" | grep -q "qwen2.5vl:7b"; then
    echo "  Testing model generation..."
    TEST_RESPONSE=$(curl -s -X POST "${API_ENDPOINT}/generate" \
        -H "Content-Type: application/json" \
        -d '{"model": "qwen2.5vl:7b", "prompt": "Say OK", "stream": false}' \
        2>/dev/null)

    # Check for error field first (model not found, etc.)
    if echo "$TEST_RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
        ERROR_MSG=$(echo "$TEST_RESPONSE" | jq -r '.error' 2>/dev/null || echo "unknown error")
        echo -e "${RED}  ✗ Generation test failed: ${ERROR_MSG}${NC}"
        echo "    Model is listed but not usable. Try: ollama pull qwen2.5vl:7b"
    # Check for successful response
    elif echo "$TEST_RESPONSE" | jq -e '.response' > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Generation test passed${NC}"
    else
        echo -e "${YELLOW}  ⚠ Generation test failed or incomplete${NC}"
        echo "    Response: $(echo "$TEST_RESPONSE" | head -c 100)"
    fi
else
    echo -e "${YELLOW}  ⚠ Cannot test - qwen2.5vl:7b not available${NC}"
fi

echo ""
echo "=================================================="
echo "For more info, see: ./scripts/health_check.sh"

