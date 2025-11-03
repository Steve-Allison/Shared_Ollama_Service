#!/bin/bash

# Pre-load models script - Ensures models are downloaded before first use
# This should be run after installation to verify/download all required models

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📥 Pre-downloading Ollama models${NC}"
echo "======================================"
echo ""

# Required models
MODELS=(
    "qwen2.5vl:7b"
    "qwen2.5:14b"
)

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}✗ Ollama is not installed${NC}"
    echo "Install Ollama first: ./scripts/install_native.sh"
    exit 1
fi

echo -e "${GREEN}✓ Ollama is installed${NC}"
echo ""

# Check if Ollama service is running
if ! curl -f -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

echo "Checking and downloading models..."
echo ""

DOWNLOADED=0
ALREADY_EXIST=0
FAILED=0

for model in "${MODELS[@]}"; do
    echo -e "${YELLOW}Processing ${model}...${NC}"
    
    # Check if model already exists
    if ollama list 2>/dev/null | grep -q "^${model}"; then
        echo -e "${GREEN}  ✓ ${model} already downloaded${NC}"
        ((ALREADY_EXIST++))
    else
        echo -e "${BLUE}  ↓ Downloading ${model}...${NC}"
        if ollama pull "${model}" > /dev/null 2>&1; then
            echo -e "${GREEN}  ✓ ${model} downloaded successfully${NC}"
            ((DOWNLOADED++))
        else
            echo -e "${RED}  ✗ Failed to download ${model}${NC}"
            ((FAILED++))
        fi
    fi
    echo ""
done

echo "======================================"
echo -e "${GREEN}✓ Pre-download complete!${NC}"
echo ""
echo "Summary:"
echo "  Downloaded: ${DOWNLOADED}"
echo "  Already exist: ${ALREADY_EXIST}"
echo "  Failed: ${FAILED}"
echo ""
echo "All models are now available locally for fast access."
echo ""
echo "Next steps:"
echo "  - Run warm-up script to preload models: ./scripts/warmup_models.sh"
echo "  - Or start using models - first request will load them into memory"

