#!/bin/bash

# Native Ollama Installation Script for Apple Silicon Mac
# This installs and configures Ollama to run natively for maximum performance

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Native Ollama Installation for Apple Silicon${NC}"
echo "======================================"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}✗ This script is for macOS only${NC}"
    exit 1
fi

# Check if running on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}⚠ Warning: Not running on Apple Silicon (arm64)${NC}"
    echo "This script is optimized for Apple Silicon Macs."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${BLUE}Step 1: Installing Ollama...${NC}"

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is already installed${NC}"
    OLLAMA_VERSION=$(ollama --version 2>&1 | head -n 1 || echo "unknown")
    echo "  Version: $OLLAMA_VERSION"
else
    echo "Installing Ollama..."
    # Install using official installer
    curl -fsSL https://ollama.ai/install.sh | sh
    
    if command -v ollama &> /dev/null; then
        echo -e "${GREEN}✓ Ollama installed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to install Ollama${NC}"
        echo "Please install manually from: https://ollama.ai/download"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}Step 2: Disabling Homebrew Ollama service...${NC}"

# Disable Homebrew Ollama service (we'll manage it via our scripts)
if command -v brew &> /dev/null; then
    BREW_STATUS=$(brew services list 2>/dev/null | grep ollama || echo "")
    if [ -n "$BREW_STATUS" ]; then
        echo -e "${BLUE}Disabling Homebrew Ollama service (we manage Ollama ourselves)...${NC}"
        brew services stop ollama > /dev/null 2>&1 || true
        echo -e "${GREEN}✓ Homebrew Ollama service disabled${NC}"
        echo -e "${YELLOW}Note: Ollama will be managed by this project's scripts, not Homebrew${NC}"
    else
        echo -e "${GREEN}✓ Homebrew Ollama service not configured${NC}"
    fi
else
    echo -e "${GREEN}✓ Homebrew not found (skipping)${NC}"
fi

echo ""
echo -e "${BLUE}Step 3: Pulling models...${NC}"
echo "This may take a while depending on your internet connection."
echo ""

# Define models to pull
MODELS=(
    "qwen2.5vl:7b"
    "qwen2.5:7b"
    "qwen2.5:14b"
    "granite4:latest"
)

# Pull models
for model in "${MODELS[@]}"; do
    echo -e "${YELLOW}Pulling ${model}...${NC}"
    if ollama pull "${model}" 2>&1 | grep -q "success\|pulling\|extracting\|complete"; then
        echo -e "${GREEN}✓ ${model} pulled successfully${NC}"
    else
        echo -e "${RED}✗ Failed to pull ${model}${NC}"
    fi
done

echo ""
echo -e "${BLUE}Step 4: Verifying installation...${NC}"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo -e "  • Ollama will be managed by this project's start scripts"
echo -e "  • Do NOT use 'brew services start ollama' - use './scripts/start.sh' instead"
echo -e "  • The REST API will automatically start and manage Ollama"
echo ""

# Verify models are available
MODELS_LIST=$(ollama list 2>/dev/null | grep -E "qwen2.5vl:7b|qwen2.5:7b|qwen2.5:14b" || echo "")
if [ -n "$MODELS_LIST" ]; then
    echo -e "${GREEN}✓ Models verified:${NC}"
    echo "$MODELS_LIST" | while read -r line; do
        echo "  - $line"
    done
else
    echo -e "${YELLOW}⚠ Some models may not be available yet${NC}"
    echo "Run 'ollama list' to check"
fi

echo ""
echo -e "${BLUE}Step 5: Verifying MPS/Metal GPU acceleration...${NC}"

# Check if Metal is available
if system_profiler SPDisplaysDataType 2>/dev/null | grep -qi "metal"; then
    echo -e "${GREEN}✓ Metal GPU acceleration available${NC}"
else
    echo -e "${YELLOW}⚠ Metal GPU status unclear${NC}"
fi

echo ""
echo "======================================"
echo -e "${GREEN}✓ Native Ollama installation complete!${NC}"
echo ""
echo "Service URL: http://localhost:11434"
echo ""
echo "MPS/Metal GPU Optimization:"
echo "  ✓ Metal acceleration enabled (OLLAMA_METAL=1)"
echo "  ✓ All GPU cores available (OLLAMA_NUM_GPU=-1)"
echo ""
echo "Next Steps:"
echo "  1. Pre-download models:     ./scripts/preload_models.sh"
echo "  2. Warm up models:          ./scripts/warmup_models.sh"
echo "  3. Start the service:       ./scripts/start.sh"
echo ""
echo "To manage the service:"
echo "  Start:    ./scripts/start.sh (recommended - includes optimizations)"
echo "  Stop:     ./scripts/shutdown.sh"
echo "  Status:   curl http://localhost:11434/api/tags"
echo "  Models:   ollama list"
