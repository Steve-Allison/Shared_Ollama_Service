#!/bin/bash

# Start Ollama Service with Apple Silicon MPS/Metal Optimizations
# Explicitly enables Metal acceleration and GPU cores for maximum performance

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

echo -e "${BLUE}🚀 Starting Ollama Service with MPS/Metal Optimization${NC}"
echo "=================================================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}✗ Ollama is not installed${NC}"
    echo "Please install Ollama first: ./scripts/install_native.sh"
    exit 1
fi

# Check if service is already running
if curl -f -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Ollama service is already running${NC}"
    echo "To restart, stop it first: ./scripts/shutdown.sh"
    exit 0
fi

# Set Apple Silicon MPS/Metal optimizations
export OLLAMA_METAL=1          # Explicitly enable Metal acceleration (MPS)
export OLLAMA_NUM_GPU=-1       # Use all available Metal GPU cores (-1 = all)

# Optional: Set other optimizations (can be overridden by env vars)
export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-5m}"

echo -e "${BLUE}Configuration:${NC}"
echo "  ✓ Metal/MPS GPU: Enabled (OLLAMA_METAL=1)"
echo "  ✓ GPU Cores: All available (OLLAMA_NUM_GPU=-1)"
echo "  ✓ Host: ${OLLAMA_HOST}"
echo "  ✓ Keep Alive: ${OLLAMA_KEEP_ALIVE}"
echo "  ✓ Logs: ${LOG_DIR}/"
echo ""

# Start Ollama service with optimizations
echo -e "${BLUE}Starting Ollama service...${NC}"
nohup ollama serve > "$LOG_DIR/ollama.log" 2> "$LOG_DIR/ollama.error.log" &

# Wait for service to start
echo -e "${BLUE}Waiting for service to start...${NC}"
sleep 3

# Verify service is running
RETRY_COUNT=0
MAX_RETRIES=5
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama service is running${NC}"
        echo ""
        echo "Service URL: http://localhost:11434"
        echo "Logs: tail -f $LOG_DIR/ollama.log"
        echo ""
        
        # Show available models
        MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | tr '\n' ' ' || echo "none")
        if [ -n "$MODELS" ] && [ "$MODELS" != "none" ]; then
            echo "Available models: $MODELS"
        else
            echo "No models downloaded yet. Run: ollama pull qwen2.5vl:7b"
        fi
        exit 0
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep 1
done

echo -e "${RED}✗ Failed to start Ollama service${NC}"
echo "Check logs: tail -f $LOG_DIR/ollama.error.log"
exit 1

