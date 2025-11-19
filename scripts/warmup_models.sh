#!/bin/bash

# Warm-up script to preload models into memory
# This reduces first-request latency by loading models before they're needed

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
API_ENDPOINT="${OLLAMA_URL}/api"

# Models to warm up (primary models, in priority order)
MODELS=(
    "qwen2.5:7b"
    "qwen2.5vl:7b"
    "qwen2.5:14b"
)

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

# Warm up each model
for model in "${MODELS[@]}"; do
    echo -e "${BLUE}Warming up ${model}...${NC}"
    
    # Send a minimal request to load the model into memory
    # Using keep_alive to keep it loaded for the specified duration
    RESPONSE=$(curl -s -X POST "${API_ENDPOINT}/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model}\",
            \"prompt\": \"Hi\",
            \"stream\": false,
            \"options\": {
                \"num_predict\": 1
            },
            \"keep_alive\": \"${KEEP_ALIVE}\"
        }" 2>/dev/null)
    
    if echo "$RESPONSE" | grep -q "response"; then
        echo -e "${GREEN}  ✓ ${model} warmed up and loaded into memory${NC}"
        echo -e "    Keep-alive: ${KEEP_ALIVE}"
    else
        echo -e "${YELLOW}  ⚠ ${model} warm-up may have failed${NC}"
        echo "    Check if model is available: ollama list"
    fi
    
    echo ""
done

echo "======================================"
echo -e "${GREEN}✓ Model warm-up complete!${NC}"
echo ""
echo "Models are now loaded in memory and ready for fast inference."
echo "They will remain loaded for ${KEEP_ALIVE} (or until unloaded manually)."
echo ""
echo "To check model status:"
echo "  curl ${API_ENDPOINT}/tags"

