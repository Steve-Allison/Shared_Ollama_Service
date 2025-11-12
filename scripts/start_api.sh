#!/bin/bash

# Start the Shared Ollama Service REST API
# This provides a language-agnostic REST API wrapper around the Ollama client

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default configuration
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

echo -e "${BLUE}🚀 Starting Shared Ollama Service REST API${NC}"
echo "=================================================="
echo ""

# Check if Ollama service is running
if ! curl -f -s "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
    echo -e "${RED}✗ Ollama service is not running at ${OLLAMA_BASE_URL}${NC}"
    echo "Start Ollama first: ./scripts/start.sh"
    exit 1
fi

echo -e "${GREEN}✓ Ollama service is running${NC}"
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        echo -e "${BLUE}Activating virtual environment...${NC}"
        source "$PROJECT_ROOT/.venv/bin/activate"
    else
        echo -e "${YELLOW}⚠ No virtual environment found. Using system Python.${NC}"
    fi
fi

# Set environment variables
export OLLAMA_BASE_URL
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

echo -e "${BLUE}Configuration:${NC}"
echo "  ✓ API Host: ${API_HOST}"
echo "  ✓ API Port: ${API_PORT}"
echo "  ✓ Ollama URL: ${OLLAMA_BASE_URL}"
echo ""

echo -e "${BLUE}Starting API server...${NC}"
echo ""
echo -e "${GRAY}API Documentation: http://${API_HOST}:${API_PORT}/api/docs${NC}"
echo -e "${GRAY}Health Check: http://${API_HOST}:${API_PORT}/api/v1/health${NC}"
echo ""

cd "$PROJECT_ROOT"

# Start uvicorn server
exec uvicorn shared_ollama.api.server:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --reload \
    --log-level info

