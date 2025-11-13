#!/bin/bash

# Start the Shared Ollama Service REST API
# The REST API manages Ollama internally - no need to start Ollama separately

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

echo -e "${BLUE}🚀 Starting Shared Ollama Service REST API${NC}"
echo "=================================================="
echo ""
echo -e "${GRAY}Note: The REST API will automatically start and manage Ollama internally${NC}"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}✗ Ollama is not installed${NC}"
    echo "Please install Ollama first: ./scripts/install_native.sh"
    exit 1
fi

# Check if REST API is already running
if curl -f -s "http://localhost:$API_PORT/api/v1/health" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ REST API is already running on port $API_PORT${NC}"
    echo "To restart, stop it first: ./scripts/shutdown.sh"
    exit 0
fi

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
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

echo -e "${BLUE}Configuration:${NC}"
echo "  ✓ API Host: ${API_HOST}"
echo "  ✓ API Port: ${API_PORT}"
echo "  ✓ Ollama: Managed internally by REST API"
echo ""

echo -e "${BLUE}Starting API server...${NC}"
echo ""
echo -e "${GRAY}API Documentation: http://${API_HOST}:${API_PORT}/api/docs${NC}"
echo -e "${GRAY}Health Check: http://${API_HOST}:${API_PORT}/api/v1/health${NC}"
echo ""
echo -e "${GRAY}The REST API will automatically start and manage Ollama internally${NC}"
echo -e "${GRAY}System optimizations will be auto-detected and applied${NC}"
echo ""

cd "$PROJECT_ROOT"

# Start uvicorn server (auto-reload disabled for production stability)
exec uvicorn shared_ollama.api.server:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --log-level info

