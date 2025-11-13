#!/bin/bash

# Start Shared Ollama Service REST API
# The REST API manages Ollama internally - no need to start Ollama separately
# System-specific optimizations are automatically detected and applied

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

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
API_PORT="${API_PORT:-8000}"
if curl -f -s "http://localhost:$API_PORT/api/v1/health" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ REST API is already running on port $API_PORT${NC}"
    echo "To restart, stop it first: ./scripts/shutdown.sh"
    exit 0
fi

echo -e "${BLUE}Configuration:${NC}"
echo "  ✓ REST API Port: ${API_PORT}"
echo "  ✓ Logs: ${LOG_DIR}/"
echo "  ✓ Ollama: Managed internally by REST API"
echo ""

# Start REST API server (manages Ollama internally)
echo -e "${BLUE}🚀 Starting REST API server...${NC}"
API_HOST="${API_HOST:-0.0.0.0}"

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        echo -e "${BLUE}Activating virtual environment...${NC}"
        source "$PROJECT_ROOT/.venv/bin/activate"
    else
        echo -e "${YELLOW}⚠ No virtual environment found. Using system Python.${NC}"
    fi
fi

# Set environment variables for API
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Start API server (will manage Ollama internally)
cd "$PROJECT_ROOT"
echo ""
echo -e "${GRAY}The REST API will automatically start and manage Ollama internally${NC}"
echo -e "${GRAY}System optimizations will be auto-detected and applied${NC}"
echo ""

# Start API server in foreground (so we can monitor it)
exec uvicorn shared_ollama.api.server:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --log-level info


