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

# Check for restart flag
RESTART=false
if [ "$1" = "--restart" ] || [ "$1" = "-r" ]; then
    RESTART=true
fi

# Check if REST API is already running
API_PORT="${API_PORT:-8000}"
if curl -f -s "http://localhost:$API_PORT/api/v1/health" > /dev/null 2>&1; then
    if [ "$RESTART" = true ]; then
        echo -e "${YELLOW}⚠ REST API is already running. Stopping it first...${NC}"
        "$SCRIPT_DIR/shutdown.sh" || true
        echo "Waiting 2 seconds for port to be released..."
        sleep 2
    else
        echo -e "${YELLOW}⚠ REST API is already running on port $API_PORT${NC}"
        echo "To restart, run: $0 --restart"
        echo "Or stop it first: ./scripts/shutdown.sh"
        exit 0
    fi
fi

echo -e "${BLUE}Configuration:${NC}"
echo "  ✓ REST API Port: ${API_PORT}"
echo "  ✓ Logs: ${LOG_DIR}/"
echo "  ✓ Ollama: Managed internally by REST API"
echo ""

# Start REST API server (manages Ollama internally)
echo -e "${BLUE}🚀 Starting REST API server...${NC}"
API_HOST="${API_HOST:-0.0.0.0}"

# Determine Python and uvicorn paths
UVICORN_CMD="uvicorn"
PYTHON_CMD="python3"

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        echo -e "${BLUE}Using virtual environment at .venv${NC}"
        UVICORN_CMD="$PROJECT_ROOT/.venv/bin/uvicorn"
        PYTHON_CMD="$PROJECT_ROOT/.venv/bin/python"
        # Verify uvicorn exists in venv
        if [ ! -f "$UVICORN_CMD" ]; then
            echo -e "${RED}✗ uvicorn not found in virtual environment${NC}"
            echo "Please install dependencies: pip install -e \".[dev]\""
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠ No virtual environment found. Using system Python.${NC}"
        # Check if uvicorn is available in system
        if ! command -v uvicorn &> /dev/null; then
            echo -e "${RED}✗ uvicorn not found. Please install dependencies or create a virtual environment.${NC}"
            exit 1
        fi
    fi
else
    # Already in a venv, use it
    UVICORN_CMD="uvicorn"
    PYTHON_CMD="python"
fi

# Set environment variables for API
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Start API server (will manage Ollama internally)
cd "$PROJECT_ROOT"
echo ""
echo -e "${GRAY}The REST API will automatically start and manage Ollama internally${NC}"
echo -e "${GRAY}System optimizations will be auto-detected and applied${NC}"
echo ""

# Start API server in background so we can run warmup
echo -e "${GREEN}Starting uvicorn server...${NC}"
echo -e "${GRAY}Command: $UVICORN_CMD shared_ollama.api.server:app --host $API_HOST --port $API_PORT${NC}"
echo ""
echo -e "${GRAY}API will be available at: http://${API_HOST}:${API_PORT}${NC}"
echo -e "${GRAY}API docs: http://${API_HOST}:${API_PORT}/docs${NC}"
echo ""

# Start the server in background
"$UVICORN_CMD" shared_ollama.api.server:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --log-level info &

UVICORN_PID=$!

# Wait for API server to be ready
echo -e "${BLUE}Waiting for API server to start...${NC}"
for i in {1..30}; do
    if curl -f -s "http://localhost:$API_PORT/api/v1/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API server is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${YELLOW}⚠ API server may not be ready yet, continuing anyway...${NC}"
    fi
    sleep 1
done

# Warm up models in background
if [ -f "$SCRIPT_DIR/warmup_models.sh" ]; then
    echo ""
    echo -e "${BLUE}🔥 Warming up models...${NC}"
    "$SCRIPT_DIR/warmup_models.sh" > /dev/null 2>&1 &
    WARMUP_PID=$!
    echo -e "${GRAY}Model warmup started in background (PID: $WARMUP_PID)${NC}"
    echo -e "${GRAY}Models will be preloaded for faster first requests${NC}"
else
    echo -e "${YELLOW}⚠ Warmup script not found at $SCRIPT_DIR/warmup_models.sh${NC}"
fi

echo ""
echo -e "${GREEN}✓ Service started successfully${NC}"
echo -e "${GRAY}API server PID: $UVICORN_PID${NC}"
echo ""

# Wait for uvicorn process (foreground)
wait $UVICORN_PID


