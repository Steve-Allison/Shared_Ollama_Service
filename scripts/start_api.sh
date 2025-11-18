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

# Stop and disable Homebrew Ollama service (we manage it ourselves)
if command -v brew &> /dev/null; then
    BREW_STATUS=$(brew services list 2>/dev/null | grep ollama || echo "")
    if [ -n "$BREW_STATUS" ]; then
        echo -e "${BLUE}Disabling Homebrew Ollama service (we manage Ollama ourselves)...${NC}"
        # Stop if running
        brew services stop ollama > /dev/null 2>&1 || true
        sleep 1
        # Ensure it's disabled (won't auto-start on boot)
        # Note: brew services stop also disables auto-start
        echo -e "${GREEN}✓ Homebrew Ollama service disabled${NC}"
    fi
fi

# Stop any launchd Ollama service
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/com.ollama.service.plist"
if [ -f "$LAUNCHD_PLIST" ]; then
    if launchctl list 2>/dev/null | grep -q "com.ollama.service"; then
        echo -e "${BLUE}Stopping launchd Ollama service...${NC}"
        launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true
        sleep 1
    fi
fi

# Kill any existing ollama serve processes
OLLAMA_PIDS=$(ps aux | grep -i "[o]llama serve" | awk '{print $2}' 2>/dev/null || true)
if [ -n "$OLLAMA_PIDS" ]; then
    echo -e "${BLUE}Stopping existing Ollama processes...${NC}"
    for pid in $OLLAMA_PIDS; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
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

# Ensure logs directory exists
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Log file paths
API_LOG="$LOG_DIR/api.log"
API_ERROR_LOG="$LOG_DIR/api.error.log"

echo -e "${BLUE}Log files:${NC}"
echo "  ✓ API logs: $API_LOG"
echo "  ✓ Error logs: $API_ERROR_LOG"
echo ""

# Start uvicorn server (auto-reload disabled for production stability)
# Redirect stdout to api.log and stderr to api.error.log
exec uvicorn shared_ollama.api.server:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --log-level info \
    > "$API_LOG" 2> "$API_ERROR_LOG"

