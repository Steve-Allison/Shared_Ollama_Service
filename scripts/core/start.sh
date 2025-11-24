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
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

echo -e "${BLUE}🚀 Starting Shared Ollama Service REST API${NC}"
echo "=================================================="
echo ""
echo -e "${GRAY}Note: The REST API will automatically start and manage Ollama internally${NC}"
echo ""

# Check for flags
RESTART=false
SKIP_VERIFY=false
FOREGROUND=false
for arg in "$@"; do
    case $arg in
        --restart|-r)
            RESTART=true
            ;;
        --skip-verify|--no-verify)
            SKIP_VERIFY=true
            ;;
        --foreground)
            FOREGROUND=true
            ;;
    esac
done

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

# ============================================================================
# Step 0: Verify Setup and Generate Optimal Configuration
# ============================================================================
if [ "$SKIP_VERIFY" = false ]; then
    echo -e "${BLUE}[Setup]${NC} Verifying configuration and setup..."
    if [ -f "$SCRIPT_DIR/verify_setup.sh" ]; then
        # Run verify_setup.sh - it will auto-detect hardware profile if needed
        # Don't fail the start process if verification has minor issues
        # (verify_setup.sh will still auto-fix what it can)
        VERIFY_OUTPUT=$(bash "$SCRIPT_DIR/verify_setup.sh" 2>&1) || VERIFY_EXIT=$?
        
        # Check if config was generated or already exists
        if echo "$VERIFY_OUTPUT" | grep -q "Optimal configuration generated\|Configuration file exists"; then
            echo -e "${GREEN}✓ Configuration verified/generated${NC}"
        fi
        
        # Show summary but don't block startup
        if echo "$VERIFY_OUTPUT" | grep -q "All checks passed"; then
            echo -e "${GREEN}✓ Setup verification passed${NC}"
        elif [ "${VERIFY_EXIT:-0}" -ne 0 ]; then
            echo -e "${YELLOW}⚠ Setup verification found issues (continuing anyway)${NC}"
            echo -e "${GRAY}Run './scripts/verify_setup.sh' manually for full details${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ verify_setup.sh not found - skipping verification${NC}"
    fi
    echo ""
else
    echo -e "${GRAY}⏭ Skipping setup verification (--skip-verify flag)${NC}"
    echo ""
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
        echo "Or stop it first: ./scripts/core/shutdown.sh"
        exit 0
    fi
fi

echo -e "${BLUE}Configuration:${NC}"
echo "  ✓ REST API Port: ${API_PORT}"
echo "  ✓ Logs: ${LOG_DIR}/"
echo "  ✓ Ollama: Managed internally by REST API"
if [ -f "$PROJECT_ROOT/config.toml" ]; then
    echo "  ✓ Configuration: config.toml loaded"
else
    echo "  ⚠ Configuration: Using defaults (no config.toml file)"
fi
echo ""

# Start REST API server (manages Ollama internally)
echo -e "${BLUE}🚀 Starting REST API server...${NC}"
API_HOST="${API_HOST:-0.0.0.0}"

# Determine Python and gunicorn paths
GUNICORN_CMD="gunicorn"
PYTHON_CMD="python3"

# Calculate optimal worker count (2 * CPU cores + 1)
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo "4")
WORKERS=$((CPU_CORES * 2 + 1))
# Cap at 8 workers for stability
if [ "$WORKERS" -gt 8 ]; then
    WORKERS=8
fi

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        echo -e "${BLUE}Using virtual environment at .venv${NC}"
        GUNICORN_CMD="$PROJECT_ROOT/.venv/bin/gunicorn"
        PYTHON_CMD="$PROJECT_ROOT/.venv/bin/python"
        # Verify gunicorn exists in venv
        if [ ! -f "$GUNICORN_CMD" ]; then
            echo -e "${RED}✗ gunicorn not found in virtual environment${NC}"
            echo "Please install dependencies: pip install -e \".[dev]\""
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠ No virtual environment found. Using system Python.${NC}"
        # Check if gunicorn is available in system
        if ! command -v gunicorn &> /dev/null; then
            echo -e "${RED}✗ gunicorn not found. Please install dependencies or create a virtual environment.${NC}"
            exit 1
        fi
    fi
else
    # Already in a venv, use it
    GUNICORN_CMD="gunicorn"
    PYTHON_CMD="python"
fi

# Set environment variables for API
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Change to project root directory
cd "$PROJECT_ROOT"
echo ""
echo -e "${GRAY}The REST API will automatically start and manage Ollama internally${NC}"
echo -e "${GRAY}System optimizations will be auto-detected and applied${NC}"
echo ""

# Start API server with gunicorn (production-grade with auto-restart)
echo -e "${GREEN}Starting gunicorn server with $WORKERS workers...${NC}"
echo -e "${GRAY}Command: $GUNICORN_CMD shared_ollama.api.server:app --workers $WORKERS --worker-class uvicorn.workers.UvicornWorker --bind $API_HOST:$API_PORT${NC}"
echo ""
echo -e "${GRAY}API will be available at: http://${API_HOST}:${API_PORT}${NC}"
echo -e "${GRAY}API docs: http://${API_HOST}:${API_PORT}/api/docs${NC}"
echo -e "${GRAY}Workers: $WORKERS (auto-restart on crash)${NC}"
echo ""

# Ensure logs directory exists
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Log file paths
API_LOG="$LOG_DIR/api.log"
API_ERROR_LOG="$LOG_DIR/api.error.log"

# Verify Python can import the module before starting
echo -e "${BLUE}Verifying Python dependencies...${NC}"
IMPORT_ERROR=$("$PYTHON_CMD" -c "import sys; sys.path.insert(0, '$PROJECT_ROOT/src'); from shared_ollama.api.server import app" 2>&1)
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to import shared_ollama.api.server${NC}"
    echo -e "${YELLOW}Error details:${NC}"
    echo "$IMPORT_ERROR" | head -5 | sed 's/^/  /'
    echo ""
    echo -e "${YELLOW}Please ensure dependencies are installed:${NC}"
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        echo "  source .venv/bin/activate"
        echo "  pip install -e \".[dev]\""
    else
        echo "  pip install -e \".[dev]\""
    fi
    exit 1
fi
echo -e "${GREEN}✓ Dependencies verified${NC}"
echo ""

# Start gunicorn server in background with log redirection
# Gunicorn provides built-in process management and auto-restart
# Access log format includes timestamp, IP, method, path, status, response time
ACCESS_LOG_FORMAT='%(t)s %(h)s:%(p)s - "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
nohup "$GUNICORN_CMD" shared_ollama.api.server:app \
    --workers "$WORKERS" \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind "$API_HOST:$API_PORT" \
    --access-logfile "$API_LOG" \
    --access-logformat "$ACCESS_LOG_FORMAT" \
    --error-logfile "$API_ERROR_LOG" \
    --log-level info \
    --timeout 120 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    > /dev/null 2>&1 &

GUNICORN_PID=$!

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
WARMUP_SCRIPT="$PROJECT_ROOT/scripts/maintenance/warmup_models.sh"
if [ -f "$WARMUP_SCRIPT" ]; then
    echo ""
    echo -e "${BLUE}🔥 Warming up models...${NC}"
    "$WARMUP_SCRIPT" > /dev/null 2>&1 &
    WARMUP_PID=$!
    echo -e "${GRAY}Model warmup started in background (PID: $WARMUP_PID)${NC}"
    echo -e "${GRAY}Models will be preloaded for faster first requests${NC}"
else
    echo -e "${YELLOW}⚠ Warmup script not found at $WARMUP_SCRIPT${NC}"
fi

echo ""
echo -e "${GREEN}✓ Service started successfully${NC}"
echo -e "${GRAY}Gunicorn master PID: $GUNICORN_PID${NC}"
echo -e "${GRAY}Workers: $WORKERS (auto-restart on crash)${NC}"
echo -e "${GRAY}Logs: $API_LOG${NC}"
echo -e "${GRAY}Errors: $API_ERROR_LOG${NC}"
echo ""

# Save PID for reference (gunicorn manages workers internally)
echo "$GUNICORN_PID" > "$PROJECT_ROOT/.api.pid"

echo -e "${CYAN}Service is running with gunicorn.${NC}"
echo -e "${CYAN}Gunicorn provides built-in process management and auto-restart.${NC}"
echo -e "${CYAN}To stop the service, run: ./scripts/core/shutdown.sh${NC}"
echo -e "${CYAN}To view logs: tail -f $API_LOG${NC}"
echo ""

if [ "$FOREGROUND" = true ]; then
    echo -e "${GRAY}Waiting for gunicorn master process (--foreground)${NC}"
    wait $GUNICORN_PID
    echo -e "${YELLOW}⚠ Gunicorn master process exited${NC}"
else
    echo -e "${GRAY}Tip: run with --foreground to block and monitor the master process${NC}"
fi


