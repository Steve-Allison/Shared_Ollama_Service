#!/bin/bash

# Start Ollama Service with System-Specific Optimizations
# Automatically detects Mac system (Apple Silicon/Intel) and adjusts optimization parameters:
# - Apple Silicon: Enables Metal/MPS GPU acceleration with all GPU cores
# - Intel Mac: CPU-only mode (Metal disabled)
# - Auto-detects CPU cores and sets optimal thread count
# - Auto-calculates memory limits based on system RAM

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

# Detect system configuration
echo -e "${BLUE}Detecting system configuration...${NC}"
if [ -f "$PROJECT_ROOT/scripts/detect_system.sh" ]; then
    # Source system detection
    SYSTEM_INFO=$("$PROJECT_ROOT/scripts/detect_system.sh" 2>/dev/null)

    # Parse system information
    ARCH=$(echo "$SYSTEM_INFO" | grep "^ARCH=" | cut -d'=' -f2)
    CHIP_TYPE=$(echo "$SYSTEM_INFO" | grep "^CHIP_TYPE=" | cut -d'=' -f2)
    CHIP_MODEL=$(echo "$SYSTEM_INFO" | grep "^CHIP_MODEL=" | cut -d'=' -f2)
    CPU_CORES=$(echo "$SYSTEM_INFO" | grep "^CPU_CORES=" | cut -d'=' -f2)
    GPU_CORES=$(echo "$SYSTEM_INFO" | grep "^GPU_CORES=" | cut -d'=' -f2)
    TOTAL_RAM_GB=$(echo "$SYSTEM_INFO" | grep "^TOTAL_RAM_GB=" | cut -d'=' -f2)

    echo -e "${GREEN}✓ Detected: ${CHIP_MODEL} (${CHIP_TYPE})${NC}"
    echo -e "${GREEN}✓ CPU Cores: ${CPU_CORES}${NC}"
    if [[ "$CHIP_TYPE" == "Apple Silicon" ]]; then
        echo -e "${GREEN}✓ GPU Cores: ${GPU_CORES}${NC}"
    fi
    echo -e "${GREEN}✓ Total RAM: ${TOTAL_RAM_GB} GB${NC}"
    echo ""
else
    # Fallback: basic detection
    ARCH=$(uname -m)
    CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "0")
    if [[ "$ARCH" == "arm64" ]]; then
        CHIP_TYPE="Apple Silicon"
        CHIP_MODEL="Apple Silicon (Unknown)"
    else
        CHIP_TYPE="Intel"
        CHIP_MODEL="Intel"
    fi
    echo -e "${YELLOW}⚠ Using basic system detection${NC}"
    echo ""
fi

# Set optimizations based on detected system
if [[ "$CHIP_TYPE" == "Apple Silicon" ]]; then
    # Apple Silicon: Enable Metal/MPS GPU acceleration
    export OLLAMA_METAL=1
    export OLLAMA_NUM_GPU=-1  # Use all available GPU cores
    METAL_STATUS="Enabled (OLLAMA_METAL=1)"
    GPU_STATUS="All available (OLLAMA_NUM_GPU=-1)"
else
    # Intel Mac: Disable Metal (not supported for GPU acceleration)
    export OLLAMA_METAL=0
    export OLLAMA_NUM_GPU=0
    METAL_STATUS="Disabled (Intel Mac - CPU only)"
    GPU_STATUS="N/A (CPU only mode)"
fi

# Set CPU thread optimization (if not already set)
if [ -z "$OLLAMA_NUM_THREAD" ] && [ "$CPU_CORES" -gt 0 ]; then
    export OLLAMA_NUM_THREAD="$CPU_CORES"
fi

# Optional: Set other optimizations (can be overridden by env vars)
export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

# Auto-calculate memory limit if not set
if [ -z "$OLLAMA_MAX_RAM" ]; then
    # Try to calculate optimal memory limit
    if [ -f "$PROJECT_ROOT/scripts/calculate_memory_limit.sh" ]; then
        CALCULATED_RAM=$("$PROJECT_ROOT/scripts/calculate_memory_limit.sh" 2>/dev/null | grep "OLLAMA_MAX_RAM=" | cut -d'=' -f2)
        if [ -n "$CALCULATED_RAM" ]; then
            export OLLAMA_MAX_RAM="$CALCULATED_RAM"
            echo -e "${GREEN}✓ Auto-calculated OLLAMA_MAX_RAM: ${OLLAMA_MAX_RAM}${NC}"
        fi
    fi
fi

echo -e "${BLUE}Optimization Configuration:${NC}"
echo "  ✓ System: ${CHIP_MODEL}"
echo "  ✓ Metal/MPS GPU: ${METAL_STATUS}"
echo "  ✓ GPU Cores: ${GPU_STATUS}"
if [ -n "$OLLAMA_NUM_THREAD" ]; then
    echo "  ✓ CPU Threads: ${OLLAMA_NUM_THREAD} (OLLAMA_NUM_THREAD=${OLLAMA_NUM_THREAD})"
fi
echo "  ✓ Host: ${OLLAMA_HOST}"
echo "  ✓ Keep Alive: ${OLLAMA_KEEP_ALIVE}"
if [ -n "$OLLAMA_MAX_RAM" ]; then
    echo "  ✓ Max RAM: ${OLLAMA_MAX_RAM}"
fi
echo "  ✓ Logs: ${LOG_DIR}/"
echo ""

# Function to format and display Ollama request logs
format_ollama_log() {
    awk -F' \\| ' '
    /\[GIN\]/ {
        # Parse GIN log format: [GIN] 2025/11/06 - 11:31:29 | 200 | 385.708µs | 127.0.0.1 | GET "/api/tags"
        # Remove [GIN] prefix and extract date/time from first field
        gsub(/\[GIN\] /, "", $1)
        split($1, datetime, " - ")
        date = datetime[1]
        time = datetime[2]

        # Extract other fields (already split by |)
        status = $2
        duration = $3
        ip = $4
        # Last field contains method and endpoint: GET "/api/tags"
        split($5, method_endpoint, " ")
        method = method_endpoint[1]
        endpoint = method_endpoint[2]
        gsub(/"/, "", endpoint)

        # Trim whitespace
        gsub(/^[ \t]+|[ \t]+$/, "", status)
        gsub(/^[ \t]+|[ \t]+$/, "", duration)
        gsub(/^[ \t]+|[ \t]+$/, "", ip)
        gsub(/^[ \t]+|[ \t]+$/, "", method)
        gsub(/^[ \t]+|[ \t]+$/, "", endpoint)

        # Determine status color
        if (status ~ /^2[0-9]{2}$/) {
            status_color = "\033[0;32m"  # GREEN
        } else if (status ~ /^[34][0-9]{2}$/) {
            status_color = "\033[1;33m"  # YELLOW
        } else if (status ~ /^[45][0-9]{2}$/) {
            status_color = "\033[0;31m"  # RED
        } else {
            status_color = "\033[0m"     # NC
        }

        # Determine method color
        if (method == "GET") {
            method_color = "\033[0;34m"  # BLUE
        } else if (method == "POST") {
            method_color = "\033[0;36m"  # CYAN
        } else if (method == "PUT") {
            method_color = "\033[1;33m"  # YELLOW
        } else if (method == "DELETE") {
            method_color = "\033[0;31m"  # RED
        } else {
            method_color = "\033[0m"     # NC
        }

        # Format and print
        printf "\033[0;90m%s %s\033[0m | %s%s\033[0m | %s%s\033[0m %-35s | \033[0;36m%s\033[0m | \033[0;90m%s\033[0m\n",
            date, time, status_color, status, method_color, method, endpoint, duration, ip
        next
    }
    {
        # Print non-GIN lines as-is (errors, etc.)
        print "\033[0;90m" $0 "\033[0m"
    }
    '
}

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
        echo ""

        # Show available models
        MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | tr '\n' ' ' || echo "none")
        if [ -n "$MODELS" ] && [ "$MODELS" != "none" ]; then
            echo "Available models: $MODELS"
            echo ""

            # Automatically warm up models
            echo -e "${BLUE}🔥 Warming up models...${NC}"
            WARMUP_SCRIPT="$PROJECT_ROOT/scripts/warmup_models.sh"
            if [ -f "$WARMUP_SCRIPT" ]; then
                # Use the same keep-alive setting as the service
                KEEP_ALIVE="$OLLAMA_KEEP_ALIVE" "$WARMUP_SCRIPT" || {
                    echo -e "${YELLOW}⚠ Model warm-up had some issues (models may still be available)${NC}"
                }
            else
                echo -e "${YELLOW}⚠ Warm-up script not found at $WARMUP_SCRIPT${NC}"
            fi
            echo ""
        else
            echo "No models downloaded yet. Run: ollama pull qwen2.5vl:7b"
            echo "Or pull all models: ollama pull qwen2.5vl:7b && ollama pull qwen2.5:7b && ollama pull qwen2.5:14b && ollama pull granite4:tiny-h"
        fi
        echo ""
        echo "=================================================="
        echo -e "${BLUE}📊 Request Monitor${NC}"
        echo "=================================================="
        echo -e "${GRAY}Monitoring Ollama requests in real-time...${NC}"
        echo -e "${GRAY}Press Ctrl+C to stop monitoring (service continues running)${NC}"
        echo ""
        echo -e "${GRAY}Date       Time     | Status | Method  Endpoint                        | Duration    | IP${NC}"
        echo -e "${GRAY}--------------------------------------------------------------------------------${NC}"

        # Start monitoring logs with formatted output
        # Use trap to handle Ctrl+C gracefully
        trap 'echo ""; echo -e "${BLUE}Monitoring stopped. Service is still running.${NC}"; echo -e "${GRAY}To view logs: tail -f $LOG_DIR/ollama.log${NC}"; echo -e "${GRAY}To stop service: ./scripts/shutdown.sh${NC}"; exit 0' INT TERM

        # Tail the log file and format it
        # Disable exit on error for tail (it may exit when log file is rotated)
        set +e
        tail -f "$LOG_DIR/ollama.log" 2>/dev/null | format_ollama_log
        set -e
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep 1
done

echo -e "${RED}✗ Failed to start Ollama service${NC}"
echo "Check logs: tail -f $LOG_DIR/ollama.error.log"
exit 1

