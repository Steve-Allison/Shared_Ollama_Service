#!/bin/bash

# System Detection Script for Mac Optimization
# Detects Mac model, chip type, CPU cores, and GPU cores
# Returns system information for optimization configuration

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "ERROR: Not running on macOS" >&2
    exit 1
fi

# Detect architecture
ARCH=$(uname -m)

# Detect CPU cores
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "0")

# Detect chip model
CHIP_MODEL=""
CHIP_TYPE=""
GPU_CORES=""

if [[ "$ARCH" == "arm64" ]]; then
    # Apple Silicon
    CHIP_TYPE="Apple Silicon"

    # Get chip name from sysctl
    CHIP_NAME=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")

    # Detect specific chip model
    if [[ "$CHIP_NAME" == *"Apple M1"* ]]; then
        if [[ "$CHIP_NAME" == *"Pro"* ]] || [[ "$CHIP_NAME" == *"Max"* ]]; then
            CHIP_MODEL="M1 Pro/Max"
            GPU_CORES="14-16"  # M1 Pro: 14, M1 Max: 16
        else
            CHIP_MODEL="M1"
            GPU_CORES="7-8"  # M1: 7 or 8 cores
        fi
    elif [[ "$CHIP_NAME" == *"Apple M2"* ]]; then
        if [[ "$CHIP_NAME" == *"Ultra"* ]]; then
            CHIP_MODEL="M2 Ultra"
            GPU_CORES="60-76"  # M2 Ultra: 60 or 76 cores
        elif [[ "$CHIP_NAME" == *"Max"* ]]; then
            CHIP_MODEL="M2 Max"
            GPU_CORES="30-38"  # M2 Max: 30 or 38 cores
        elif [[ "$CHIP_NAME" == *"Pro"* ]]; then
            CHIP_MODEL="M2 Pro"
            GPU_CORES="16-19"  # M2 Pro: 16 or 19 cores
        else
            CHIP_MODEL="M2"
            GPU_CORES="8-10"  # M2: 8 or 10 cores
        fi
    elif [[ "$CHIP_NAME" == *"Apple M3"* ]]; then
        if [[ "$CHIP_NAME" == *"Ultra"* ]]; then
            CHIP_MODEL="M3 Ultra"
            GPU_CORES="80"  # M3 Ultra: 80 cores
        elif [[ "$CHIP_NAME" == *"Max"* ]]; then
            CHIP_MODEL="M3 Max"
            GPU_CORES="40"  # M3 Max: 40 cores
        elif [[ "$CHIP_NAME" == *"Pro"* ]]; then
            CHIP_MODEL="M3 Pro"
            GPU_CORES="18"  # M3 Pro: 18 cores
        else
            CHIP_MODEL="M3"
            GPU_CORES="10"  # M3: 10 cores
        fi
    elif [[ "$CHIP_NAME" == *"Apple M4"* ]]; then
        if [[ "$CHIP_NAME" == *"Ultra"* ]]; then
            CHIP_MODEL="M4 Ultra"
            GPU_CORES="80"  # Estimated, may vary
        elif [[ "$CHIP_NAME" == *"Max"* ]]; then
            CHIP_MODEL="M4 Max"
            GPU_CORES="40"  # Estimated, may vary
        elif [[ "$CHIP_NAME" == *"Pro"* ]]; then
            CHIP_MODEL="M4 Pro"
            GPU_CORES="20"  # Estimated, may vary
        else
            CHIP_MODEL="M4"
            GPU_CORES="10"  # M4: 10 cores
        fi
    else
        # Unknown Apple Silicon chip
        CHIP_MODEL="Apple Silicon (Unknown)"
        GPU_CORES="unknown"
    fi

    # Try to get actual GPU core count from system_profiler
    if command -v system_profiler &> /dev/null; then
        ACTUAL_GPU=$(system_profiler SPDisplaysDataType 2>/dev/null | grep -i "Metal.*Family" | head -1 || echo "")
        # Note: system_profiler doesn't directly report GPU core count, but we can infer from chip model
    fi
else
    # Intel Mac
    CHIP_TYPE="Intel"
    CHIP_MODEL=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Intel (Unknown)")
    GPU_CORES="0"  # Intel Macs don't use Metal for Ollama GPU acceleration
fi

# Detect total RAM
if [[ "$OSTYPE" == "darwin"* ]]; then
    TOTAL_RAM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f\n", $1/1024/1024/1024}' || echo "0")
else
    TOTAL_RAM_GB=0
fi

# Output as key-value pairs for easy parsing
echo "ARCH=$ARCH"
echo "CHIP_TYPE=$CHIP_TYPE"
echo "CHIP_MODEL=$CHIP_MODEL"
echo "CPU_CORES=$CPU_CORES"
echo "GPU_CORES=$GPU_CORES"
echo "TOTAL_RAM_GB=$TOTAL_RAM_GB"

# If verbose mode requested, output human-readable info
if [[ "${1:-}" == "--verbose" ]] || [[ "${1:-}" == "-v" ]]; then
    echo "" >&2
    echo -e "${BLUE}System Detection Results:${NC}" >&2
    echo -e "  Architecture: ${GREEN}$ARCH${NC}" >&2
    echo -e "  Chip Type: ${GREEN}$CHIP_TYPE${NC}" >&2
    echo -e "  Chip Model: ${GREEN}$CHIP_MODEL${NC}" >&2
    echo -e "  CPU Cores: ${GREEN}$CPU_CORES${NC}" >&2
    echo -e "  GPU Cores: ${GREEN}$GPU_CORES${NC}" >&2
    echo -e "  Total RAM: ${GREEN}${TOTAL_RAM_GB} GB${NC}" >&2
fi

