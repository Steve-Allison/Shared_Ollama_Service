#!/bin/bash

# Generate Optimal Configuration Script
# Auto-detects system hardware and generates optimal .env configuration
# Works across multiple machines with different hardware specs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/env.example"

echo -e "${BLUE}🔍 Auto-Detecting System Hardware and Generating Optimal Configuration${NC}"
echo "========================================================================"
echo ""

# Source detection scripts
DETECT_SCRIPT="$SCRIPT_DIR/detect_system.sh"
MEMORY_SCRIPT="$SCRIPT_DIR/calculate_memory_limit.sh"

# Detect system capabilities
if [ ! -f "$DETECT_SCRIPT" ]; then
    echo -e "${RED}✗ System detection script not found: $DETECT_SCRIPT${NC}"
    exit 1
fi

if [ ! -f "$MEMORY_SCRIPT" ]; then
    echo -e "${RED}✗ Memory calculation script not found: $MEMORY_SCRIPT${NC}"
    exit 1
fi

echo -e "${CYAN}[1/5]${NC} Detecting system hardware..."
SYSTEM_INFO=$(bash "$DETECT_SCRIPT" 2>/dev/null)
# Parse key-value pairs safely
while IFS='=' read -r key value; do
    # Skip empty lines and comments
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    # Export variables safely
    export "$key=$value"
done <<< "$SYSTEM_INFO"

echo -e "${GREEN}✓${NC} Architecture: $ARCH"
echo -e "${GREEN}✓${NC} Chip Type: $CHIP_TYPE"
echo -e "${GREEN}✓${NC} Chip Model: $CHIP_MODEL"
echo -e "${GREEN}✓${NC} CPU Cores: $CPU_CORES"
echo -e "${GREEN}✓${NC} GPU Cores: $GPU_CORES"
echo -e "${GREEN}✓${NC} Total RAM: ${TOTAL_RAM_GB} GB"
echo ""

# Calculate optimal parallel models first (needed for memory calculation)
echo -e "${CYAN}[2/6]${NC} Calculating optimal parallel model configuration..."
# Estimate based on total RAM, accounting for RAG systems and other services
# Reserve: 8GB system + 8GB RAG + 4GB safety = 20GB
AVAILABLE_FOR_OLLAMA=$((TOTAL_RAM_GB - 20))
# Largest model is ~10GB, so calculate how many can fit
# Use conservative estimate: need space for models + inference buffer
LARGEST_MODEL_GB=10
INFERENCE_BUFFER_GB=4
MODELS_PER_GB=$((LARGEST_MODEL_GB + INFERENCE_BUFFER_GB))
OLLAMA_NUM_PARALLEL=$((AVAILABLE_FOR_OLLAMA / MODELS_PER_GB))
# Cap at reasonable limits
if [ $OLLAMA_NUM_PARALLEL -lt 1 ]; then
    OLLAMA_NUM_PARALLEL=1
elif [ $OLLAMA_NUM_PARALLEL -gt 3 ]; then
    OLLAMA_NUM_PARALLEL=3  # Max 3 parallel models (even on high-end systems)
fi
echo -e "${GREEN}✓${NC} Recommended OLLAMA_NUM_PARALLEL: $OLLAMA_NUM_PARALLEL"
echo ""

# Calculate optimal memory limit (now that we know parallel models)
echo -e "${CYAN}[3/6]${NC} Calculating optimal memory allocation..."
# Export parallel models so memory script can use it
export OLLAMA_NUM_PARALLEL
MEMORY_OUTPUT=$(bash "$MEMORY_SCRIPT" 2>&1)
OLLAMA_MAX_RAM=$(echo "$MEMORY_OUTPUT" | grep "OLLAMA_MAX_RAM=" | head -1 | sed 's/.*OLLAMA_MAX_RAM=//' | sed 's/GB/GB/')
if [ -z "$OLLAMA_MAX_RAM" ]; then
    # Fallback calculation (shouldn't happen, but safety net)
    echo -e "${YELLOW}⚠${NC} Could not parse memory output, using fallback calculation"
    SYSTEM_RESERVE_GB=20  # System + RAG + safety
    OLLAMA_MAX_RAM_GB=$((TOTAL_RAM_GB - SYSTEM_RESERVE_GB))
    if [ $OLLAMA_MAX_RAM_GB -lt 4 ]; then
        OLLAMA_MAX_RAM_GB=4
    fi
    OLLAMA_MAX_RAM="${OLLAMA_MAX_RAM_GB}GB"
fi
echo -e "${GREEN}✓${NC} Recommended OLLAMA_MAX_RAM: $OLLAMA_MAX_RAM"
echo ""

# Determine GPU/Metal settings
echo -e "${CYAN}[4/6]${NC} Configuring GPU acceleration..."
if [[ "$ARCH" == "arm64" ]] && [[ "$CHIP_TYPE" == "Apple Silicon" ]]; then
    OLLAMA_METAL="1"
    OLLAMA_NUM_GPU="-1"  # Use all GPU cores
    echo -e "${GREEN}✓${NC} Metal acceleration: Enabled"
    echo -e "${GREEN}✓${NC} GPU cores: All ($GPU_CORES available)"
else
    OLLAMA_METAL="0"
    OLLAMA_NUM_GPU="0"
    echo -e "${YELLOW}⚠${NC} Metal acceleration: Not available (Intel Mac or non-macOS)"
fi
echo ""

# Detect network configuration
echo -e "${CYAN}[5/6]${NC} Detecting network configuration..."
# Get local IP addresses (excluding localhost)
if [[ "$OSTYPE" == "darwin"* ]]; then
    LOCAL_IPS=$(ifconfig 2>/dev/null | grep -E "inet " | grep -v "127.0.0.1" | awk '{print $2}' | head -3 || echo "")
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LOCAL_IPS=$(ip addr show 2>/dev/null | grep -E "inet " | grep -v "127.0.0.1" | awk '{print $2}' | cut -d'/' -f1 | head -3 || echo "")
fi

if [ -n "$LOCAL_IPS" ]; then
    echo -e "${GREEN}✓${NC} Network interfaces detected"
    echo "$LOCAL_IPS" | while read -r ip; do
        if [ -n "$ip" ]; then
            echo -e "${GRAY}  - $ip${NC}"
        fi
    done
fi

# Default to localhost for security (can be changed to 0.0.0.0 for network access)
OLLAMA_HOST="localhost"
API_HOST="0.0.0.0"  # REST API can be network-accessible if needed
NETWORK_ACCESS="disabled"
echo -e "${GREEN}✓${NC} Network access: Disabled (localhost only for security)"
echo -e "${GRAY}  Ollama service: localhost only${NC}"
echo -e "${GRAY}  REST API: 0.0.0.0 (network accessible on port 8000)${NC}"
echo -e "${GRAY}  To enable network access, set OLLAMA_HOST=0.0.0.0 in .env${NC}"
echo ""

# Generate .env file
echo -e "${CYAN}[6/6]${NC} Generating optimized .env file..."

# Backup existing .env if it exists
if [ -f "$ENV_FILE" ]; then
    BACKUP_FILE="${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$ENV_FILE" "$BACKUP_FILE"
    echo -e "${YELLOW}⚠${NC} Backed up existing .env to: $BACKUP_FILE"
fi

# Start with example file if it exists, otherwise create new
if [ -f "$ENV_EXAMPLE" ]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    echo -e "${GREEN}✓${NC} Started with env.example template"
else
    touch "$ENV_FILE"
    echo -e "${YELLOW}⚠${NC} env.example not found, creating new .env file"
fi

# Function to update or add config value
update_config() {
    local key=$1
    local value=$2
    local comment=$3
    
    # Remove existing line if present
    sed -i.bak "/^${key}=/d" "$ENV_FILE" 2>/dev/null || true
    
    # Add new line with comment if provided
    if [ -n "$comment" ]; then
        echo "" >> "$ENV_FILE"
        echo "# $comment" >> "$ENV_FILE"
        echo "# Auto-generated by generate_optimal_config.sh on $(date)" >> "$ENV_FILE"
        echo "${key}=${value}" >> "$ENV_FILE"
    else
        echo "${key}=${value}" >> "$ENV_FILE"
    fi
}

# Update Ollama optimizations
echo "" >> "$ENV_FILE"
echo "# ============================================================================" >> "$ENV_FILE"
echo "# Auto-Generated Optimal Configuration" >> "$ENV_FILE"
echo "# Generated on: $(date)" >> "$ENV_FILE"
echo "# System: $CHIP_MODEL ($ARCH) - ${TOTAL_RAM_GB}GB RAM, ${CPU_CORES} CPU cores" >> "$ENV_FILE"
echo "# ============================================================================" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"

# Update OLLAMA_METAL
sed -i.bak "/^OLLAMA_METAL=/d" "$ENV_FILE" 2>/dev/null || true
echo "# Metal GPU acceleration (Apple Silicon only)" >> "$ENV_FILE"
echo "OLLAMA_METAL=$OLLAMA_METAL" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"

# Update OLLAMA_NUM_GPU
sed -i.bak "/^OLLAMA_NUM_GPU=/d" "$ENV_FILE" 2>/dev/null || true
echo "# Number of GPU cores (-1 = all available)" >> "$ENV_FILE"
echo "OLLAMA_NUM_GPU=$OLLAMA_NUM_GPU" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"

# Update OLLAMA_NUM_THREAD
sed -i.bak "/^OLLAMA_NUM_THREAD=/d" "$ENV_FILE" 2>/dev/null || true
echo "# CPU threads (auto-detected: $CPU_CORES cores)" >> "$ENV_FILE"
echo "OLLAMA_NUM_THREAD=$CPU_CORES" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"

# Update OLLAMA_MAX_RAM
sed -i.bak "/^OLLAMA_MAX_RAM=/d" "$ENV_FILE" 2>/dev/null || true
echo "# Maximum RAM for Ollama (calculated: ${TOTAL_RAM_GB}GB total - 25% system reserve)" >> "$ENV_FILE"
echo "OLLAMA_MAX_RAM=$OLLAMA_MAX_RAM" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"

# Update OLLAMA_NUM_PARALLEL
sed -i.bak "/^OLLAMA_NUM_PARALLEL=/d" "$ENV_FILE" 2>/dev/null || true
echo "# Number of parallel models (calculated based on available RAM)" >> "$ENV_FILE"
echo "OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"

# Update OLLAMA_KEEP_ALIVE (use longer for better performance)
sed -i.bak "/^OLLAMA_KEEP_ALIVE=/d" "$ENV_FILE" 2>/dev/null || true
echo "# Model keep-alive duration (30m recommended for better performance)" >> "$ENV_FILE"
echo "OLLAMA_KEEP_ALIVE=30m" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"

# Update OLLAMA_HOST (network access)
sed -i.bak "/^OLLAMA_HOST=/d" "$ENV_FILE" 2>/dev/null || true
echo "# Ollama service host (0.0.0.0 = network accessible, localhost = local only)" >> "$ENV_FILE"
echo "OLLAMA_HOST=$OLLAMA_HOST" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"

# Update API_HOST (network access)
sed -i.bak "/^API_HOST=/d" "$ENV_FILE" 2>/dev/null || true
echo "# REST API host (0.0.0.0 = network accessible, localhost = local only)" >> "$ENV_FILE"
echo "API_HOST=$API_HOST" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"

# Clean up backup files
rm -f "${ENV_FILE}.bak" 2>/dev/null || true

echo -e "${GREEN}✓${NC} Configuration written to: $ENV_FILE"
echo ""

# Summary
echo -e "${BLUE}📊 Configuration Summary${NC}"
echo "========================================================================"
echo -e "  ${CYAN}System:${NC} $CHIP_MODEL ($ARCH)"
echo -e "  ${CYAN}RAM:${NC} ${TOTAL_RAM_GB}GB total → ${OLLAMA_MAX_RAM} for Ollama"
echo -e "  ${CYAN}CPU:${NC} $CPU_CORES cores"
echo -e "  ${CYAN}GPU:${NC} $GPU_CORES cores (Metal: $([ "$OLLAMA_METAL" = "1" ] && echo "Enabled" || echo "Disabled"))"
echo -e "  ${CYAN}Parallel Models:${NC} $OLLAMA_NUM_PARALLEL"
echo -e "  ${CYAN}Keep-Alive:${NC} 30m"
echo -e "  ${CYAN}Network Access:${NC} Localhost only (Ollama), Network accessible (REST API)"
echo ""
echo -e "${GREEN}✓ Configuration complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review the generated .env file: cat $ENV_FILE"
echo "  2. Restart the service to apply new settings: ./scripts/start.sh"
echo "  3. To enable network access for Ollama, set OLLAMA_HOST=0.0.0.0 in .env"
echo ""

