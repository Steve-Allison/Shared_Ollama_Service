#!/bin/bash

# Generate Optimal Configuration Script
# Auto-detects system hardware and reports optimal configuration
# Works across multiple machines with different hardware specs without .env files

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
MODEL_PROFILE_FILE="$PROJECT_ROOT/config/model_profiles.yaml"

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

if [ ! -f "$MODEL_PROFILE_FILE" ]; then
    echo -e "${RED}✗ Model profile configuration not found: $MODEL_PROFILE_FILE${NC}"
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

# Load model profile defaults
PROFILE_DEFAULTS=$(PROJECT_ROOT="$PROJECT_ROOT" ARCH="$ARCH" TOTAL_RAM_GB="$TOTAL_RAM_GB" python3 - <<'PY'
import json, math, os, yaml
project_root = os.environ["PROJECT_ROOT"]
profile_path = os.path.join(project_root, "config", "model_profiles.yaml")
with open(profile_path, "r", encoding="utf-8") as fh:
    data = yaml.safe_load(fh) or {}
profiles = data.get("profiles") or {}
ram = int(os.environ.get("TOTAL_RAM_GB", 0) or 0)
arch = os.environ.get("ARCH")
selected = profiles.get("default", {}) if isinstance(profiles, dict) else {}
for profile in profiles.values():
    if not isinstance(profile, dict):
        continue
    match = profile.get("match") or {}
    min_ram = match.get("min_ram_gb", 0)
    max_ram = match.get("max_ram_gb", math.inf)
    match_arch = match.get("arch")
    if ram >= min_ram and ram <= max_ram and (match_arch is None or match_arch == arch):
        selected = profile
        break
defaults = selected.get("defaults") or {}
print(json.dumps(defaults))
PY
)

DEFAULT_VLM_MODEL=$(echo "$PROFILE_DEFAULTS" | jq -r '.vlm_model // "qwen3-vl:8b-instruct-q4_K_M"')
DEFAULT_TEXT_MODEL=$(echo "$PROFILE_DEFAULTS" | jq -r '.text_model // "qwen3:14b-q4_K_M"')
REQUIRED_MODELS=$(echo "$PROFILE_DEFAULTS" | jq -r '(.required_models // []) | join(",")')
WARMUP_MODELS=$(echo "$PROFILE_DEFAULTS" | jq -r '(.warmup_models // []) | join(",")')
MODEL_MEMORY_HINTS=$(echo "$PROFILE_DEFAULTS" | jq -r '(.memory_hints // {}) | to_entries | map("\(.key):\(.value)") | join(",")')
PROFILE_LARGEST_MODEL_GB=$(echo "$PROFILE_DEFAULTS" | jq -r '.largest_model_gb // 19')
PROFILE_INFERENCE_BUFFER_GB=$(echo "$PROFILE_DEFAULTS" | jq -r '.inference_buffer_gb // 4')
PROFILE_SERVICE_OVERHEAD_GB=$(echo "$PROFILE_DEFAULTS" | jq -r '.service_overhead_gb // 2')

export OLLAMA_DEFAULT_VLM_MODEL="$DEFAULT_VLM_MODEL"
export OLLAMA_DEFAULT_TEXT_MODEL="$DEFAULT_TEXT_MODEL"
export OLLAMA_REQUIRED_MODELS="${REQUIRED_MODELS:-$DEFAULT_VLM_MODEL,$DEFAULT_TEXT_MODEL}"
export OLLAMA_WARMUP_MODELS="${WARMUP_MODELS:-$DEFAULT_VLM_MODEL,$DEFAULT_TEXT_MODEL}"
export OLLAMA_MODEL_MEMORY_HINTS="$MODEL_MEMORY_HINTS"
export OLLAMA_LARGEST_MODEL_GB="$PROFILE_LARGEST_MODEL_GB"
export OLLAMA_INFERENCE_BUFFER_GB="$PROFILE_INFERENCE_BUFFER_GB"
export OLLAMA_SERVICE_OVERHEAD_GB="$PROFILE_SERVICE_OVERHEAD_GB"

# Calculate optimal parallel models first (needed for memory calculation)
echo -e "${CYAN}[2/6]${NC} Calculating optimal parallel model configuration..."
# Estimate based on total RAM, accounting for RAG systems and other services
# Reserve: 8GB system + 8GB RAG + 4GB safety = 20GB
AVAILABLE_FOR_OLLAMA=$((TOTAL_RAM_GB - 20))
# Largest model based on selected profile
# Use conservative estimate: need space for models + inference buffer
LARGEST_MODEL_GB=${PROFILE_LARGEST_MODEL_GB:-10}
INFERENCE_BUFFER_GB=${PROFILE_INFERENCE_BUFFER_GB:-4}
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
echo -e "${GRAY}  To enable network access, set OLLAMA_HOST=0.0.0.0 before starting the service${NC}"
echo ""

# Final summary
echo -e "${CYAN}[6/6]${NC} Final configuration (applied dynamically at runtime)"
echo ""
echo -e "${BLUE}📊 Configuration Summary${NC}"
echo "========================================================================"
echo -e "  ${CYAN}System:${NC} $CHIP_MODEL ($ARCH)"
echo -e "  ${CYAN}RAM:${NC} ${TOTAL_RAM_GB}GB total → ${OLLAMA_MAX_RAM} for Ollama"
echo -e "  ${CYAN}CPU:${NC} $CPU_CORES cores"
echo -e "  ${CYAN}GPU:${NC} $GPU_CORES cores (Metal: $([ "$OLLAMA_METAL" = "1" ] && echo "Enabled" || echo "Disabled"))"
echo -e "  ${CYAN}Parallel Models:${NC} $OLLAMA_NUM_PARALLEL"
echo -e "  ${CYAN}Keep-Alive:${NC} 30m"
echo -e "  ${CYAN}Network Access:${NC} Localhost only (Ollama), Network accessible (REST API)"
echo -e "  ${CYAN}Default VLM:${NC} $OLLAMA_DEFAULT_VLM_MODEL"
echo -e "  ${CYAN}Default Text:${NC} $OLLAMA_DEFAULT_TEXT_MODEL"
echo -e "  ${CYAN}Warmup Models:${NC} $OLLAMA_WARMUP_MODELS"
echo ""
echo -e "${GREEN}✓ Configuration complete!${NC}"
echo ""
echo -e "${YELLOW}Note:${NC} Values are applied automatically via config/model_profiles.yaml."
echo "      To override any setting temporarily, export the corresponding environment"
echo "      variable before running a script. No .env file is required."
echo ""

