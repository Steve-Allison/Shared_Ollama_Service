#!/bin/bash

# Calculate optimal OLLAMA_MAX_RAM based on model requirements
# Calculates based on actual model memory needs + parallel execution + buffers
# Leaves adequate room for RAG systems and other services on the same machine

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}📊 Calculating Optimal Ollama Memory Limit${NC}"
echo "=========================================="
echo ""

# Detect system RAM in GB
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    TOTAL_RAM_GB=$(sysctl -n hw.memsize | awk '{printf "%.0f\n", $1/1024/1024/1024}')
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
else
    echo -e "${YELLOW}⚠ Warning: Unable to detect RAM on this system${NC}"
    echo "Please set OLLAMA_MAX_RAM manually in your environment"
    exit 1
fi

echo -e "${GREEN}✓ Total System RAM: ${TOTAL_RAM_GB} GB${NC}"
echo ""

# Load environment configuration if available
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

DEFAULT_VLM_MODEL="${OLLAMA_DEFAULT_VLM_MODEL:-qwen3-vl:8b-instruct-q4_K_M}"
DEFAULT_TEXT_MODEL="${OLLAMA_DEFAULT_TEXT_MODEL:-qwen3:14b-q4_K_M}"
REQUIRED_MODELS_CSV="${OLLAMA_REQUIRED_MODELS:-$DEFAULT_VLM_MODEL,$DEFAULT_TEXT_MODEL}"
MODEL_MEMORY_HINTS="${OLLAMA_MODEL_MEMORY_HINTS:-$DEFAULT_VLM_MODEL:6,$DEFAULT_TEXT_MODEL:8}"

IFS=',' read -r -a REQUIRED_MODELS <<< "$REQUIRED_MODELS_CSV"
declare -A MODEL_MEMORY_MAP
IFS=',' read -r -a HINT_PAIRS <<< "$MODEL_MEMORY_HINTS"
for pair in "${HINT_PAIRS[@]}"; do
    model_name="${pair%%:*}"
    mem_hint="${pair##*:}"
    if [ -n "$model_name" ] && [ -n "$mem_hint" ]; then
        MODEL_MEMORY_MAP["$model_name"]="$mem_hint"
    fi
done

LARGEST_MODEL=${OLLAMA_LARGEST_MODEL_GB:-19}

# Get parallel model count (default to 2 if not set, max 3)
PARALLEL_MODELS=${OLLAMA_NUM_PARALLEL:-2}
if [ "$PARALLEL_MODELS" -gt 3 ]; then
    PARALLEL_MODELS=3
fi

echo -e "${CYAN}Model Memory Requirements:${NC}"
for model in "${REQUIRED_MODELS[@]}"; do
    hint="${MODEL_MEMORY_MAP[$model]}"
    if [ -z "$hint" ]; then
        hint=8
    fi
    if [ "$model" == "${REQUIRED_MODELS[-1]}" ] && [ "$hint" -eq "$LARGEST_MODEL" ]; then
        echo "  - $model: ${hint} GB (largest)"
    else
        echo "  - $model: ${hint} GB"
    fi
done
echo ""

# Calculate Ollama memory needs based on actual requirements
# Formula: (largest_model × parallel_count) + inference_buffer + overhead
INFERENCE_BUFFER=${OLLAMA_INFERENCE_BUFFER_GB:-4}
OVERHEAD=${OLLAMA_SERVICE_OVERHEAD_GB:-2}

OLLAMA_REQUIRED_GB=$((LARGEST_MODEL * PARALLEL_MODELS + INFERENCE_BUFFER + OVERHEAD))

echo -e "${CYAN}Ollama Memory Calculation:${NC}"
echo "  - Largest model: ${LARGEST_MODEL} GB"
echo "  - Parallel models: ${PARALLEL_MODELS}"
echo "  - Model memory: $((LARGEST_MODEL * PARALLEL_MODELS)) GB"
echo "  - Inference buffer: ${INFERENCE_BUFFER} GB"
echo "  - Service overhead: ${OVERHEAD} GB"
echo "  - ${CYAN}Total Ollama needs: ${OLLAMA_REQUIRED_GB} GB${NC}"
echo ""

# Reserve memory for other services
# System overhead (macOS/OS needs)
SYSTEM_OVERHEAD=8

# RAG systems buffer (vector DBs, embeddings, document processing)
# Default: 8GB, but can be adjusted via RAG_RESERVE_GB env var
RAG_RESERVE_GB=${RAG_RESERVE_GB:-8}

# Safety buffer (for other processes, spikes, etc.)
SAFETY_BUFFER=4

TOTAL_RESERVE=$((SYSTEM_OVERHEAD + RAG_RESERVE_GB + SAFETY_BUFFER))

echo -e "${CYAN}Memory Reserves:${NC}"
echo "  - System overhead: ${SYSTEM_OVERHEAD} GB"
echo "  - RAG systems: ${RAG_RESERVE_GB} GB (set RAG_RESERVE_GB env var to customize)"
echo "  - Safety buffer: ${SAFETY_BUFFER} GB"
echo "  - ${CYAN}Total reserves: ${TOTAL_RESERVE} GB${NC}"
echo ""

# Calculate Ollama limit
OLLAMA_MAX_RAM_GB=$((TOTAL_RAM_GB - TOTAL_RESERVE))

# Ensure we have at least what Ollama needs
if [ $OLLAMA_MAX_RAM_GB -lt $OLLAMA_REQUIRED_GB ]; then
    echo -e "${YELLOW}⚠ Warning: Calculated limit (${OLLAMA_MAX_RAM_GB}GB) is less than required (${OLLAMA_REQUIRED_GB}GB)${NC}"
    echo -e "${YELLOW}  Setting to required amount. Consider reducing parallel models or RAG reserve.${NC}"
    OLLAMA_MAX_RAM_GB=$OLLAMA_REQUIRED_GB
fi

# Cap at reasonable maximum (don't use more than 80% of total RAM)
MAX_REASONABLE=$((TOTAL_RAM_GB * 80 / 100))
if [ $OLLAMA_MAX_RAM_GB -gt $MAX_REASONABLE ]; then
    echo -e "${YELLOW}⚠ Capping at 80% of total RAM for safety${NC}"
    OLLAMA_MAX_RAM_GB=$MAX_REASONABLE
fi

# Ensure minimum of 4GB for Ollama (if system has very little RAM)
if [ $OLLAMA_MAX_RAM_GB -lt 4 ]; then
    OLLAMA_MAX_RAM_GB=4
    echo -e "${YELLOW}⚠ Warning: System has limited RAM. Setting minimum Ollama limit.${NC}"
fi

echo -e "${GREEN}✓ Ollama Required: ${OLLAMA_REQUIRED_GB} GB${NC}"
echo -e "${GREEN}✓ System Reserves: ${TOTAL_RESERVE} GB${NC}"
echo -e "${GREEN}✓ Recommended OLLAMA_MAX_RAM: ${OLLAMA_MAX_RAM_GB} GB${NC}"
echo ""

# Output in different formats
echo "To set this in your environment:"
echo ""
echo "  # Export for current session:"
echo "  export OLLAMA_MAX_RAM=${OLLAMA_MAX_RAM_GB}GB"
echo ""
echo "  # Or add to your shell profile (~/.zshrc or ~/.bashrc):"
echo "  echo 'export OLLAMA_MAX_RAM=${OLLAMA_MAX_RAM_GB}GB' >> ~/.zshrc"
echo ""
echo "  # Or set in .env file (see scripts/generate_optimal_config.sh):"
echo "  # Add to EnvironmentVariables in plist:"
echo "  # <key>OLLAMA_MAX_RAM</key>"
echo "  # <string>${OLLAMA_MAX_RAM_GB}GB</string>"
echo ""

# Customization options
echo "Customization Options:"
echo ""
echo "  # Adjust RAG systems reserve (default: 8GB):"
echo "  export RAG_RESERVE_GB=12  # For larger RAG systems"
echo "  ./scripts/calculate_memory_limit.sh"
echo ""
echo "  # Adjust parallel models (affects memory calculation):"
echo "  export OLLAMA_NUM_PARALLEL=2  # Default: 2, Max: 3"
echo "  ./scripts/calculate_memory_limit.sh"
echo ""

# Output for use in scripts
echo "For use in scripts, this value is:"
echo "  OLLAMA_MAX_RAM=${OLLAMA_MAX_RAM_GB}GB"
echo ""

# Save to a temporary file for other scripts to use
OLLAMA_MAX_RAM_FILE="${TMPDIR:-/tmp}/ollama_max_ram.txt"
echo "${OLLAMA_MAX_RAM_GB}GB" > "$OLLAMA_MAX_RAM_FILE"
echo -e "${BLUE}✓ Value saved to: $OLLAMA_MAX_RAM_FILE${NC}"

