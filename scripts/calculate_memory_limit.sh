#!/bin/bash

# Calculate optimal OLLAMA_MAX_RAM based on system RAM
# Leaves 25% of RAM for system or minimum 8GB, whichever is larger

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
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

# Calculate system reserve (25% of total or 8GB minimum)
SYSTEM_RESERVE_GB=$((TOTAL_RAM_GB * 25 / 100))
if [ $SYSTEM_RESERVE_GB -lt 8 ]; then
    SYSTEM_RESERVE_GB=8
fi

# Calculate Ollama limit (total - system reserve)
OLLAMA_MAX_RAM_GB=$((TOTAL_RAM_GB - SYSTEM_RESERVE_GB))

# Ensure minimum of 4GB for Ollama (if system has very little RAM)
if [ $OLLAMA_MAX_RAM_GB -lt 4 ]; then
    OLLAMA_MAX_RAM_GB=4
    echo -e "${YELLOW}⚠ Warning: System has limited RAM. Setting minimum Ollama limit.${NC}"
fi

echo -e "${GREEN}✓ System Reserve: ${SYSTEM_RESERVE_GB} GB${NC}"
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
echo "  # Or set in launchd service (see scripts/setup_launchd.sh):"
echo "  # Add to EnvironmentVariables in plist:"
echo "  # <key>OLLAMA_MAX_RAM</key>"
echo "  # <string>${OLLAMA_MAX_RAM_GB}GB</string>"
echo ""

# Output for use in scripts
echo "For use in scripts, this value is:"
echo "  OLLAMA_MAX_RAM=${OLLAMA_MAX_RAM_GB}GB"
echo ""

# Save to a temporary file for other scripts to use
OLLAMA_MAX_RAM_FILE="${TMPDIR:-/tmp}/ollama_max_ram.txt"
echo "${OLLAMA_MAX_RAM_GB}GB" > "$OLLAMA_MAX_RAM_FILE"
echo -e "${BLUE}✓ Value saved to: $OLLAMA_MAX_RAM_FILE${NC}"

