#!/bin/bash

# Disable Homebrew Ollama Service
# This script ensures Homebrew doesn't manage Ollama - our project scripts will manage it

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Disabling Homebrew Ollama Service${NC}"
echo "=========================================="
echo ""

if ! command -v brew &> /dev/null; then
    echo -e "${GREEN}✓ Homebrew not installed (nothing to disable)${NC}"
    exit 0
fi

BREW_STATUS=$(brew services list 2>/dev/null | grep ollama || echo "")

if [ -z "$BREW_STATUS" ]; then
    echo -e "${GREEN}✓ Homebrew Ollama service not configured${NC}"
    exit 0
fi

echo -e "${BLUE}Current Homebrew Ollama status:${NC}"
echo "$BREW_STATUS"
echo ""

# Stop the service if running
if echo "$BREW_STATUS" | grep -q "started"; then
    echo -e "${BLUE}Stopping Homebrew Ollama service...${NC}"
    brew services stop ollama > /dev/null 2>&1 || true
    sleep 2
    echo -e "${GREEN}✓ Homebrew Ollama service stopped${NC}"
fi

# Verify it's disabled
FINAL_STATUS=$(brew services list 2>/dev/null | grep ollama || echo "")
if [ -n "$FINAL_STATUS" ]; then
    if echo "$FINAL_STATUS" | grep -q "started"; then
        echo -e "${YELLOW}⚠ Warning: Service may still be running${NC}"
        echo -e "${YELLOW}  Try: brew services stop ollama${NC}"
    else
        echo -e "${GREEN}✓ Homebrew Ollama service is now disabled${NC}"
        echo ""
        echo -e "${BLUE}Note:${NC}"
        echo -e "  • Ollama will be managed by this project's scripts"
        echo -e "  • Use './scripts/start_api.sh' to start the service"
        echo -e "  • Do NOT use 'brew services start ollama'"
    fi
else
    echo -e "${GREEN}✓ Homebrew Ollama service removed${NC}"
fi

