#!/bin/bash

# Setup Ollama as a macOS Launch Agent for automatic startup
# This creates a launchd service that starts Ollama automatically

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Setting up Ollama as Launch Agent${NC}"
echo "======================================"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}✗ This script is for macOS only${NC}"
    exit 1
fi

# Find Ollama binary
OLLAMA_BIN=$(which ollama 2>/dev/null || echo "/usr/local/bin/ollama")

if [ ! -f "$OLLAMA_BIN" ]; then
    echo -e "${RED}✗ Ollama not found${NC}"
    echo "Please install Ollama first: ./scripts/install_native.sh"
    exit 1
fi

echo -e "${GREEN}✓ Found Ollama at: $OLLAMA_BIN${NC}"

# Get project root directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"

# Create logs directory in project
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✓ Logs directory: $LOG_DIR${NC}"

# Calculate optimal memory limit if not set
if [ -z "$OLLAMA_MAX_RAM" ]; then
    echo ""
    echo -e "${BLUE}Calculating optimal memory limit...${NC}"
    if [ -f "$PROJECT_ROOT/scripts/calculate_memory_limit.sh" ]; then
        CALCULATED_RAM=$("$PROJECT_ROOT/scripts/calculate_memory_limit.sh" 2>/dev/null | grep "OLLAMA_MAX_RAM=" | cut -d'=' -f2)
        if [ -n "$CALCULATED_RAM" ]; then
            OLLAMA_MAX_RAM="$CALCULATED_RAM"
            echo -e "${GREEN}✓ Auto-calculated OLLAMA_MAX_RAM: ${OLLAMA_MAX_RAM}${NC}"
        fi
    fi
fi

# Create launchd plist
LAUNCHD_DIR="$HOME/Library/LaunchAgents"
PLIST_FILE="$LAUNCHD_DIR/com.ollama.service.plist"

mkdir -p "$LAUNCHD_DIR"

echo ""
echo -e "${BLUE}Creating launchd service file...${NC}"

cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ollama.service</string>
    <key>ProgramArguments</key>
    <array>
        <string>$OLLAMA_BIN</string>
        <string>serve</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/ollama.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/ollama.error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>OLLAMA_HOST</key>
        <string>0.0.0.0:11434</string>
        <key>OLLAMA_KEEP_ALIVE</key>
        <string>5m</string>
        <key>OLLAMA_METAL</key>
        <string>1</string>
        <key>OLLAMA_NUM_GPU</key>
        <string>-1</string>$(if [ -n "$OLLAMA_MAX_RAM" ]; then echo "
        <key>OLLAMA_MAX_RAM</key>
        <string>$OLLAMA_MAX_RAM</string>"; fi)
    </dict>
</dict>
</plist>
EOF

echo -e "${GREEN}✓ Created: $PLIST_FILE${NC}"

# Load the service
echo ""
echo -e "${BLUE}Loading launchd service...${NC}"

# Unload if already loaded
if launchctl list | grep -q "com.ollama.service"; then
    echo "Unloading existing service..."
    launchctl unload "$PLIST_FILE" 2>/dev/null || true
fi

# Load the service
launchctl load "$PLIST_FILE"

echo -e "${GREEN}✓ Service loaded${NC}"

# Wait a moment for service to start
sleep 2

# Verify service is running
if curl -f -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama service is running${NC}"
else
    echo -e "${YELLOW}⚠ Service may still be starting...${NC}"
    echo "Check status with: launchctl list | grep ollama"
fi

echo ""
echo "======================================"
echo -e "${GREEN}✓ Launch Agent setup complete!${NC}"
echo ""
echo "Service Management:"
echo "  Start:   launchctl load $PLIST_FILE"
echo "  Stop:    launchctl unload $PLIST_FILE"
echo "  Status:  launchctl list | grep ollama"
echo "  Logs:    tail -f $LOG_DIR/ollama.log"
echo "  Errors:  tail -f $LOG_DIR/ollama.error.log"
echo ""
echo "Ollama will now start automatically on login."
echo ""
echo -e "${YELLOW}Optional:${NC} To preload models on startup, add warmup to your login items or cron."
echo "  See: ./scripts/warmup_models.sh"

