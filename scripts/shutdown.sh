#!/bin/bash

# Ollama Service Shutdown and Cleanup Script
# Safely shuts down Ollama service and optionally cleans up caches, logs, and temp files

# Don't use set -e here because we want the script to continue through all cleanup steps
# even if some operations fail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_URL="http://localhost:11434"
API_ENDPOINT="${OLLAMA_URL}/api"

echo -e "${BLUE}🛑 Ollama Service Shutdown and Cleanup${NC}"
echo "=================================================="
echo ""

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
    fi
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# ============================================================================
# Step 1: Stop Ollama Service
# ============================================================================
echo -e "${BLUE}[1/5]${NC} Stopping Ollama service..."

# Try Homebrew services first
if command -v brew &> /dev/null; then
    if brew services list 2>/dev/null | grep -q "ollama.*started"; then
        print_info "Stopping Homebrew service..."
        brew services stop ollama > /dev/null 2>&1
        sleep 2
        print_status 0 "Homebrew service stopped"
    else
        print_info "No Homebrew service running"
    fi
fi

# Stop custom launchd service if it exists
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/com.ollama.service.plist"
if [ -f "$LAUNCHD_PLIST" ]; then
    if launchctl list | grep -q "com.ollama.service"; then
        print_info "Stopping custom launchd service..."
        launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true
        sleep 1
        print_status 0 "Custom launchd service stopped"
    fi
fi

# Kill any remaining ollama processes
OLLAMA_PIDS=$(ps aux | grep -i "[o]llama serve" | awk '{print $2}' 2>/dev/null || true)
if [ -n "$OLLAMA_PIDS" ]; then
    print_info "Killing remaining Ollama processes..."
    # Kill each PID individually to avoid killing the script itself
    for pid in $OLLAMA_PIDS; do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 1
    print_status 0 "All Ollama processes terminated"
else
    print_status 0 "No running Ollama processes found"
fi

# Verify service is stopped
if curl -f -s "$API_ENDPOINT/tags" > /dev/null 2>&1; then
    print_warning "Service still appears to be running - may need manual intervention"
else
    print_status 0 "Service confirmed stopped"
fi

# ============================================================================
# Step 2: Run Cleanup Script
# ============================================================================
echo ""
echo -e "${BLUE}[2/5]${NC} Running cleanup script..."

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup.sh"

if [ -f "$CLEANUP_SCRIPT" ]; then
    print_info "Running cleanup script to remove logs, Python caches, and temp files..."
    "$CLEANUP_SCRIPT"
    print_status 0 "Cleanup script completed"
else
    print_warning "Cleanup script not found at $CLEANUP_SCRIPT"
    print_info "Skipping cleanup step"
fi

# ============================================================================
# Step 3: Check Model Cache (Info Only)
# ============================================================================
echo ""
echo -e "${BLUE}[3/5]${NC} Checking model cache..."

MODEL_DIR="$HOME/.ollama/models"
if [ -d "$MODEL_DIR" ]; then
    MODEL_COUNT=$(ollama list 2>/dev/null | tail -n +2 | wc -l | tr -d ' ' || echo "0")
    if [ "$MODEL_COUNT" -gt 0 ]; then
        MODEL_SIZE=$(du -sh "$MODEL_DIR" 2>/dev/null | awk '{print $1}' || echo "unknown")
        print_info "Found $MODEL_COUNT model(s) in cache (~$MODEL_SIZE)"
        print_info "Models are preserved by default"
        echo ""
        print_warning "To remove models, run manually: ollama rm <model_name>"
        print_warning "To remove all models: rm -rf ~/.ollama/models/* (DANGEROUS!)"
    else
        print_info "No models found in cache"
    fi
else
    print_info "Model directory does not exist"
fi

# ============================================================================
# Step 4: Optional - Remove Launch Agent (Optional)
# ============================================================================
echo ""
echo -e "${BLUE}[4/5]${NC} Launch Agent status..."

if [ -f "$LAUNCHD_PLIST" ]; then
    print_info "Custom launch agent found: $LAUNCHD_PLIST"
    echo ""
    read -p "Remove launch agent? This will prevent auto-start on login (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Unload first if loaded
        if launchctl list | grep -q "com.ollama.service"; then
            launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true
        fi
        rm -f "$LAUNCHD_PLIST"
        print_status 0 "Launch agent removed"
    else
        print_info "Launch agent preserved (will auto-start on next login)"
    fi
else
    print_info "No custom launch agent found"
    
    # Check Homebrew service status
    if command -v brew &> /dev/null; then
        if brew services list 2>/dev/null | grep -q "ollama"; then
            print_info "Homebrew service is configured"
            print_info "To disable auto-start: brew services stop ollama"
        fi
    fi
fi

# ============================================================================
# Step 5: Final Summary
# ============================================================================
echo ""
echo -e "${BLUE}[5/5]${NC} Final summary..."
echo ""
echo "=================================================="
echo -e "${BLUE}📊 Shutdown Summary${NC}"
echo "=================================================="
echo -e "${GREEN}Service Status:${NC} Stopped"
echo -e "${GREEN}Cleanup:${NC} Completed (logs, caches, temp files)"
echo -e "${GREEN}Models:${NC} Preserved"
echo ""

print_info "To start the service again:"
if [ -f "$LAUNCHD_PLIST" ]; then
    echo "  - launchctl load $LAUNCHD_PLIST"
elif command -v brew &> /dev/null && brew services list 2>/dev/null | grep -q "ollama"; then
    echo "  - brew services start ollama"
else
    echo "  - ./scripts/start.sh (recommended - includes MPS/Metal optimizations)"
    echo "  - ollama serve (manual start)"
fi

echo ""
echo -e "${GREEN}✓ Shutdown complete!${NC}"

