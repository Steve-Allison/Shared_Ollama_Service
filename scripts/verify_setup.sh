#!/bin/bash

# Comprehensive Ollama Setup Verification Script
# This is your one-stop-shop to verify everything is working correctly
# It checks installation, service status, model availability, and performs health checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
API_ENDPOINT="${OLLAMA_URL}/api"

# Required models
REQUIRED_MODELS=("qwen2.5vl:7b" "qwen2.5:14b")

# Status tracking
STEPS_PASSED=0
STEPS_FAILED=0
AUTO_FIXED=0

echo -e "${BLUE}🚀 Ollama Service - Complete Setup Verification${NC}"
echo "=================================================="
echo ""

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
        STEPS_PASSED=$((STEPS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $2"
        STEPS_FAILED=$((STEPS_FAILED + 1))
    fi
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# ============================================================================
# Step 1: Check Ollama Installation
# ============================================================================
echo -e "${BLUE}[1/6]${NC} Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 | head -n 1 || echo "installed")
    print_status 0 "Ollama is installed ($OLLAMA_VERSION)"
else
    print_status 1 "Ollama is not installed"
    echo ""
    print_info "Attempting to install Ollama..."
    
    # Try Homebrew first (macOS)
    if command -v brew &> /dev/null; then
        echo "Installing via Homebrew..."
        if brew install ollama > /dev/null 2>&1; then
            print_status 0 "Ollama installed via Homebrew"
            AUTO_FIXED=$((AUTO_FIXED + 1))
        else
            print_status 1 "Failed to install via Homebrew"
            echo "Please install manually: https://ollama.ai/download"
            exit 1
        fi
    else
        print_warning "Homebrew not found. Please install Ollama manually:"
        echo "  - Download: https://ollama.ai/download"
        echo "  - Or run: ./scripts/install_native.sh"
        exit 1
    fi
fi

# ============================================================================
# Step 2: Check/Start Ollama Service
# ============================================================================
echo ""
echo -e "${BLUE}[2/6]${NC} Checking Ollama service status..."
if curl -f -s "${API_ENDPOINT}/tags" > /dev/null 2>&1; then
    print_status 0 "Ollama service is running"
else
    print_status 1 "Ollama service is not accessible"
    echo ""
    print_info "Attempting to start service..."
    
    # Try to start via Homebrew services (macOS)
    if command -v brew &> /dev/null && brew services list 2>/dev/null | grep -q ollama; then
        echo "Starting via Homebrew services..."
        brew services start ollama > /dev/null 2>&1 || ollama serve > /dev/null 2>&1 &
    else
        echo "Starting Ollama service directly..."
        ollama serve > /dev/null 2>&1 &
    fi
    
    echo "Waiting 5 seconds for service to start..."
    sleep 5
    
    if curl -f -s "${API_ENDPOINT}/tags" > /dev/null 2>&1; then
        print_status 0 "Ollama service started successfully"
        ((AUTO_FIXED++))
    else
        print_status 1 "Failed to start Ollama service"
        echo ""
        print_warning "Try starting manually:"
        echo "  - ollama serve"
        echo "  - Or: brew services start ollama"
        exit 1
    fi
fi

# ============================================================================
# Step 3: Check Model Availability
# ============================================================================
echo ""
echo -e "${BLUE}[3/6]${NC} Checking model availability..."
MODELS_JSON=$(curl -s "${API_ENDPOINT}/tags" 2>/dev/null || echo "")
MODELS_LIST=$(echo "$MODELS_JSON" | jq -r '.models[].name' 2>/dev/null || echo "")

# Fallback: use ollama list if API doesn't have models yet
if [ -z "$MODELS_LIST" ] || [ "$MODELS_LIST" = "" ]; then
    MODELS_LIST=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}' || echo "")
fi

MISSING_MODELS=()
for model in "${REQUIRED_MODELS[@]}"; do
    if echo "$MODELS_LIST" | grep -q "^${model}$"; then
        print_status 0 "Model '${model}' is downloaded"
    else
        print_status 1 "Model '${model}' is NOT downloaded"
        MISSING_MODELS+=("$model")
    fi
done

# ============================================================================
# Step 4: Download Missing Models
# ============================================================================
MODELS_DOWNLOADED=false
if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo ""
    echo -e "${BLUE}[4/6]${NC} Downloading missing models..."
    print_info "Found ${#MISSING_MODELS[@]} missing model(s)"
    
    for model in "${MISSING_MODELS[@]}"; do
        echo ""
        print_info "Downloading ${model}..."
        echo "This may take several minutes depending on your connection..."
        
        # Run the pull command and capture output
        if ollama pull "${model}" > /tmp/ollama_pull_${model//[:\/]/_}.log 2>&1; then
            # Download command completed, verify the model exists
            sleep 2  # Brief wait for Ollama to register the model
            if ollama list 2>/dev/null | awk '{print $1}' | grep -q "^${model}$"; then
                print_status 0 "${model} downloaded successfully"
                AUTO_FIXED=$((AUTO_FIXED + 1))
                MODELS_DOWNLOADED=true
            else
                print_status 1 "Download completed but model not found in list"
            fi
        else
            # Pull command failed, but check if model exists anyway
            sleep 2
            if ollama list 2>/dev/null | awk '{print $1}' | grep -q "^${model}$"; then
                print_status 0 "${model} already exists"
                AUTO_FIXED=$((AUTO_FIXED + 1))
                MODELS_DOWNLOADED=true
            else
                print_status 1 "Failed to download ${model}"
            fi
        fi
        rm -f /tmp/ollama_pull_${model//[:\/]/_}.log
    done
else
    echo ""
    echo -e "${BLUE}[4/6]${NC} All models already downloaded - skipping"
    print_status 0 "No models to download"
fi

# Refresh models list after downloads
# Wait a moment for API to sync after downloads
if [ "$MODELS_DOWNLOADED" = true ]; then
    echo ""
    print_info "Waiting for API to sync..."
    sleep 3
fi

MODELS_JSON=$(curl -s "${API_ENDPOINT}/tags" 2>/dev/null || echo "")
MODELS_LIST=$(echo "$MODELS_JSON" | jq -r '.models[].name' 2>/dev/null || echo "")

# Fallback: use ollama list if API doesn't have models yet
if [ -z "$MODELS_LIST" ] || [ "$MODELS_LIST" = "" ]; then
    print_warning "API not returning models, using ollama list as fallback"
    MODELS_LIST=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}' || echo "")
fi

# ============================================================================
# Step 5: Verify All Required Models
# ============================================================================
echo ""
echo -e "${BLUE}[5/6]${NC} Verifying all required models..."
ALL_MODELS_PRESENT=true
for model in "${REQUIRED_MODELS[@]}"; do
    if echo "$MODELS_LIST" | grep -q "^${model}$"; then
        print_status 0 "Model '${model}' verified"
    else
        print_status 1 "Model '${model}' still missing"
        ALL_MODELS_PRESENT=false
    fi
done

# ============================================================================
# Step 6: Health Check - Test Model Generation
# ============================================================================
echo ""
echo -e "${BLUE}[6/6]${NC} Running health check (model generation test)..."
if echo "$MODELS_LIST" | grep -q "qwen2.5vl:7b"; then
    print_info "Testing model generation with qwen2.5vl:7b..."
    TEST_PROMPT='{"model": "qwen2.5vl:7b", "prompt": "Say hello in one word", "stream": false}'
    
    RESPONSE=$(curl -s -X POST "${API_ENDPOINT}/generate" \
        -H "Content-Type: application/json" \
        -d "$TEST_PROMPT" 2>/dev/null)
    
    if echo "$RESPONSE" | jq -r '.response' > /dev/null 2>&1; then
        MODEL_RESPONSE=$(echo "$RESPONSE" | jq -r '.response' | head -c 50)
        print_status 0 "Model generation test passed"
        print_info "Response preview: ${MODEL_RESPONSE}..."
    else
        print_status 1 "Model generation test failed"
        print_warning "Model may need more time to initialize"
    fi
else
    print_warning "Cannot test generation - primary model not available"
fi

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=================================================="
echo -e "${BLUE}📊 Verification Summary${NC}"
echo "=================================================="
echo -e "${GREEN}Passed: ${STEPS_PASSED}${NC}"
echo -e "${RED}Failed: ${STEPS_FAILED}${NC}"
if [ $AUTO_FIXED -gt 0 ]; then
    echo -e "${BLUE}Auto-fixed: ${AUTO_FIXED}${NC}"
fi
echo ""

# Display service info
if curl -f -s "${API_ENDPOINT}/tags" > /dev/null 2>&1; then
    echo -e "${GREEN}Service Status:${NC}"
    echo "  URL: $OLLAMA_URL"
    echo "  Status: Running"
    echo ""
    echo -e "${GREEN}Available Models:${NC}"
    if [ -n "$MODELS_LIST" ]; then
        echo "$MODELS_LIST" | while read -r model; do
            echo "  - $model"
        done
    else
        echo "  (none)"
    fi
    echo ""
    
    # Storage info
    if [ -d ~/.ollama/models ]; then
        STORAGE_SIZE=$(du -sh ~/.ollama/models 2>/dev/null | awk '{print $1}' || echo "unknown")
        echo -e "${GREEN}Storage:${NC}"
        echo "  Location: ~/.ollama/models"
        echo "  Size: $STORAGE_SIZE"
    fi
fi

echo ""
if [ $STEPS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓✓✓ All checks passed! Ollama service is ready to use.${NC}"
    echo ""
    echo "Next steps:"
    echo "  - Start using models: curl http://localhost:11434/api/generate -d '{\"model\":\"qwen2.5vl:7b\",\"prompt\":\"Hello\"}'"
    echo "  - Warm up models: ./scripts/warmup_models.sh"
    echo "  - View logs: tail -f ./logs/ollama.log"
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please review above and fix issues.${NC}"
    exit 1
fi

