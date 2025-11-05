#!/bin/bash

# Cleanup Script for Shared Ollama Service
# Removes log files, Python caches, and other temporary files

# Don't use set -e here because we want the script to continue through all cleanup steps
# even if some operations fail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}🧹 Cleanup Script for Shared Ollama Service${NC}"
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

# Function to format file sizes
format_size() {
    local size=$1
    if [ $size -lt 1024 ]; then
        echo "${size}B"
    elif [ $size -lt 1048576 ]; then
        echo "$((size / 1024))KB"
    else
        echo "$((size / 1024 / 1024))MB"
    fi
}

# ============================================================================
# Step 1: Clean Log Files
# ============================================================================
echo -e "${BLUE}[1/5]${NC} Cleaning log files..."

LOG_DIR="$PROJECT_ROOT/logs"
OLD_LOG_DIR="$HOME/.ollama"
LOG_FILES=(
    "$LOG_DIR/ollama.log"
    "$LOG_DIR/ollama.error.log"
    "$OLD_LOG_DIR/ollama.log"
    "$OLD_LOG_DIR/ollama.error.log"
)

LOG_SIZE=0
LOG_COUNT=0
for log_file in "${LOG_FILES[@]}"; do
    if [ -f "$log_file" ]; then
        SIZE=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null || echo "0")
        LOG_SIZE=$((LOG_SIZE + SIZE))
        LOG_COUNT=$((LOG_COUNT + 1))
    fi
done

# Also check for any other .log files in the project
if [ -d "$LOG_DIR" ]; then
    OTHER_LOGS=$(find "$LOG_DIR" -maxdepth 1 -name "*.log" -type f 2>/dev/null | grep -v -E "(ollama\.log|ollama\.error\.log)" || true)
    if [ -n "$OTHER_LOGS" ]; then
        while IFS= read -r log_file; do
            if [ -f "$log_file" ]; then
                SIZE=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null || echo "0")
                LOG_SIZE=$((LOG_SIZE + SIZE))
                LOG_COUNT=$((LOG_COUNT + 1))
            fi
        done <<< "$OTHER_LOGS"
    fi
fi

if [ $LOG_COUNT -gt 0 ]; then
    LOG_SIZE_FORMATTED=$(format_size $LOG_SIZE)
    print_info "Found $LOG_COUNT log file(s) totaling ~${LOG_SIZE_FORMATTED}"
    
    # Remove log files (not just truncate)
    for log_file in "${LOG_FILES[@]}"; do
        if [ -f "$log_file" ]; then
            rm -f "$log_file" 2>/dev/null || true
        fi
    done
    
    # Clean other log files
    if [ -n "$OTHER_LOGS" ]; then
        while IFS= read -r log_file; do
            rm -f "$log_file" 2>/dev/null || true
        done <<< "$OTHER_LOGS"
    fi
    
    print_status 0 "Removed $LOG_COUNT log file(s)"
else
    print_status 0 "No log files found"
fi

# Remove logs directory if it exists and is empty (or only contains .gitkeep)
# Since logs/ is gitignored, .gitkeep isn't tracked anyway, so we can remove the whole directory
if [ -d "$LOG_DIR" ]; then
    # Check if directory is empty or only contains .gitkeep
    CONTENTS=$(find "$LOG_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ' || echo "0")
    if [ "$CONTENTS" -eq 0 ] || ([ "$CONTENTS" -eq 1 ] && [ -f "$LOG_DIR/.gitkeep" ]); then
        rm -rf "$LOG_DIR" 2>/dev/null || true
        print_status 0 "Removed empty logs directory"
    fi
fi

# ============================================================================
# Step 2: Clean Python Caches
# ============================================================================
echo ""
echo -e "${BLUE}[2/5]${NC} Cleaning Python caches..."

PYCACHE_COUNT=0
PYCACHE_SIZE=0

# Find and remove __pycache__ directories
PYCACHE_DIRS=$(find "$PROJECT_ROOT" -type d -name "__pycache__" 2>/dev/null || true)
if [ -n "$PYCACHE_DIRS" ]; then
    while IFS= read -r cache_dir; do
        if [ -d "$cache_dir" ]; then
            SIZE=$(du -sk "$cache_dir" 2>/dev/null | awk '{print $1}' || echo "0")
            PYCACHE_SIZE=$((PYCACHE_SIZE + SIZE))
            PYCACHE_COUNT=$((PYCACHE_COUNT + 1))
            rm -rf "$cache_dir" 2>/dev/null || true
        fi
    done <<< "$PYCACHE_DIRS"
fi

# Find and remove .pyc, .pyo, .pyd files
PYC_FILES=$(find "$PROJECT_ROOT" -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) 2>/dev/null || true)
if [ -n "$PYC_FILES" ]; then
    PYC_COUNT=$(echo "$PYC_FILES" | wc -l | tr -d ' ')
    while IFS= read -r pyc_file; do
        if [ -f "$pyc_file" ]; then
            SIZE=$(stat -f%z "$pyc_file" 2>/dev/null || stat -c%s "$pyc_file" 2>/dev/null || echo "0")
            PYCACHE_SIZE=$((PYCACHE_SIZE + SIZE / 1024))
            rm -f "$pyc_file" 2>/dev/null || true
        fi
    done <<< "$PYC_FILES"
fi

# Remove .pytest_cache
if [ -d "$PROJECT_ROOT/.pytest_cache" ]; then
    SIZE=$(du -sk "$PROJECT_ROOT/.pytest_cache" 2>/dev/null | awk '{print $1}' || echo "0")
    PYCACHE_SIZE=$((PYCACHE_SIZE + SIZE))
    PYCACHE_COUNT=$((PYCACHE_COUNT + 1))
    rm -rf "$PROJECT_ROOT/.pytest_cache" 2>/dev/null || true
fi

# Remove .ruff_cache
if [ -d "$PROJECT_ROOT/.ruff_cache" ]; then
    SIZE=$(du -sk "$PROJECT_ROOT/.ruff_cache" 2>/dev/null | awk '{print $1}' || echo "0")
    PYCACHE_SIZE=$((PYCACHE_SIZE + SIZE))
    PYCACHE_COUNT=$((PYCACHE_COUNT + 1))
    rm -rf "$PROJECT_ROOT/.ruff_cache" 2>/dev/null || true
fi

# Remove .mypy_cache
if [ -d "$PROJECT_ROOT/.mypy_cache" ]; then
    SIZE=$(du -sk "$PROJECT_ROOT/.mypy_cache" 2>/dev/null | awk '{print $1}' || echo "0")
    PYCACHE_SIZE=$((PYCACHE_SIZE + SIZE))
    PYCACHE_COUNT=$((PYCACHE_COUNT + 1))
    rm -rf "$PROJECT_ROOT/.mypy_cache" 2>/dev/null || true
fi

# Remove .pyright (if exists as directory)
if [ -d "$PROJECT_ROOT/.pyright" ]; then
    SIZE=$(du -sk "$PROJECT_ROOT/.pyright" 2>/dev/null | awk '{print $1}' || echo "0")
    PYCACHE_SIZE=$((PYCACHE_SIZE + SIZE))
    PYCACHE_COUNT=$((PYCACHE_COUNT + 1))
    rm -rf "$PROJECT_ROOT/.pyright" 2>/dev/null || true
fi

if [ $PYCACHE_COUNT -gt 0 ] || [ $PYCACHE_SIZE -gt 0 ]; then
    PYCACHE_SIZE_FORMATTED=$(format_size $((PYCACHE_SIZE * 1024)))
    print_status 0 "Cleaned Python caches (~${PYCACHE_SIZE_FORMATTED})"
else
    print_status 0 "No Python caches found"
fi

# ============================================================================
# Step 3: Clean Temporary Files
# ============================================================================
echo ""
echo -e "${BLUE}[3/5]${NC} Cleaning temporary files..."

TEMP_FILES_CLEANED=0
TEMP_SIZE=0

# Clean up temp pull logs
if ls /tmp/ollama_pull_*.log 2>/dev/null | grep -q .; then
    TEMP_COUNT=$(ls /tmp/ollama_pull_*.log 2>/dev/null | wc -l | tr -d ' ')
    TEMP_FILES_CLEANED=$((TEMP_FILES_CLEANED + TEMP_COUNT))
    for temp_file in /tmp/ollama_pull_*.log; do
        if [ -f "$temp_file" ]; then
            SIZE=$(stat -f%z "$temp_file" 2>/dev/null || stat -c%s "$temp_file" 2>/dev/null || echo "0")
            TEMP_SIZE=$((TEMP_SIZE + SIZE))
        fi
        rm -f "$temp_file" 2>/dev/null || true
    done
fi

# Clean up any other ollama temp files
if [ -d "/tmp" ]; then
    TEMP_OLLAMA_FILES=$(find /tmp -maxdepth 1 -name "*ollama*" -type f 2>/dev/null || true)
    if [ -n "$TEMP_OLLAMA_FILES" ]; then
        while IFS= read -r temp_file; do
            if [ -f "$temp_file" ]; then
                SIZE=$(stat -f%z "$temp_file" 2>/dev/null || stat -c%s "$temp_file" 2>/dev/null || echo "0")
                TEMP_SIZE=$((TEMP_SIZE + SIZE))
                TEMP_FILES_CLEANED=$((TEMP_FILES_CLEANED + 1))
                rm -f "$temp_file" 2>/dev/null || true
            fi
        done <<< "$TEMP_OLLAMA_FILES"
    fi
fi

# Clean up .tmp, .bak, .swp files in project
TEMP_PROJECT_FILES=$(find "$PROJECT_ROOT" -type f \( -name "*.tmp" -o -name "*.bak" -o -name "*.swp" -o -name "*~" \) 2>/dev/null || true)
if [ -n "$TEMP_PROJECT_FILES" ]; then
    while IFS= read -r temp_file; do
        if [ -f "$temp_file" ]; then
            SIZE=$(stat -f%z "$temp_file" 2>/dev/null || stat -c%s "$temp_file" 2>/dev/null || echo "0")
            TEMP_SIZE=$((TEMP_SIZE + SIZE))
            TEMP_FILES_CLEANED=$((TEMP_FILES_CLEANED + 1))
            rm -f "$temp_file" 2>/dev/null || true
        fi
    done <<< "$TEMP_PROJECT_FILES"
fi

if [ $TEMP_FILES_CLEANED -gt 0 ]; then
    TEMP_SIZE_FORMATTED=$(format_size $TEMP_SIZE)
    print_status 0 "Cleaned $TEMP_FILES_CLEANED temporary file(s) (~${TEMP_SIZE_FORMATTED})"
else
    print_status 0 "No temporary files found"
fi

# ============================================================================
# Step 4: Clean Build Artifacts
# ============================================================================
echo ""
echo -e "${BLUE}[4/5]${NC} Cleaning build artifacts..."

BUILD_CLEANED=0
BUILD_SIZE=0

# Remove build/ directory
if [ -d "$PROJECT_ROOT/build" ]; then
    SIZE=$(du -sk "$PROJECT_ROOT/build" 2>/dev/null | awk '{print $1}' || echo "0")
    BUILD_SIZE=$((BUILD_SIZE + SIZE))
    BUILD_CLEANED=$((BUILD_CLEANED + 1))
    rm -rf "$PROJECT_ROOT/build" 2>/dev/null || true
fi

# Remove dist/ directory
if [ -d "$PROJECT_ROOT/dist" ]; then
    SIZE=$(du -sk "$PROJECT_ROOT/dist" 2>/dev/null | awk '{print $1}' || echo "0")
    BUILD_SIZE=$((BUILD_SIZE + SIZE))
    BUILD_CLEANED=$((BUILD_CLEANED + 1))
    rm -rf "$PROJECT_ROOT/dist" 2>/dev/null || true
fi

# Remove *.egg-info directories
EGG_INFO_DIRS=$(find "$PROJECT_ROOT" -maxdepth 2 -type d -name "*.egg-info" 2>/dev/null || true)
if [ -n "$EGG_INFO_DIRS" ]; then
    while IFS= read -r egg_dir; do
        if [ -d "$egg_dir" ]; then
            SIZE=$(du -sk "$egg_dir" 2>/dev/null | awk '{print $1}' || echo "0")
            BUILD_SIZE=$((BUILD_SIZE + SIZE))
            BUILD_CLEANED=$((BUILD_CLEANED + 1))
            rm -rf "$egg_dir" 2>/dev/null || true
        fi
    done <<< "$EGG_INFO_DIRS"
fi

# Remove .eggs/ directory
if [ -d "$PROJECT_ROOT/.eggs" ]; then
    SIZE=$(du -sk "$PROJECT_ROOT/.eggs" 2>/dev/null | awk '{print $1}' || echo "0")
    BUILD_SIZE=$((BUILD_SIZE + SIZE))
    BUILD_CLEANED=$((BUILD_CLEANED + 1))
    rm -rf "$PROJECT_ROOT/.eggs" 2>/dev/null || true
fi

# Remove *.egg, *.whl, *.tar.gz files
BUILD_FILES=$(find "$PROJECT_ROOT" -maxdepth 1 -type f \( -name "*.egg" -o -name "*.whl" -o -name "*.tar.gz" \) 2>/dev/null || true)
if [ -n "$BUILD_FILES" ]; then
    while IFS= read -r build_file; do
        if [ -f "$build_file" ]; then
            SIZE=$(stat -f%z "$build_file" 2>/dev/null || stat -c%s "$build_file" 2>/dev/null || echo "0")
            BUILD_SIZE=$((BUILD_SIZE + SIZE / 1024))
            BUILD_CLEANED=$((BUILD_CLEANED + 1))
            rm -f "$build_file" 2>/dev/null || true
        fi
    done <<< "$BUILD_FILES"
fi

if [ $BUILD_CLEANED -gt 0 ]; then
    BUILD_SIZE_FORMATTED=$(format_size $((BUILD_SIZE * 1024)))
    print_status 0 "Cleaned build artifacts (~${BUILD_SIZE_FORMATTED})"
else
    print_status 0 "No build artifacts found"
fi

# ============================================================================
# Step 5: Clean Coverage/Test Artifacts
# ============================================================================
echo ""
echo -e "${BLUE}[5/5]${NC} Cleaning test artifacts..."

TEST_CLEANED=0
TEST_SIZE=0

# Remove .coverage file
if [ -f "$PROJECT_ROOT/.coverage" ]; then
    SIZE=$(stat -f%z "$PROJECT_ROOT/.coverage" 2>/dev/null || stat -c%s "$PROJECT_ROOT/.coverage" 2>/dev/null || echo "0")
    TEST_SIZE=$((TEST_SIZE + SIZE))
    TEST_CLEANED=$((TEST_CLEANED + 1))
    rm -f "$PROJECT_ROOT/.coverage" 2>/dev/null || true
fi

# Remove htmlcov/ directory
if [ -d "$PROJECT_ROOT/htmlcov" ]; then
    SIZE=$(du -sk "$PROJECT_ROOT/htmlcov" 2>/dev/null | awk '{print $1}' || echo "0")
    TEST_SIZE=$((TEST_SIZE + SIZE))
    TEST_CLEANED=$((TEST_CLEANED + 1))
    rm -rf "$PROJECT_ROOT/htmlcov" 2>/dev/null || true
fi

# Remove .hypothesis/ directory
if [ -d "$PROJECT_ROOT/.hypothesis" ]; then
    SIZE=$(du -sk "$PROJECT_ROOT/.hypothesis" 2>/dev/null | awk '{print $1}' || echo "0")
    TEST_SIZE=$((TEST_SIZE + SIZE))
    TEST_CLEANED=$((TEST_CLEANED + 1))
    rm -rf "$PROJECT_ROOT/.hypothesis" 2>/dev/null || true
fi

# Remove .tox/ directory
if [ -d "$PROJECT_ROOT/.tox" ]; then
    SIZE=$(du -sk "$PROJECT_ROOT/.tox" 2>/dev/null | awk '{print $1}' || echo "0")
    TEST_SIZE=$((TEST_SIZE + SIZE))
    TEST_CLEANED=$((TEST_CLEANED + 1))
    rm -rf "$PROJECT_ROOT/.tox" 2>/dev/null || true
fi

# Remove *.cover files
COVER_FILES=$(find "$PROJECT_ROOT" -type f -name "*.cover" 2>/dev/null || true)
if [ -n "$COVER_FILES" ]; then
    while IFS= read -r cover_file; do
        if [ -f "$cover_file" ]; then
            SIZE=$(stat -f%z "$cover_file" 2>/dev/null || stat -c%s "$cover_file" 2>/dev/null || echo "0")
            TEST_SIZE=$((TEST_SIZE + SIZE))
            TEST_CLEANED=$((TEST_CLEANED + 1))
            rm -f "$cover_file" 2>/dev/null || true
        fi
    done <<< "$COVER_FILES"
fi

if [ $TEST_CLEANED -gt 0 ]; then
    TEST_SIZE_FORMATTED=$(format_size $TEST_SIZE)
    print_status 0 "Cleaned test artifacts (~${TEST_SIZE_FORMATTED})"
else
    print_status 0 "No test artifacts found"
fi

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=================================================="
echo -e "${BLUE}📊 Cleanup Summary${NC}"
echo "=================================================="

TOTAL_SIZE=$((LOG_SIZE + PYCACHE_SIZE * 1024 + TEMP_SIZE + BUILD_SIZE * 1024 + TEST_SIZE))
TOTAL_SIZE_FORMATTED=$(format_size $TOTAL_SIZE)

echo -e "${GREEN}Logs:${NC} Cleaned"
echo -e "${GREEN}Python Caches:${NC} Cleaned"
echo -e "${GREEN}Temp Files:${NC} Cleaned"
echo -e "${GREEN}Build Artifacts:${NC} Cleaned"
echo -e "${GREEN}Test Artifacts:${NC} Cleaned"
echo ""
echo -e "${BLUE}Total space freed:${NC} ~${TOTAL_SIZE_FORMATTED}"
echo ""
echo -e "${GREEN}✓ Cleanup complete!${NC}"

