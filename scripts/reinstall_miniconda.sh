#!/bin/bash

# Miniconda Reinstallation Script
# This script reinstalls miniconda while preserving existing environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

MINICONDA_DIR="/Users/steveallison/opt/miniconda3"
ENVS_DIR="${MINICONDA_DIR}/envs"
BACKUP_DIR="${MINICONDA_DIR}_backup_$(date +%Y%m%d_%H%M%S)"

echo -e "${BLUE}🔄 Miniconda Reinstallation Script${NC}"
echo "=========================================="
echo ""

# Check if miniconda exists
if [ ! -d "$MINICONDA_DIR" ]; then
    echo -e "${RED}✗ Miniconda not found at $MINICONDA_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Miniconda found at: $MINICONDA_DIR${NC}"
echo ""

# List environments
echo -e "${BLUE}Existing environments:${NC}"
if [ -d "$ENVS_DIR" ]; then
    ls -1 "$ENVS_DIR" | grep -v "^\.DS_Store$" | grep -v "^\.conda_envs_dir_test$" | while read env; do
        if [ -d "$ENVS_DIR/$env" ]; then
            SIZE=$(du -sh "$ENVS_DIR/$env" 2>/dev/null | awk '{print $1}')
            echo "  - $env ($SIZE)"
        fi
    done
else
    echo "  No environments found"
fi

echo ""
echo -e "${YELLOW}⚠ WARNING: This will:${NC}"
echo "  1. Backup current miniconda installation"
echo "  2. Remove the base environment (but preserve other envs)"
echo "  3. Download and reinstall miniconda"
echo "  4. Restore your environments"
echo ""
# Auto-confirm if CONDA_REINSTALL_AUTO is set
if [ "${CONDA_REINSTALL_AUTO:-}" != "yes" ]; then
    read -p "Continue? (yes/no): " -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
else
    echo "Auto-confirming (CONDA_REINSTALL_AUTO=yes)..."
fi

# Step 1: Backup environments
echo -e "${BLUE}[1/4]${NC} Backing up environments..."
if [ -d "$ENVS_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
    cp -R "$ENVS_DIR" "$BACKUP_DIR/envs"
    echo -e "${GREEN}✓ Environments backed up to: $BACKUP_DIR/envs${NC}"
else
    echo -e "${YELLOW}⚠ No environments to backup${NC}"
fi

# Step 2: Remove miniconda (but preserve envs)
echo ""
echo -e "${BLUE}[2/4]${NC} Removing old miniconda installation..."
if [ -d "$ENVS_DIR" ]; then
    # Temporarily move envs out of the way
    mv "$ENVS_DIR" "${ENVS_DIR}_temp"
    echo -e "${GREEN}✓ Environments temporarily moved${NC}"
fi

# Remove everything except the envs
cd "$MINICONDA_DIR/.."
rm -rf miniconda3
echo -e "${GREEN}✓ Old installation removed${NC}"

# Step 3: Download and install miniconda
echo ""
echo -e "${BLUE}[3/4]${NC} Downloading and installing miniconda..."

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" == "arm64" ] || [ "$ARCH" == "aarch64" ]; then
    INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
else
    INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
fi

INSTALLER_URL="https://repo.anaconda.com/miniconda/${INSTALLER}"
INSTALLER_PATH="/tmp/${INSTALLER}"

echo "Downloading: $INSTALLER_URL"
curl -L -o "$INSTALLER_PATH" "$INSTALLER_URL"

echo "Installing miniconda..."
bash "$INSTALLER_PATH" -b -p "$MINICONDA_DIR" -u

# Clean up installer
rm -f "$INSTALLER_PATH"
echo -e "${GREEN}✓ Miniconda installed${NC}"

# Step 4: Restore environments
echo ""
echo -e "${BLUE}[4/4]${NC} Restoring environments..."
if [ -d "${ENVS_DIR}_temp" ]; then
    mv "${ENVS_DIR}_temp" "$ENVS_DIR"
    echo -e "${GREEN}✓ Environments restored${NC}"
elif [ -d "$BACKUP_DIR/envs" ]; then
    cp -R "$BACKUP_DIR/envs"/* "$ENVS_DIR/"
    echo -e "${GREEN}✓ Environments restored from backup${NC}"
else
    echo -e "${YELLOW}⚠ No environments to restore${NC}"
fi

# Initialize conda for shell
echo ""
echo -e "${BLUE}Initializing conda...${NC}"
"$MINICONDA_DIR/bin/conda" init zsh 2>/dev/null || "$MINICONDA_DIR/bin/conda" init bash 2>/dev/null || true

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Reinstallation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Restart your terminal or run: source ~/.zshrc"
echo "  2. Verify environments: conda env list"
echo "  3. Test an environment: conda activate ai_prompts"
echo ""
echo "Backup location: $BACKUP_DIR"
echo "You can remove it after verifying everything works."

