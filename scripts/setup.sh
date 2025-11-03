#!/bin/bash

# Shared Ollama Service Setup Script
# This script sets up the shared Ollama service with all required models

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Shared Ollama Service Setup${NC}"
echo "======================================"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker is installed${NC}"

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}✗ Docker Compose is not installed${NC}"
    echo "Install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose is installed${NC}"

echo ""
echo -e "${BLUE}Starting Ollama service...${NC}"
docker-compose up -d

echo ""
echo "Waiting for service to be ready..."
sleep 10

# Check if service is running
if ! curl -f -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${RED}✗ Service failed to start${NC}"
    echo "Check logs: docker-compose logs ollama"
    exit 1
fi
echo -e "${GREEN}✓ Service is running${NC}"

echo ""
echo -e "${BLUE}Pulling models...${NC}"
echo "This may take a while depending on your internet connection."
echo ""

# Define models to pull
MODELS=(
    "llama3.1:8b"
    "mistral"
)

# Pull models
for model in "${MODELS[@]}"; do
    echo -e "${YELLOW}Pulling ${model}...${NC}"
    if docker-compose exec -T ollama ollama pull "${model}" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ ${model} pulled successfully${NC}"
    else
        echo -e "${RED}✗ Failed to pull ${model}${NC}"
    fi
done

echo ""
echo -e "${BLUE}Running health check...${NC}"
./scripts/health_check.sh

echo ""
echo "======================================"
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Run health check: ./scripts/health_check.sh"
echo "2. Update project configurations to use http://localhost:11434"
echo "3. Test integration in your projects"
echo ""
echo "Service URL: http://localhost:11434"
echo "Available models:"
docker-compose exec -T ollama ollama list
echo ""
echo "View logs: docker-compose logs -f ollama"
