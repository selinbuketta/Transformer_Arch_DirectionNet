#!/bin/bash

# Docker Context Helper for DirectionNet
# This ensures you're using the correct Docker context with GPU support

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Docker Context Configuration for GPU Support            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check current context
CURRENT_CONTEXT=$(docker context show)
echo -e "${YELLOW}Current Docker context:${NC} $CURRENT_CONTEXT"
echo ""

# Check if NVIDIA runtime is available in current context
echo -e "${YELLOW}Checking NVIDIA runtime availability...${NC}"
NVIDIA_RUNTIME=$(docker info 2>/dev/null | grep -i "Runtimes:" | grep -i "nvidia")

if [ -n "$NVIDIA_RUNTIME" ]; then
    echo -e "${GREEN}✓ NVIDIA runtime is available in current context${NC}"
    docker info | grep -i runtime
    echo ""
    echo -e "${GREEN}You're all set! GPU access should work.${NC}"
    exit 0
else
    echo -e "${RED}✗ NVIDIA runtime is NOT available in current context${NC}"
    echo ""
fi

# If not in default context, switch to it
if [ "$CURRENT_CONTEXT" != "default" ]; then
    echo -e "${YELLOW}Switching to 'default' Docker context for GPU support...${NC}"
    docker context use default
    echo ""
    
    # Check again
    NVIDIA_RUNTIME_DEFAULT=$(docker info 2>/dev/null | grep -i "Runtimes:" | grep -i "nvidia")
    if [ -n "$NVIDIA_RUNTIME_DEFAULT" ]; then
        echo -e "${GREEN}✓ NVIDIA runtime is now available!${NC}"
        docker info | grep -i runtime
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  GPU Support Enabled!                                        ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "You can now use GPU with Docker commands."
        echo "Example: docker run --gpus all directionnet:latest"
        exit 0
    else
        echo -e "${RED}✗ NVIDIA runtime still not available${NC}"
        echo ""
        echo "Please run the following commands to fix this:"
        echo ""
        echo -e "${YELLOW}sudo nvidia-ctk runtime configure --runtime=docker --set-as-default${NC}"
        echo -e "${YELLOW}sudo systemctl restart docker${NC}"
        echo -e "${YELLOW}docker context use default${NC}"
        exit 1
    fi
fi

echo -e "${YELLOW}You are already using the 'default' context, but NVIDIA runtime is not available.${NC}"
echo ""
echo "Please run the following commands to configure it:"
echo ""
echo -e "${YELLOW}sudo nvidia-ctk runtime configure --runtime=docker --set-as-default${NC}"
echo -e "${YELLOW}sudo systemctl restart docker${NC}"
exit 1
