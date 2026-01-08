#!/bin/bash

# Simple GPU Test Script for DirectionNet Docker Container
# This script tests different methods to enable GPU access

set +e  # Don't exit on error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     DirectionNet GPU Access Test                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Test 1: Standard --gpus all flag
echo -e "${YELLOW}Test 1: Using --gpus all flag${NC}"
if docker run --rm --gpus all directionnet:latest python3.6 -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))" 2>/dev/null; then
    echo -e "${GREEN}✓ Method 1 works: --gpus all${NC}"
    echo "USE_METHOD=gpus" > /tmp/docker_gpu_method.txt
    exit 0
else
    echo -e "${RED}✗ Method 1 failed${NC}"
fi
echo ""

# Test 2: Using --runtime=nvidia
echo -e "${YELLOW}Test 2: Using --runtime=nvidia flag${NC}"
if docker run --rm --runtime=nvidia directionnet:latest python3.6 -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))" 2>/dev/null; then
    echo -e "${GREEN}✓ Method 2 works: --runtime=nvidia${NC}"
    echo "USE_METHOD=runtime" > /tmp/docker_gpu_method.txt
    exit 0
else
    echo -e "${RED}✗ Method 2 failed${NC}"
fi
echo ""

# Test 3: Manual library mounting
echo -e "${YELLOW}Test 3: Manual library and device mounting${NC}"
if docker run --rm \
    --device=/dev/nvidiactl \
    --device=/dev/nvidia-uvm \
    --device=/dev/nvidia-uvm-tools \
    --device=/dev/nvidia-modeset \
    --device=/dev/nvidia0 \
    -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
    -v /usr/lib/x86_64-linux-gnu/libcuda.so.570.195.03:/usr/local/cuda/lib64/libcuda.so.1:ro \
    -v /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.195.03:/usr/local/cuda/lib64/libnvidia-ml.so.1:ro \
    -v /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.570.195.03:/usr/local/cuda/lib64/libnvidia-ptxjitcompiler.so.1:ro \
    -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    directionnet:latest \
    python3.6 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPUs:', len(gpus)); exit(0 if len(gpus) > 0 else 1)" 2>&1 | grep -q "GPUs: [1-9]"; then
    echo -e "${GREEN}✓ Method 3 works: Manual mounting${NC}"
    echo "USE_METHOD=manual" > /tmp/docker_gpu_method.txt
    exit 0
else
    echo -e "${RED}✗ Method 3 failed${NC}"
fi
echo ""

# All methods failed
echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║  All GPU access methods failed                               ║${NC}"
echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Recommendations:${NC}"
echo "1. Reboot your system: sudo reboot"
echo "2. After reboot, run this script again"
echo "3. If still failing, check: docker info | grep -i runtime"
echo ""
exit 1
