#!/bin/bash

# Fix NVIDIA Docker GPU Access
# This script fixes the "could not select device driver" error

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Fixing NVIDIA Docker GPU Access                         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

print_status() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Step 1: Update daemon.json
echo -e "${YELLOW}Step 1: Updating Docker daemon configuration${NC}"
print_status "Backing up current daemon.json..."
if [ -f /etc/docker/daemon.json ]; then
    sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.backup
    print_success "Backup created at /etc/docker/daemon.json.backup"
fi

print_status "Creating new daemon.json configuration..."
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF
print_success "daemon.json updated"

# Step 2: Reload Docker daemon
echo ""
echo -e "${YELLOW}Step 2: Reloading Docker daemon${NC}"
print_status "Stopping Docker..."
sudo systemctl stop docker

print_status "Reloading systemd daemon..."
sudo systemctl daemon-reload

print_status "Starting Docker..."
sudo systemctl start docker

sleep 3

if sudo systemctl is-active --quiet docker; then
    print_success "Docker daemon restarted successfully"
else
    print_error "Failed to restart Docker daemon"
    exit 1
fi

# Step 3: Test GPU access
echo ""
echo -e "${YELLOW}Step 3: Testing GPU access${NC}"
print_status "Testing with --runtime=nvidia flag..."

if docker run --rm --runtime=nvidia nvidia/cuda:11.2.2-base-ubuntu18.04 nvidia-smi; then
    print_success "GPU is accessible with --runtime=nvidia flag!"
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    Fix Successful! 🎉                        ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Note:${NC} Use ${YELLOW}--runtime=nvidia${NC} instead of ${YELLOW}--gpus all${NC}"
    echo ""
    echo -e "Example:"
    echo -e "  docker run --rm ${YELLOW}--runtime=nvidia${NC} nvidia/cuda:11.2.2-base-ubuntu18.04 nvidia-smi"
    echo ""
else
    print_error "GPU access still not working"
    echo ""
    echo -e "${YELLOW}Additional troubleshooting needed:${NC}"
    echo "1. Check NVIDIA driver: nvidia-smi"
    echo "2. Check nvidia-container-toolkit: dpkg -l | grep nvidia-container"
    echo "3. Reboot your system: sudo reboot"
    exit 1
fi
