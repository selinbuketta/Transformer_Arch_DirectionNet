#!/bin/bash

# DirectionNet Docker Complete Setup Script
# Run this script to set up Docker environment for GPU support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     DirectionNet Docker Setup - GPU Enabled                 ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Step 1: Check prerequisites
echo -e "\n${YELLOW}═══════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Step 1: Checking Prerequisites${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}\n"

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_success "Docker is installed: $DOCKER_VERSION"
else
    print_error "Docker is not installed!"
    exit 1
fi

# Check NVIDIA drivers
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA drivers are installed"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | while read line; do
        echo "  → $line"
    done
else
    print_error "NVIDIA drivers are not installed!"
    exit 1
fi

# Step 2: Start Docker daemon
echo -e "\n${YELLOW}═══════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Step 2: Starting Docker Daemon${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}\n"

print_status "Checking Docker daemon status..."
if sudo systemctl is-active --quiet docker; then
    print_success "Docker daemon is already running"
else
    print_status "Starting Docker daemon..."
    sudo systemctl start docker
    sleep 2
    if sudo systemctl is-active --quiet docker; then
        print_success "Docker daemon started successfully"
    else
        print_error "Failed to start Docker daemon"
        exit 1
    fi
fi

# Enable Docker to start on boot
print_status "Enabling Docker to start on boot..."
sudo systemctl enable docker &> /dev/null
print_success "Docker will start automatically on boot"

# Step 3: Check/Install nvidia-docker2
echo -e "\n${YELLOW}═══════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Step 3: Checking NVIDIA Docker Runtime${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}\n"

if dpkg -l | grep -q nvidia-docker2; then
    print_success "nvidia-docker2 is installed"
else
    print_warning "nvidia-docker2 is not installed"
    echo ""
    read -p "Do you want to install nvidia-docker2? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Adding NVIDIA Docker repository..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        
        print_status "Installing nvidia-docker2..."
        sudo apt-get update
        sudo apt-get install -y nvidia-docker2
        
        print_status "Restarting Docker daemon..."
        sudo systemctl restart docker
        sleep 2
        print_success "nvidia-docker2 installed successfully"
    else
        print_warning "Skipping nvidia-docker2 installation"
    fi
fi

# Step 4: Test GPU access
echo -e "\n${YELLOW}═══════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Step 4: Testing GPU Access in Docker${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}\n"

print_status "Testing GPU access with Docker..."
if docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu18.04 nvidia-smi &> /dev/null; then
    print_success "GPU is accessible from Docker containers!"
    echo ""
    docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu18.04 nvidia-smi
else
    print_error "GPU is not accessible from Docker containers"
    print_warning "You may need to restart the Docker daemon or reboot your system"
    exit 1
fi

# Step 5: Build DirectionNet Docker image
echo -e "\n${YELLOW}═══════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Step 5: Building DirectionNet Docker Image${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}\n"

print_status "Building Docker image (this may take 10-15 minutes)..."
echo ""

if docker build -t directionnet:latest .; then
    print_success "Docker image built successfully!"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Step 6: Test the built image
echo -e "\n${YELLOW}═══════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Step 6: Testing Built Image${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}\n"

print_status "Testing Python version..."
PYTHON_VERSION=$(docker run --rm directionnet:latest python3.6 --version)
print_success "Python version: $PYTHON_VERSION"

print_status "Testing TensorFlow GPU availability..."
TF_GPU=$(docker run --rm --gpus all directionnet:latest python3.6 -c "import tensorflow as tf; print('GPUs Available:', len(tf.config.list_physical_devices('GPU')))")
echo "  → $TF_GPU"

print_status "Testing installed packages..."
docker run --rm directionnet:latest python3.6 -c "import tensorflow, numpy, matplotlib; print('✓ Core packages imported successfully')"

# Final summary
echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete! 🎉                        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo -e "  1. Run interactively:"
echo -e "     ${YELLOW}./docker-run.sh run${NC}"
echo ""
echo -e "  2. Run training:"
echo -e "     ${YELLOW}./docker-run.sh train --your_flags${NC}"
echo ""
echo -e "  3. Run evaluation:"
echo -e "     ${YELLOW}./docker-run.sh eval --your_flags${NC}"
echo ""
echo -e "  4. Start TensorBoard:"
echo -e "     ${YELLOW}./docker-run.sh tensorboard${NC}"
echo -e "     Then open: ${BLUE}http://localhost:6006${NC}"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo -e "  • DOCKER_GUIDE.md - Comprehensive guide"
echo -e "  • DOCKER_QUICK_REFERENCE.txt - Quick commands"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
echo ""
