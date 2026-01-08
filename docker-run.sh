#!/bin/bash

IMAGE_NAME="directionnet_tf212_cudnn89"
CONTAINER_NAME="e2e_net_container"
PROJECT_DIR="/home/sbuket/projects/Git-Repos/e2e-net-x"

# ---------------------------
# Helper: Print usage
# ---------------------------
usage() {
    echo "Usage: $0 {build|run|shell|stop}"
    exit 1
}

# ---------------------------
# Build image
# ---------------------------
build_image() {
    echo ">>> Building Docker image: $IMAGE_NAME"
    docker build -t $IMAGE_NAME .
}

# ---------------------------
# Run container
# ---------------------------
run_container() {
    echo ">>> Starting container: $CONTAINER_NAME"

    docker run -it --rm \
        --name $CONTAINER_NAME \
        --gpus all \
        -p 6006:6006 \
        -p 8888:8888 \
        -v $PROJECT_DIR:/app \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        $IMAGE_NAME
}

# ---------------------------
# Open an interactive shell
# (without deleting container)
# ---------------------------
shell_container() {
    echo ">>> Starting container shell: $CONTAINER_NAME"

    docker run -it \
        --name $CONTAINER_NAME \
        --gpus all \
        -p 6006:6006 \
        -p 8888:8888 \
        -v $PROJECT_DIR:/app \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        $IMAGE_NAME \
        /bin/bash
}

# ---------------------------
# Stop container
# ---------------------------
stop_container() {
    echo ">>> Stopping: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME
}

# ---------------------------
# Main
# ---------------------------
case "$1" in
    build)
        build_image
        ;;
    run)
        run_container
        ;;
    shell)
        shell_container
        ;;
    stop)
        stop_container
        ;;
          tensorboard)
    echo ">>> Starting TensorBoard on port 6006..."
    docker exec -it e2e_net_container tensorboard --logdir /app/checkpoints  --host 0.0.0.0 --port 6006
    ;;

    *)
        usage
        ;;
esac
