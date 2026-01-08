FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------
# 1) Base packages
# ------------------------------
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    build-essential \
    git wget curl ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# ------------------------------
# 2) Remove ALL NVIDIA repo definitions to avoid conflicts
# ------------------------------
RUN sed -i '/developer.download.nvidia.com/d' /etc/apt/sources.list || true && \
    rm -f /etc/apt/sources.list.d/*.list || true && \
    rm -f /etc/apt/sources.list.d/*cuda* || true && \
    rm -f /etc/apt/sources.list.d/*nvidia* || true

# ------------------------------
# 3) Add clean NVIDIA repo for cuDNN 8.9
# ------------------------------
RUN mkdir -p /usr/share/keyrings && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    | gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg

RUN echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
    > /etc/apt/sources.list.d/cuda-cudnn.list

# ------------------------------
# 4) Install cuDNN 8.9
# ------------------------------
RUN apt-get update && apt-get install -y \
    libcudnn8=8.9.* \
    libcudnn8-dev=8.9.* && \
    apt-mark hold libcudnn8 libcudnn8-dev
    
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcublas-12-1 \
        libcublas-dev-12-1 && \
    rm -rf /var/lib/apt/lists/*
# ------------------------------
# 5) TensorFlow 2.12 (GPU included)
# ------------------------------
RUN pip install --upgrade pip setuptools wheel
RUN pip install tensorflow==2.12.0

# ------------------------------
# 6) Extra libraries
# ------------------------------
RUN pip install scipy matplotlib pandas scikit-learn pillow tqdm tensorboard jupyterlab
RUN pip install numpy==1.26.4
RUN pip install git+https://github.com/tensorflow/graphics.git
RUN pip install opencv-python==4.8.1.78
RUN pip install tensorflow_probability==0.20.1

CMD ["/bin/bash"]
