FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# =====================
# Install Python 3.11
# =====================
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set python default ke 3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# =====================
# Install Python packages
# =====================
RUN pip install jupyterlab "numpy<2" tensorflow==2.14.0

# Set TensorFlow env
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=2

# Set workdir
WORKDIR /workspace
EXPOSE 8888


CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--port=8888", "--no-browser", "--notebook-dir=/workspace"]