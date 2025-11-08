FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.11 default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# PyTorch CUDA 12.1 wheels
RUN pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Flower 1.23.0 + clean compatible scientific stack
# Note: grpcio-tools was removed because its protobuf requirement conflicts with flwr's range.
# If you need grpcio-tools for building protos in-container, install a compatible version
# or perform proto compilation outside the final runtime image.
RUN pip install \
    flwr==1.23.0 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scikit-learn==1.5.2 \
    matplotlib==3.9.2 \
    tqdm==4.66.5 \
    grpcio==1.65.5 \
    protobuf==4.25.8

# Monitoring
RUN pip install \
    prometheus-client==0.20.0 \
    psutil==5.9.8

# Create dirs
RUN mkdir -p /app/src /app/checkpoints /app/certs /app/logs

# Ports
EXPOSE 8080 8443 9091

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print(torch.cuda.is_available())" || exit 1

CMD ["python3", "/app/src/server.py"]
