FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 from deadsnakes PPA (Ubuntu 22.04 ships 3.10)
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python/python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Install PyTorch (CUDA 12.1) and Triton
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir triton

WORKDIR /app
COPY . .

CMD ["python", "-m", "kernels.main"]
