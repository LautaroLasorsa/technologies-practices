FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git llvm-15 llvm-15-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install apache-tvm numpy torch --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY . .

CMD ["python", "-m", "tvm_practice.main"]
