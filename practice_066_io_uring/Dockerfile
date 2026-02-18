FROM ubuntu:24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies:
# - liburing-dev: io_uring headers and static library (userspace helpers)
# - build-essential: gcc, make, etc. for native crate compilation
# - curl: for rustup installer download
# - pkg-config: needed by some Rust crates to find system libraries
# - netcat-openbsd: for testing the TCP echo server (nc command)
RUN apt-get update && apt-get install -y --no-install-recommends \
    liburing-dev \
    build-essential \
    curl \
    pkg-config \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup (stable toolchain)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy Cargo manifest first for dependency caching.
# Docker layer caching means if only source files change (not Cargo.toml),
# the dependency download/compile layer is reused.
COPY Cargo.toml Cargo.lock* ./

# Create a dummy main.rs so cargo can fetch and compile dependencies.
# This is a standard Docker trick for Rust: compile deps once, then
# replace the source with real code.
RUN mkdir -p src && echo 'fn main() { println!("placeholder"); }' > src/main.rs
RUN cargo build 2>/dev/null || true

# Now copy the real source code (invalidates only the final compile layer)
COPY src/ src/

# Build the real project
RUN cargo build

# Create a /tmp/iouring_test directory for file I/O exercises
RUN mkdir -p /tmp/iouring_test

# Default command: keep the container running for interactive use
CMD ["sleep", "infinity"]
