# eBPF Development Environment with aya (Rust)
#
# This image provides everything needed to compile and run eBPF programs:
# - Rust stable + nightly toolchains (nightly needed for #![no_std] BPF compilation)
# - bpf-linker (LLVM-based linker for BPF ELF objects)
# - bpftool (inspect loaded BPF programs and maps)
# - Kernel headers and BTF support (via host mount)
#
# IMPORTANT: This container MUST run with --privileged or specific capabilities
# (CAP_BPF, CAP_PERFMON, CAP_SYS_ADMIN) because loading BPF programs into the
# kernel requires elevated privileges. The eBPF programs run in the HOST kernel,
# not in a container-specific kernel.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies:
# - build-essential: gcc, make (needed for some Rust crate builds)
# - pkg-config, libssl-dev: required by many Rust networking crates
# - linux-tools-common, bpftool: inspect and debug loaded BPF programs
# - clang, llvm: LLVM toolchain (bpf-linker depends on LLVM for BPF code generation)
# - libelf-dev, zlib1g-dev: ELF parsing libraries used by BPF tooling
# - iproute2, iputils-ping, curl, net-tools: network utilities for testing XDP programs
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    wget \
    git \
    clang \
    llvm \
    libelf-dev \
    zlib1g-dev \
    linux-tools-common \
    linux-headers-generic \
    iproute2 \
    iputils-ping \
    net-tools \
    bpftool \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup
# We install both stable (for userspace code) and nightly (for BPF #![no_std] code).
# The rust-src component is required for cross-compiling to bpfel-unknown-none target.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Install nightly toolchain with rust-src (required for BPF target compilation)
# The bpfel-unknown-none target is a "tier 3" target that requires -Zbuild-std,
# which in turn requires the rust-src component to rebuild core for the BPF target.
RUN rustup toolchain install nightly --component rust-src

# Install bpf-linker: the LLVM-based linker that produces BPF ELF objects from Rust code.
# --no-default-features avoids pulling in the bundled LLVM (we use the system LLVM).
# This can take a few minutes to compile.
RUN cargo +stable install bpf-linker

# Create working directory
WORKDIR /app

# Copy the entire project into the container
COPY . /app/

# Mount debugfs at container start (needed for tracepoint access)
# This is done in the entrypoint because it requires privileges
RUN echo '#!/bin/bash\nmount -t debugfs debugfs /sys/kernel/debug 2>/dev/null || true\nexec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
