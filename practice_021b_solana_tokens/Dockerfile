# Solana + Anchor development container
# Based on Rust bookworm with Solana CLI, AVM/Anchor, and Node.js
#
# Anchor 0.32.x requires Rust 1.89+ (stabilized Span::local_file for IDL generation)
# Solana CLI stable channel from anza.xyz

FROM rust:1.89-bookworm

# ── System dependencies (Anchor/Solana build requirements) ──────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libudev-dev \
    llvm \
    libclang-dev \
    protobuf-compiler \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Node.js 20 LTS + Yarn (for Anchor TypeScript tests) ────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g yarn

# ── Solana CLI (stable channel from Anza) ───────────────────────────────
RUN sh -c "$(curl -sSfL https://release.anza.xyz/stable/install)" \
    && echo 'export PATH="/root/.local/share/solana/install/active_release/bin:$PATH"' >> /root/.bashrc
ENV PATH="/root/.local/share/solana/install/active_release/bin:${PATH}"

# ── AVM (Anchor Version Manager) + Anchor CLI ──────────────────────────
RUN cargo install --git https://github.com/coral-xyz/anchor avm --locked --force \
    && avm install latest \
    && avm use latest

# ── Solana local dev config (keypair + localhost) ───────────────────────
RUN solana-keygen new --no-bip39-passphrase --force \
    && solana config set --url localhost

# ── Pre-download BPF/SBF platform-tools (~100MB) ────────────────────
# Without this, the first `anchor build` stalls downloading the SDK.
RUN cargo-build-sbf --version || true

# ── Verify installations ───────────────────────────────────────────────
RUN rustc --version && solana --version && anchor --version && node --version

WORKDIR /workspace
