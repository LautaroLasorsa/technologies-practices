# Setup Guide: Solana + Anchor via Docker

Everything runs inside a Docker container. No WSL, no local Rust/Solana installation needed.

---

## Prerequisites

- **Docker Desktop for Windows** -- [Install here](https://docs.docker.com/desktop/install/windows-install/) if not already installed
- Docker Compose (bundled with Docker Desktop)

---

## Step 1: Build and Start the Container

From this practice folder (in PowerShell or cmd):

```batch
dev.bat up
```

First build takes ~10-15 minutes (downloads Rust image, compiles AVM/Anchor, installs Solana CLI). Subsequent starts are instant thanks to Docker layer caching.

Verify the container is running:

```batch
dev.bat versions
```

Expected output (versions may be newer):

```
rustc 1.89.x
solana-cli 2.1.x (Agave)
anchor-cli 0.32.x
v20.x.x
```

---

## Step 2: Open a Shell Inside the Container

```batch
dev.bat shell
```

You're now in a Linux bash shell at `/workspace` (this folder, volume-mounted).

---

## Step 3: Initialize the Anchor Project

Inside the container shell:

```bash
anchor init solana_practice
cd solana_practice
```

This generates:

```
solana_practice/
  Anchor.toml          # Project config (cluster, program IDs, test command)
  Cargo.toml           # Rust workspace
  programs/
    solana_practice/
      Cargo.toml       # Program dependencies (anchor-lang)
      src/
        lib.rs         # Your program code -- this is where you write instructions
  tests/
    solana_practice.ts  # TypeScript test file (default uses @coral-xyz/anchor)
  app/                  # Frontend placeholder (ignore for this practice)
  migrations/
    deploy.ts           # Migration script
```

Or from Windows directly:

```batch
dev.bat init solana_practice
```

---

## Step 4: Build and Test

Inside the container:

```bash
anchor build
# First build is slow (~2 min). Compiles your program to target/deploy/solana_practice.so

anchor test
# Spins up solana-test-validator, deploys, runs tests, shuts down
# You should see "1 passing"
```

Or from Windows:

```batch
dev.bat build
dev.bat test
```

---

## Step 5: Install LiteSVM Test Dependencies

Inside the container:

```bash
cd solana_practice
npm install --save-dev litesvm anchor-litesvm
```

---

## Step 6: Editor Setup

### VS Code (recommended for full IDE support)

1. Install the **Dev Containers** extension
2. Open this folder in VS Code
3. VS Code detects `.devcontainer/devcontainer.json` and prompts "Reopen in Container"
4. Full rust-analyzer with proc-macro expansion works inside the container

### Zed

See `zed-remote.md` for options. The simplest: edit files locally in Zed, run builds via `dev.bat build`.

---

## Step 7: Verify Everything Works

```batch
dev.bat versions
dev.bat build
dev.bat test
```

---

## Workflow Summary

| Task | Command |
|------|---------|
| Start container | `dev.bat up` |
| Stop container | `dev.bat down` |
| Open shell | `dev.bat shell` |
| Build program | `dev.bat build` |
| Run tests | `dev.bat test` |
| Init new project | `dev.bat init <name>` |
| Start validator | `dev.bat validator` |
| Check versions | `dev.bat versions` |

Edit files in your preferred editor on Windows. The folder is volume-mounted into the container at `/workspace`, so changes appear instantly.

---

## Troubleshooting

### First build is very slow

Expected. Compiling AVM from source takes ~5-10 min. Docker caches this layer -- rebuilds only re-run if the Dockerfile changes.

### "anchor build" fails with BPF/SBF errors

Run `solana-install update` inside the container. The Solana cache volume should persist BPF tools, but sometimes needs a refresh.

### "Error: Account not found" in tests

You probably forgot to include an account in the `#[derive(Accounts)]` struct. Anchor requires ALL accounts to be passed explicitly.

### Port 8899 already in use

Another process (or container) is using the Solana RPC port. Stop other containers: `docker compose down` or change the host port in `docker-compose.yml`.

### Cargo cache not persisting

Verify named volumes exist: `docker volume ls | grep cargo`. If they were pruned, the next build re-downloads crates (one-time cost).

### Volume mount performance on Windows

Docker Desktop's WSL2 backend provides reasonable I/O. If builds feel slow, ensure Docker Desktop is using the WSL2 backend (Settings > General > "Use the WSL 2 based engine").

---

## Project Layout After All Phases

By the end of this practice, your `programs/solana_practice/src/lib.rs` will contain:

- **Counter** (Phase 2) -- basic account creation and mutation
- **PDA Counter + User Profile** (Phase 3) -- deterministic addressing
- **Vault** (Phase 5) -- CPI and PDA-signed transfers
- **Events** (Phase 6) -- custom errors and event emission

You can also split into multiple programs under `programs/` if you prefer separation.
