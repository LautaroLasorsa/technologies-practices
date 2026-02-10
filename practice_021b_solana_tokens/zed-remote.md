# Using Zed Editor with the Solana Dev Container

Three options for editing code in Zed while building inside the Docker container.

---

## Option 1: Local Edit + Container Build (Recommended)

The simplest approach. Files live on your Windows filesystem (volume-mounted into the container). You edit in Zed locally; builds run in Docker.

1. Start the container: `dev.bat up`
2. Open this folder in Zed normally (Windows path)
3. Edit `.rs`, `.ts`, `.toml` files in Zed -- they're mounted at `/workspace` inside the container
4. Build/test from a terminal: `dev.bat build` / `dev.bat test`
5. Or open a shell: `dev.bat shell` and run commands directly

**Limitation**: rust-analyzer in Zed won't resolve Solana/Anchor types locally (the toolchain is inside Docker). You get syntax highlighting and basic Rust support, but no proc-macro expansion or Anchor-specific completions.

**Workaround**: Install rust-analyzer locally and point it at the container's sysroot via a `rust-toolchain.toml`, or use Option 2.

---

## Option 2: Zed Remote Development (SSH into Container)

Zed supports remote development over SSH. You can SSH into the running container for full LSP support.

1. Start the container: `dev.bat up`
2. Install an SSH server inside the container (add to Dockerfile if permanent):
   ```bash
   docker compose exec dev bash -c "apt-get update && apt-get install -y openssh-server && service ssh start"
   ```
3. Configure SSH port forwarding in `docker-compose.yml` (add `- "2222:22"` to ports)
4. Connect Zed via `zed ssh://root@localhost:2222/workspace`

**Benefit**: Full rust-analyzer support with proc-macros, Anchor completions, and jump-to-definition.

**Downside**: More setup, SSH server in dev container.

---

## Option 3: VS Code Dev Containers (Alternative Editor)

If you need full IDE integration, VS Code with the Dev Containers extension provides the smoothest experience:

1. Install the "Dev Containers" extension in VS Code
2. Open this folder in VS Code
3. VS Code detects `.devcontainer/devcontainer.json` and prompts to reopen in container
4. Full rust-analyzer, proc-macro expansion, and Anchor support inside the container

---

## Recommendation

**Start with Option 1** for this practice. You get fast editing in Zed, and `dev.bat build`/`dev.bat test` handles compilation. If you find yourself needing Anchor-aware completions, switch to Option 2 or 3.
