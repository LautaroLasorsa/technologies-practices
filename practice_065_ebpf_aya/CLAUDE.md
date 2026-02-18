# Practice 065 — eBPF Observability: Kernel Tracing with aya

## Technologies

aya, aya-ebpf, aya-build, bpf-linker, eBPF, BPF maps, kprobe, uprobe, tracepoint, XDP

## Stack

Rust, Docker (Ubuntu Linux)

## Theoretical Context

### What is eBPF?

eBPF (extended Berkeley Packet Filter) is a technology that allows running sandboxed programs **inside the Linux kernel** without modifying kernel source code or loading kernel modules. Originally designed for packet filtering (BPF, 1992), it evolved into a general-purpose in-kernel virtual machine (Linux 3.18+, 2014) capable of hooking into almost any kernel event: syscalls, network packets, function calls, tracepoints, and more.

**The problem it solves**: Traditionally, observing kernel behavior required either (a) recompiling the kernel with instrumentation, (b) loading kernel modules (risky, crash-prone), or (c) using slow userspace tools like `strace`. eBPF provides **safe, efficient, dynamic instrumentation** with no kernel recompilation and near-zero overhead.

**Industry adoption**: Datadog, Cloudflare, Netflix, Meta, and Google use eBPF in production. Cilium (Kubernetes networking) is built on eBPF. Tools like `bpftrace`, `bcc`, and Falco use eBPF for observability, security, and networking.

### How eBPF Works Internally

An eBPF program goes through these stages:

1. **Write**: Program written in a restricted C-like language (or Rust with aya). Compiled to eBPF bytecode (a RISC-like instruction set with 11 registers, 512 bytes of stack).

2. **Load**: Userspace calls `bpf()` syscall to load the bytecode into the kernel.

3. **Verify**: The **eBPF verifier** statically analyzes the program to guarantee safety:
   - No infinite loops (programs must terminate — bounded loop counts)
   - No out-of-bounds memory access (all pointer arithmetic is checked)
   - No invalid memory reads (kernel memory must be read via `bpf_probe_read_kernel()`)
   - Stack size limited to 512 bytes (256 with tail calls)
   - Program size limited (1 million instructions as of Linux 5.2)
   - The verifier simulates all execution paths — if any path is unsafe, the program is rejected

4. **JIT-compile**: After verification, the bytecode is JIT-compiled to native machine code for near-native performance.

5. **Attach**: The compiled program is attached to a **hook point** (kprobe, tracepoint, XDP, etc.).

6. **Execute**: Every time the hook fires (e.g., a syscall is called), the eBPF program runs. It can read context, update maps, and emit events to userspace.

### eBPF Program Types (Hook Points)

Each program type defines **where** in the kernel the program attaches and **what context** it receives:

| Program Type | Attach Point | Context | Use Case |
|---|---|---|---|
| **kprobe** | Any kernel function entry | `ProbeContext` (function args) | Trace syscalls, internal kernel functions |
| **kretprobe** | Kernel function return | `ProbeContext` (return value) | Capture return values, measure latency |
| **uprobe** | Userspace function entry | `ProbeContext` (function args) | Trace library/application functions |
| **uretprobe** | Userspace function return | `ProbeContext` (return value) | Measure userspace function latency |
| **tracepoint** | Pre-defined kernel trace points | `TracePointContext` (structured data) | Stable kernel events (syscall enter/exit, sched, net) |
| **raw_tracepoint** | Same as tracepoint, raw args | `RawTracePointContext` | Lower overhead, raw argument access |
| **XDP** | Network driver (before sk_buff) | `XdpContext` (raw packet data) | Packet filtering, DDoS mitigation, load balancing |
| **tc** (traffic control) | Network stack (after sk_buff) | `TcContext` | Packet mangling, policy enforcement |
| **cgroup_skb** | Per-cgroup network | `SkBuffContext` | Container-level network policy |

**kprobe vs tracepoint**: kprobes can attach to ANY kernel function but are unstable across kernel versions (function signatures change). Tracepoints are stable ABI hooks placed at well-known locations (`/sys/kernel/debug/tracing/events/`). Prefer tracepoints when available; use kprobes for functions without tracepoints.

### BPF Maps: Kernel <-> Userspace Communication

eBPF programs cannot directly print output or call userspace. Instead, they communicate through **BPF maps** — shared data structures accessible from both kernel and userspace:

| Map Type | Description | Use Case |
|---|---|---|
| **HashMap** | Key-value store | Count events per PID, track connections |
| **Array** | Fixed-size indexed array | Configuration, per-CPU counters |
| **PerCpuArray** | Per-CPU array (no lock contention) | High-performance counters |
| **PerCpuHashMap** | Per-CPU hash map | Per-CPU event aggregation |
| **PerfEventArray** | Per-CPU ring buffers (perf API) | Stream events to userspace |
| **RingBuf** | Single shared ring buffer (Linux 5.8+) | Preferred for per-event data streaming |
| **LpmTrie** | Longest prefix match trie | IP prefix matching, routing |
| **ProgramArray** | Array of eBPF programs | Tail calls (chaining programs) |

**PerfEventArray vs RingBuf**: PerfEventArray uses per-CPU buffers (no ordering across CPUs, potential data loss under pressure). RingBuf is a single shared buffer with strong ordering and precise notifications. **Prefer RingBuf** for new code on Linux 5.8+.

### aya: Pure-Rust eBPF

[aya](https://github.com/aya-rs/aya) is a Rust-native eBPF framework. Unlike the traditional C-based toolchains (libbpf + clang), aya provides:

- **No C dependencies**: No libbpf, no clang, no kernel headers at runtime. Just Rust + `bpf-linker`.
- **Two crate ecosystem**:
  - `aya-ebpf` — BPF-side library (`#![no_std]`, compiles to `bpfel-unknown-none` target)
  - `aya` — Userspace loader library (loads BPF programs, manages maps, reads events)
- **Shared types** via a `-common` crate that both sides depend on
- **Compile-time inclusion**: BPF bytecode is compiled separately, then included in the userspace binary via `aya::include_bytes_aligned!`
- **CO-RE (Compile Once, Run Everywhere)**: With BTF (BPF Type Format) support, a single binary works across kernel versions without recompilation. BTF provides type information that allows the loader to relocate struct field accesses at load time.

**Project structure** (aya-template pattern):

```
my-project/
  Cargo.toml              # Workspace root
  my-project-ebpf/        # BPF-side crate (#![no_std], target = bpfel-unknown-none)
    Cargo.toml             # depends on aya-ebpf, aya-log-ebpf
    src/main.rs            # BPF programs (#[kprobe], #[xdp], etc.)
  my-project-common/       # Shared types (used by both)
    Cargo.toml
    src/lib.rs
  my-project/              # Userspace loader
    Cargo.toml             # depends on aya, aya-log, tokio, clap
    build.rs               # Uses aya-build to compile the BPF crate
    src/main.rs            # Loads BPF, attaches programs, reads maps
```

**Build flow**:
1. `aya-build` in `build.rs` compiles the `-ebpf` crate with `bpf-linker` to BPF ELF
2. `aya::include_bytes_aligned!` embeds the ELF in the userspace binary
3. At runtime, `aya::Ebpf::load()` loads the ELF, the kernel verifies it, JIT compiles it
4. Userspace code attaches programs and reads/writes maps

### The eBPF Verifier — Why It Matters

The verifier is the **safety guarantee** that makes eBPF unique. Every loaded program is statically verified before execution. Common verifier rejections:

- **Unbounded loops**: Loops must have a provable upper bound (or use `bpf_loop()` on 5.17+)
- **Uninitialized reads**: All stack variables must be initialized before use
- **Pointer arithmetic overflow**: Cannot create pointers past allocation bounds
- **Missing null checks**: Map lookups return `Option`; must check before dereferencing
- **Exceeding stack limit**: 512 bytes total stack space
- **Invalid helper calls**: Can only call kernel helpers allowed for that program type

In aya/Rust, many verifier issues are caught at compile time by the type system, but some (like bounds checking on packet data) still require runtime checks that the verifier validates.

### BTF and CO-RE

**BTF (BPF Type Format)** is a compact metadata format embedded in the kernel (and in compiled BPF programs) that describes data types (structs, unions, enums). It enables:

- **CO-RE**: A BPF program compiled against kernel headers v5.10 can run on v5.15 because BTF tells the loader how struct layouts changed, and the loader patches field offsets at load time.
- **No kernel headers at runtime**: With BTF, you don't need `/usr/src/linux-headers-*` installed on the target machine.
- **Available at**: `/sys/kernel/btf/vmlinux` (Linux 5.2+ with `CONFIG_DEBUG_INFO_BTF=y`)

### Running eBPF in Docker

eBPF programs interact with the **host kernel** — the container shares the host's kernel. This means:

- The Docker container must run with `--privileged` or specific capabilities (`CAP_BPF`, `CAP_PERFMON`, `CAP_SYS_ADMIN`)
- Mount `/sys/kernel/btf/vmlinux` from the host for BTF support
- Mount `/sys/kernel/debug` (debugfs) for tracepoints
- The host kernel must be Linux 5.10+ with BTF enabled (Docker Desktop on Windows uses a Linux VM with a recent kernel, which supports this)
- **Important**: The BPF programs trace the **host (VM) kernel**, not the container. When running on Docker Desktop for Windows, you're tracing the Hyper-V/WSL2 Linux VM kernel.

## Description

This practice builds five eBPF programs of increasing complexity using aya, exploring the core eBPF program types and map mechanisms:

1. **Kprobe syscall tracer** — Attach to `sys_enter_openat` to trace file opens, logging PID and filename
2. **BPF HashMap event counter** — Count syscall invocations per PID using a kernel-side HashMap, read from userspace
3. **Tracepoint process monitor** — Use the stable `sched_process_exec` tracepoint to monitor process executions
4. **XDP packet counter** — Attach to the loopback interface to count packets by protocol (TCP/UDP/ICMP)
5. **Capstone: Process activity dashboard** — Combine kprobes, tracepoints, and maps to build a real-time process monitoring tool

Each exercise teaches a different eBPF concept while reinforcing the aya project structure and BPF map communication patterns.

## Instructions

### Prerequisites

- Docker Desktop for Windows (with WSL2 backend — provides a Linux 5.15+ kernel with BTF)
- Git Bash or any terminal to run shell commands

### Exercise 1: Environment Setup & Hello Kprobe (Docker + first BPF program)

**What you learn**: How to set up an eBPF development environment in Docker, the aya project structure, writing your first kprobe that fires on every `sys_openat` syscall.

**Context**: Before writing any eBPF code, you need a Linux environment with kernel headers and BTF support. Docker Desktop's WSL2 VM provides a modern kernel. Inside the container, you'll install Rust's nightly toolchain (required for `#![no_std]` BPF compilation) and `bpf-linker`. The first kprobe is intentionally minimal — it just logs that the probe fired — to validate the entire build/load/attach pipeline works.

1. Build the Docker development image: `docker compose build`
2. Start the container: `docker compose run --rm ebpf bash`
3. Inside the container, examine the project structure in `/app`
4. Open `ebpf/src/kprobe_open.rs` — implement the `TODO(human)` to create a kprobe that fires on `sys_openat`
5. Open `userspace/src/bin/ex1_kprobe_open.rs` — implement the `TODO(human)` to load and attach the kprobe
6. Build: `cargo build-ebpf && cargo build`
7. Run: `cargo run --bin ex1_kprobe_open`
8. In another terminal in the same container, run `cat /etc/hostname` or `ls /` — you should see log output from the kprobe

### Exercise 2: BPF HashMap — Counting Syscalls per PID

**What you learn**: BPF maps as the kernel<->userspace communication mechanism. How to declare a `HashMap` in the BPF program, write to it from kernel space, and read from userspace. Per-PID aggregation.

**Context**: eBPF programs cannot print directly to stdout — they communicate via maps. A `HashMap<u32, u64>` keyed by PID lets you count how many times each process calls `sys_openat`. The userspace program periodically iterates the map to display counts. This pattern (aggregate in kernel, read in userspace) is fundamental to all eBPF observability tools.

1. Open `ebpf/src/kprobe_counter.rs` — implement the `TODO(human)` to declare a BPF HashMap and increment a counter per PID
2. Open `userspace/src/bin/ex2_kprobe_counter.rs` — implement the `TODO(human)` to read the HashMap and display per-PID counts
3. Build and run, then generate activity in another terminal
4. Observe the per-PID syscall counts updating in real time

### Exercise 3: Tracepoint — Process Execution Monitor

**What you learn**: The difference between kprobes (unstable, any function) and tracepoints (stable ABI). How to read structured data from a `TracePointContext`. Using `read_at()` to extract fields at known offsets.

**Context**: The `sched_process_exec` tracepoint fires whenever a new program is executed (via `execve`). Unlike kprobes, tracepoints have a stable format defined in `/sys/kernel/debug/tracing/events/sched/sched_process_exec/format`. You read fields at fixed byte offsets. This is the same mechanism that tools like `execsnoop` from BCC use.

1. Inside the container, examine the tracepoint format: `cat /sys/kernel/debug/tracing/events/sched/sched_process_exec/format`
2. Open `ebpf/src/tracepoint_exec.rs` — implement the `TODO(human)` to attach to `sched/sched_process_exec` and extract the PID
3. Open `userspace/src/bin/ex3_tracepoint_exec.rs` — implement the `TODO(human)` to receive events via PerfEventArray and print them
4. Open `common/src/lib.rs` — implement the shared `ExecEvent` struct used by both BPF and userspace
5. Build, run, and execute commands in another terminal to see them traced

### Exercise 4: XDP Packet Counter

**What you learn**: XDP (eXpress Data Path) programs that run at the network driver level before the kernel stack. Parsing raw Ethernet/IP headers. Using `PerCpuArray` for lock-free counting.

**Context**: XDP is the fastest packet processing hook in Linux — packets are processed before socket buffer allocation. Your program will parse Ethernet and IPv4 headers to classify packets by protocol (TCP, UDP, ICMP, other) and count them in a `PerCpuArray`. The `ptr_at` pattern for safe pointer arithmetic is essential — the BPF verifier requires explicit bounds checks on all packet data accesses.

1. Open `ebpf/src/xdp_counter.rs` — implement the `TODO(human)` for:
   - The `ptr_at<T>` bounds-checking helper
   - Ethernet header parsing
   - IPv4 protocol extraction
   - Incrementing the correct counter in a PerCpuArray
2. Open `userspace/src/bin/ex4_xdp_counter.rs` — implement the `TODO(human)` to attach to `lo` and periodically read/display the per-protocol counts
3. Build, run, and generate traffic: `ping -c 5 127.0.0.1` and `curl http://127.0.0.1:80` (or any local request)

### Exercise 5: Capstone — Real-Time Process Activity Dashboard

**What you learn**: Combining multiple BPF programs and maps. Using RingBuf for efficient event streaming. Building a complete observability tool.

**Context**: Production eBPF tools (like Datadog's system-probe or Falco) combine multiple program types. This exercise attaches a kprobe to `sys_openat` and a tracepoint to `sched_process_exec`, streams events through a shared `RingBuf`, and displays a unified process activity dashboard. RingBuf (Linux 5.8+) is the modern replacement for PerfEventArray — it uses a single shared buffer with strong ordering guarantees and precise wakeup notifications.

1. Open `common/src/lib.rs` — implement the `TODO(human)` for the `ActivityEvent` enum (OpenFile, ExecProcess) with a discriminant tag
2. Open `ebpf/src/dashboard_open.rs` — implement the `TODO(human)` kprobe that pushes OpenFile events to a RingBuf
3. Open `ebpf/src/dashboard_exec.rs` — implement the `TODO(human)` tracepoint that pushes ExecProcess events to the same RingBuf
4. Open `userspace/src/bin/ex5_dashboard.rs` — implement the `TODO(human)` to load both programs, consume the RingBuf, and display a formatted dashboard
5. Build, run, and observe the unified event stream as you interact with the system

## Motivation

- **Industry standard for production observability**: Datadog, Cilium, Cloudflare, Netflix, and Meta all rely on eBPF for kernel-level observability, networking, and security.
- **Rust + eBPF is the emerging stack**: aya eliminates the C/clang dependency, making eBPF development accessible to Rust developers. The Rust type system catches many verifier issues at compile time.
- **Deep kernel understanding**: Writing eBPF programs requires understanding syscalls, kernel data structures, network stacks — knowledge that transfers to any systems programming role.
- **Complements existing skills**: With CP background (low-level thinking), Rust proficiency, and Docker experience, eBPF is a natural extension into kernel-level systems programming.
- **High-demand niche**: eBPF expertise is rare and highly valued in infrastructure, security, and observability teams.

## Commands

| Phase | Command | Description |
|---|---|---|
| **Setup** | `docker compose build` | Build the Docker development image with Rust, bpf-linker, and kernel headers |
| **Setup** | `docker compose run --rm ebpf bash` | Start an interactive shell in the privileged container |
| **Build** | `cargo build-ebpf` | Compile all eBPF programs (inside container) |
| **Build** | `cargo build` | Compile userspace binaries (inside container) |
| **Build** | `cargo build-ebpf && cargo build` | Full build: eBPF programs + userspace loaders |
| **Run Ex1** | `RUST_LOG=info cargo run --bin ex1_kprobe_open` | Run the kprobe syscall tracer |
| **Run Ex2** | `RUST_LOG=info cargo run --bin ex2_kprobe_counter` | Run the per-PID syscall counter |
| **Run Ex3** | `RUST_LOG=info cargo run --bin ex3_tracepoint_exec` | Run the tracepoint process execution monitor |
| **Run Ex4** | `RUST_LOG=info cargo run --bin ex4_xdp_counter` | Run the XDP packet counter (attaches to `lo`) |
| **Run Ex5** | `RUST_LOG=info cargo run --bin ex5_dashboard` | Run the capstone process activity dashboard |
| **Debug** | `cat /sys/kernel/debug/tracing/events/sched/sched_process_exec/format` | Inspect tracepoint format fields and offsets |
| **Debug** | `bpftool prog list` | List currently loaded BPF programs in the kernel |
| **Debug** | `bpftool map list` | List BPF maps and their types |
| **Debug** | `bpftool map dump id <ID>` | Dump contents of a specific BPF map |
| **Test traffic** | `ping -c 5 127.0.0.1` | Generate ICMP traffic for XDP counter (Exercise 4) |
| **Test traffic** | `curl http://127.0.0.1` | Generate TCP traffic for XDP counter (Exercise 4) |
| **Cleanup** | `docker compose down` | Stop and remove containers |

## Notes

*(To be filled during practice)*
