# Practice 066: io_uring — Modern Linux Async I/O

## Technologies

- **io-uring** (crate) — Low-level Rust bindings to Linux io_uring: `IoUring`, `squeue::Entry`, `cqueue::Entry`, opcodes
- **tokio-uring** (crate) — Async runtime built on io_uring with Tokio compatibility (separate runtime, not a Tokio plugin)
- **liburing** — C library providing user-space helpers for io_uring (installed in the container as `liburing-dev`)
- **tokio** — Standard async runtime, used here for comparison benchmarks only

## Stack

- Rust (cargo, edition 2021)
- Docker (Linux container with liburing-dev)

## Theoretical Context

### The Evolution of Linux I/O

Linux I/O has evolved through five generations, each addressing the limitations of its predecessor:

| Generation | API | Year | Model | Limitations |
|-----------|-----|------|-------|-------------|
| 1st | `read()`/`write()` | 1991 | Blocking | One thread per I/O, poor concurrency |
| 2nd | `select()` | 1983 | Readiness (fd_set) | O(n) scan on every call, 1024 fd limit |
| 3rd | `poll()` | 1997 | Readiness (pollfd array) | No fd limit but still O(n) per call |
| 4th | `epoll` | 2002 | Readiness (kernel event set) | O(1) readiness checks, but still 1 syscall per I/O op |
| 5th | `io_uring` | 2019 | Completion (ring buffers) | Batched, zero-copy, kernel does the I/O |

The fundamental shift from epoll to io_uring is the move from a **reactor** model (kernel notifies you that a file descriptor is ready, then you do the I/O) to a **proactor** model (you tell the kernel what I/O to do, the kernel does it and notifies you when it is done). This eliminates an entire class of syscalls: with epoll, a TCP echo server needs `epoll_wait` + `read` + `write` per message; with io_uring, a single `io_uring_enter` can submit both the read and write and collect earlier completions.

Sources: [io_uring(7) man page](https://man7.org/linux/man-pages/man7/io_uring.7.html), [Efficient I/O with io_uring — Jens Axboe](https://kernel.dk/io_uring.pdf)

### Ring Buffer Architecture: SQ and CQ

io_uring is built on two ring buffers shared between user space and kernel space via `mmap`:

```
User Space                        Kernel Space
+-----------------------+         +-----------------------+
| Submission Queue (SQ) |  mmap   | Reads SQEs, executes  |
| [SQE][SQE][SQE]...   | ------> | I/O operations        |
+-----------------------+         +-----------------------+
                                            |
+-----------------------+         +---------v-------------+
| Completion Queue (CQ) |  mmap   | Writes CQEs when done |
| [CQE][CQE][CQE]...   | <------ |                       |
+-----------------------+         +-----------------------+
```

**Submission Queue Entries (SQEs)** describe operations the user wants the kernel to perform. Each SQE contains:
- `opcode` — which operation (read, write, accept, connect, etc.)
- `fd` — file descriptor to operate on
- `addr` / `len` — buffer address and length
- `off` — file offset (for file operations)
- `user_data` — opaque 64-bit value returned in the CQE (for correlating requests to responses)
- `flags` — per-SQE flags (IO_LINK for chaining, FIXED_FILE for pre-registered fds, etc.)

**Completion Queue Entries (CQEs)** report results. Each CQE contains:
- `user_data` — matches the SQE's user_data (your correlation ID)
- `res` — result code (like a syscall return: bytes transferred, or negative errno)
- `flags` — additional flags (IORING_CQE_F_BUFFER for buffer selection, IORING_CQE_F_MORE for multishot)

**The Indirection Array:** The SQ does not directly contain SQEs. Instead, the SQ ring holds *indices* into a separate SQE array. This allows the kernel to consume SQEs in any order (e.g., if one SQE is a fast NOP and another is a slow read, the kernel can complete the NOP first) while the user fills the SQE array sequentially. The CQ ring, in contrast, directly indexes its CQE array.

**Memory Ordering:** The rings use `std::sync::atomic` semantics. The user updates the SQ tail (producer) and reads the CQ head (consumer). The kernel updates the CQ tail (producer) and reads the SQ head (consumer). `Ordering::Release` when writing tails, `Ordering::Acquire` when reading heads. This is lockless communication between user and kernel.

Sources: [Lord of the io_uring — Low-level Interface](https://unixism.net/loti/low_level.html), [Efficient I/O with io_uring](https://kernel.dk/io_uring.pdf)

### The SQE/CQE Lifecycle

A typical io_uring operation follows this lifecycle:

1. **Create the ring:** `io_uring_setup()` allocates the SQ/CQ ring buffers and SQE array, returning a file descriptor. The `io-uring` crate wraps this as `IoUring::new(entries)` where `entries` is the ring size (rounded up to power of 2).

2. **Prepare SQEs:** Build an SQE describing the operation. In the `io-uring` crate, use `opcode::Read::new(fd, buf, len).build().user_data(42)`. The `.build()` call produces an `squeue::Entry`.

3. **Push to SQ:** Add the SQE to the submission ring. `ring.submission().push(&sqe)` writes the entry and advances the SQ tail. This is `unsafe` because the kernel will read the buffer pointer — you must ensure the buffer lives until the CQE arrives.

4. **Submit:** Call `ring.submit()` or `ring.submit_and_wait(n)`. This invokes `io_uring_enter()` which tells the kernel to process the SQ. `submit_and_wait(n)` blocks until at least `n` CQEs are available.

5. **Reap CQEs:** Read completions from the CQ ring via `ring.completion().next()`. Each CQE has `.result()` (the return value) and `.user_data()` (your correlation ID). The user then advances the CQ head.

6. **Handle results:** Check `cqe.result()`. Positive = bytes transferred. Zero = EOF. Negative = `-errno` (e.g., -11 = EAGAIN).

Sources: [io-uring crate docs](https://docs.rs/io-uring/latest/io_uring/), [tokio-rs/io-uring GitHub](https://github.com/tokio-rs/io-uring)

### Proactor vs Reactor Pattern

| Aspect | Reactor (epoll) | Proactor (io_uring) |
|--------|-----------------|---------------------|
| **Notification** | "FD is ready for read" | "Read completed, here is data" |
| **Who does I/O** | User space (after notification) | Kernel (user submits request) |
| **Syscalls per op** | `epoll_wait` + `read`/`write` | One `io_uring_enter` for N ops |
| **Buffer ownership** | User owns buffer entire time | Kernel borrows buffer during op |
| **Batching** | Not built-in (one event at a time) | Native (submit N ops, wait for M) |
| **Zero-copy** | Requires splice/sendfile hacks | Built-in with fixed buffers |
| **Programming model** | Callback on readiness | Submit-and-forget + poll completions |

The proactor model is more natural for high-throughput servers: you describe *what* I/O you want, and the kernel tells you *when it is done*. No "ready but would block" edge cases, no EAGAIN retry loops.

Sources: [Why you should use io_uring for network I/O — Red Hat](https://developers.redhat.com/articles/2023/04/12/why-you-should-use-iouring-network-io), [io_uring vs epoll — Alibaba Cloud](https://www.alibabacloud.com/blog/io-uring-vs--epoll-which-is-better-in-network-programming_599544)

### Fixed Buffers and Registered File Descriptors

Two performance features eliminate repeated kernel-user data marshaling:

**Fixed Buffers (`register_buffers`):** Pre-register a set of buffers with the kernel via `io_uring_register(IORING_REGISTER_BUFFERS, ...)`. Subsequent operations use `ReadFixed`/`WriteFixed` opcodes with a buffer index instead of a pointer. The kernel pins these pages, avoiding repeated `get_user_pages()` calls. This is critical for high-IOPS workloads where page table walks dominate.

**Registered File Descriptors (`register_files`):** Pre-register file descriptors with `IORING_REGISTER_FILES`. Operations then use `types::Fixed(index)` instead of `types::Fd(raw_fd)`. This avoids per-operation `fget()`/`fput()` overhead in the kernel. Registered fds are especially valuable for long-lived connections (TCP servers) where thousands of operations use the same fds.

Sources: [io-uring crate — register module](https://docs.rs/io-uring/latest/io_uring/register/index.html), [Efficient I/O with io_uring](https://kernel.dk/io_uring.pdf)

### io_uring Performance vs epoll

Benchmarks show context-dependent results:

- **TCP echo (high concurrency, 1000+ connections):** io_uring ~10% higher throughput than epoll due to batched submissions amortizing syscall overhead.
- **Database I/O (high IOPS):** io_uring reduces CPU utilization by ~30% and reaches 5M+ IOPS with low queue depth.
- **Tail latency:** io_uring shows consistently lower p99 latency under saturation.
- **Low concurrency (few connections, small batches):** epoll can match or beat io_uring because io_uring's ring buffer overhead is not amortized.

The key insight: **io_uring wins when you can batch**. A single `io_uring_enter` submitting 32 operations is cheaper than 32 individual `read()`/`write()` syscalls. The more operations you can batch, the larger the advantage.

Real-world adoption: RocksDB (Facebook), PostgreSQL (direct I/O), Cloudflare (Pingora HTTP proxy), TigerBeetle (financial database), Bun (JavaScript runtime).

Sources: [io_uring for High-Performance DBMSs — arXiv](https://arxiv.org/html/2512.04859v2), [liburing issue #536 benchmarks](https://github.com/axboe/liburing/issues/536)

### The io-uring Crate (Low-Level)

The `io-uring` crate (version 0.7) provides direct Rust bindings to Linux io_uring. Key types:

```rust
use io_uring::{IoUring, opcode, types, squeue, cqueue};

// Create a ring with 256 entries
let mut ring = IoUring::new(256)?;

// Build a read operation
let read_sqe = opcode::Read::new(types::Fd(fd), buf.as_mut_ptr(), buf.len() as u32)
    .build()
    .user_data(0x01);

// Submit
unsafe { ring.submission().push(&read_sqe)?; }
ring.submit_and_wait(1)?;

// Reap completion
let cqe = ring.completion().next().unwrap();
assert_eq!(cqe.user_data(), 0x01);
let bytes_read = cqe.result(); // positive = bytes, negative = -errno
```

The crate mirrors liburing's design: opcodes are builders, entries are POD-like structs, and `unsafe` is required for `push()` because the kernel will read raw pointers from the SQE.

Source: [io-uring crate docs](https://docs.rs/io-uring/latest/io_uring/)

### The tokio-uring Crate (High-Level Async)

`tokio-uring` wraps io_uring in an async runtime compatible with Tokio. Key differences from standard Tokio:

1. **Separate runtime:** `tokio_uring::start(async { ... })` creates its own event loop, not `#[tokio::main]`.
2. **Buffer ownership:** Operations take buffers by value (not `&mut [u8]`) because the kernel borrows the buffer. Results return `(Result, buf)` tuples so you get the buffer back.
3. **Thread-per-core:** No work-stealing. Tasks stay on the thread that spawned them. `tokio_uring::spawn` accepts `!Send` futures.
4. **Explicit close:** Resources need `.close().await` rather than relying on `Drop` (because closing is an async operation via io_uring).

```rust
tokio_uring::start(async {
    let file = tokio_uring::fs::File::open("data.txt").await.unwrap();
    let buf = vec![0u8; 4096];
    let (result, buf) = file.read_at(buf, 0).await;
    let n = result.unwrap();
    println!("Read {} bytes", n);
});
```

Sources: [tokio-uring docs](https://docs.rs/tokio-uring/), [tokio-uring DESIGN.md](https://github.com/tokio-rs/tokio-uring/blob/master/DESIGN.md), [Announcing tokio-uring — Tokio blog](https://tokio.rs/blog/2021-07-tokio-uring)

### Docker and io_uring: Seccomp Considerations

Docker's default seccomp profile **blocks io_uring syscalls** (`io_uring_setup`, `io_uring_enter`, `io_uring_register`) as a security measure. To use io_uring inside a container, you must either:

1. **Disable seccomp entirely:** `--security-opt seccomp=unconfined` (simplest, used in this practice)
2. **Custom seccomp profile:** Copy Docker's default profile and add the three io_uring syscalls to the whitelist (more secure for production)

The host kernel must support io_uring (Linux 5.1+, recommended 5.10+). On WSL2, the default kernel (5.15+) supports io_uring. Docker Desktop on WSL2 shares the WSL2 kernel, so io_uring works as long as seccomp is disabled.

Sources: [Docker seccomp docs](https://docs.docker.com/engine/security/seccomp/), [moby/moby #47532 — io_uring seccomp](https://github.com/moby/moby/issues/47532)

## Description

Build progressively complex I/O programs using both the low-level `io-uring` crate and the high-level `tokio-uring` runtime, all running inside a Docker container with a Linux kernel:

1. **Ring setup and basic file read** — Create an `IoUring`, build a `Read` SQE, submit, and reap the CQE
2. **File write and multi-operation batching** — Write data, then submit multiple SQEs in a single `io_uring_enter` call
3. **Fixed buffers for zero-copy I/O** — Register buffers with the kernel and use `ReadFixed`/`WriteFixed` opcodes
4. **tokio-uring async file I/O** — Use the high-level async API with buffer ownership semantics
5. **tokio-uring TCP echo server** — Accept connections and echo data using the proactor model
6. **Capstone: benchmark io_uring vs standard I/O** — Compare throughput and latency for file and network workloads

## Instructions

### Phase 0: Docker Environment Setup (~10 min)

The practice runs entirely inside a Docker container because io_uring is a Linux-only kernel interface.

1. Review the `Dockerfile`: Ubuntu base, `liburing-dev` for headers, Rust toolchain via `rustup`
2. Review `docker-compose.yml`: note `security_opt: [seccomp:unconfined]` which allows io_uring syscalls
3. Build and start the container with `docker compose up --build -d`
4. Enter the container with `docker compose exec dev bash`
5. Verify io_uring support: `cat /proc/version` (should show kernel 5.10+)
6. Build the project: `cargo build` inside `/app`

**Why this matters:** Understanding the container setup teaches you about io_uring's system requirements and security implications. The seccomp bypass is a real-world consideration when deploying io_uring-based applications in containers.

### Phase 1: Basic Ring Setup and File Read (~20 min)

**Exercise 1 — `src/ex1_basic_read.rs`**

This exercise teaches the fundamental io_uring lifecycle: create ring, prepare SQE, submit, reap CQE. You will:

- Create an `IoUring` instance with `IoUring::new(8)`
- Open a file and create a `Read` opcode SQE targeting a heap-allocated buffer
- Push the SQE to the submission queue (requires `unsafe` because the kernel reads raw pointers)
- Call `submit_and_wait(1)` to submit and block for one completion
- Read the CQE, check `result()` for bytes read, and print the file contents

This is the "hello world" of io_uring. Every subsequent exercise builds on this pattern: build opcode, push SQE, submit, reap CQE.

### Phase 2: Write and Batched Submissions (~20 min)

**Exercise 2 — `src/ex2_write_and_batch.rs`**

This exercise demonstrates two key io_uring advantages: file writing and batched submission. You will:

- Write data to a file using `opcode::Write`
- Read it back and verify the contents match
- Submit **multiple SQEs in a single `submit()` call** — this is where io_uring shines over epoll. One syscall for N operations.
- Use `user_data` values to correlate CQEs with their SQEs (since completions may arrive out of order)

The batch submission pattern is the core performance primitive. A real-world database might batch 32 WAL writes into one `io_uring_enter` call, eliminating 31 syscalls compared to `write()`.

### Phase 3: Fixed Buffers for Zero-Copy (~15 min)

**Exercise 3 — `src/ex3_fixed_buffers.rs`**

This exercise teaches io_uring's fixed buffer mechanism for zero-copy I/O:

- Pre-allocate a set of buffers and register them with the kernel using `submitter().register_buffers()`
- Use `opcode::WriteFixed` and `opcode::ReadFixed` with buffer indices instead of raw pointers
- The kernel pins these pages, avoiding `get_user_pages()` on every operation
- Compare the code structure with Exercise 1 to see how fixed buffers change the API

Fixed buffers matter for high-IOPS workloads (NVMe SSDs, high-frequency networking) where the page table walk overhead of `get_user_pages()` becomes a bottleneck.

### Phase 4: tokio-uring Async File I/O (~20 min)

**Exercise 4 — `src/ex4_tokio_uring_file.rs`**

This exercise uses `tokio-uring` for high-level async file I/O:

- Start the tokio-uring runtime with `tokio_uring::start()`
- Open a file, write data, and read it back using `tokio_uring::fs::File`
- Observe the **buffer ownership** pattern: `file.write_at(buf, offset).await` returns `(Result, buf)` — you get the buffer back after the kernel is done with it
- This pattern is fundamentally different from Tokio's `&mut [u8]` because io_uring's proactor model needs to own the buffer while the kernel operates on it

Understanding buffer ownership is the key conceptual leap for tokio-uring. Unlike Tokio where you pass a mutable reference (and the runtime notifies readiness), tokio-uring takes ownership because the kernel is actively writing into your buffer.

### Phase 5: tokio-uring TCP Echo Server (~25 min)

**Exercise 5 — `src/ex5_tcp_echo.rs`**

This exercise builds a TCP echo server using tokio-uring's networking API:

- Bind a `TcpListener` and accept connections in a loop
- Spawn a task per connection with `tokio_uring::spawn` (not `tokio::spawn` — tasks are `!Send`)
- Read from the client, echo back, handle EOF and errors
- This is the proactor pattern in action: you submit a read, the kernel completes it, you get data + buffer back

The TCP echo server is the canonical io_uring benchmark. Compare the code structure with an epoll-based echo server: no `poll_read` / `poll_write` states, no EAGAIN handling, no readiness notifications.

### Phase 6: Capstone — Benchmark io_uring vs Standard I/O (~15 min)

**Exercise 6 — `src/ex6_benchmark.rs`**

Compare io_uring against standard library I/O to measure the real-world performance difference:

- **File benchmark:** Write + read a large file (several MB) using both `std::fs` and io_uring with batched operations
- **Measure wall-clock time** for each approach
- Observe the difference (or lack thereof) for your specific workload and explain why

This exercise grounds the theory in data. io_uring's advantage is workload-dependent: it excels with many concurrent operations and large batch sizes, but may not show dramatic improvement for simple sequential file I/O.

## Motivation

- **Modern Linux I/O foundation:** io_uring is rapidly becoming the standard for high-performance Linux I/O. It is adopted by RocksDB, PostgreSQL, Cloudflare Pingora, TigerBeetle, and the Bun JavaScript runtime.
- **Systems programming depth:** Understanding ring buffers, shared memory between user/kernel, memory ordering, and zero-copy patterns deepens your systems programming knowledge beyond what epoll-based runtimes teach.
- **Rust ecosystem relevance:** The `io-uring` and `tokio-uring` crates are the foundation for next-generation Rust async runtimes (monoio, glommio). Understanding them prepares you for the shift from epoll-based tokio to io_uring-based runtimes.
- **Complements Practice 060b (Async Runtime Internals):** That practice covers how Tokio works internally with epoll. This practice covers the next evolution: how io_uring replaces epoll and changes the async programming model from reactor to proactor.

## Commands

All commands run from `practice_066_io_uring/`.

### Docker Environment

| Command | Description |
|---------|-------------|
| `docker compose up --build -d` | Build the Docker image (installs liburing-dev, Rust toolchain) and start the container |
| `docker compose exec dev bash` | Open a shell inside the running container |
| `docker compose down` | Stop and remove the container |
| `docker compose logs -f` | Stream container logs |

### Inside the Container (all cargo commands run from `/app`)

| Command | Description |
|---------|-------------|
| `cargo build` | Compile the project (first build downloads dependencies) |
| `cargo run` | Run all exercises sequentially |
| `cargo run -- 1` | Run only Exercise 1: Basic Ring Setup and File Read |
| `cargo run -- 2` | Run only Exercise 2: Write and Batched Submissions |
| `cargo run -- 3` | Run only Exercise 3: Fixed Buffers for Zero-Copy I/O |
| `cargo run -- 4` | Run only Exercise 4: tokio-uring Async File I/O |
| `cargo run -- 5` | Run only Exercise 5: TCP Echo Server (listens on port 7878) |
| `cargo run -- 6` | Run only Exercise 6: Benchmark io_uring vs Standard I/O |

### Testing the TCP Echo Server (Exercise 5)

| Command | Description |
|---------|-------------|
| `cargo run -- 5` | Start the echo server on port 7878 (inside container) |
| `echo "hello io_uring" \| nc localhost 7878` | Send a test message from another terminal inside the container |

### Quick Start (from Windows host)

| Command | Description |
|---------|-------------|
| `bash run.sh` | One-shot: builds image, starts container, opens shell |

## Notes

