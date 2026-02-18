# Practice 064: WebAssembly from Systems Languages — WASM & WASI

## Technologies

- **wasm-pack** — Build tool that compiles Rust to `.wasm` and generates JS glue code (package.json, .d.ts)
- **wasm-bindgen** — Rust crate for high-level Rust-JavaScript interop (`#[wasm_bindgen]`, `JsValue`)
- **wasmtime** (Rust crate) — Embeddable WebAssembly runtime; `Engine`, `Store`, `Linker`, `Module`, `Instance`
- **wasmtime-wasi** — WASI implementation for wasmtime; `WasiCtxBuilder`, `add_to_linker_sync`
- **WASI** (WebAssembly System Interface) — Capability-based system interface for WASM outside the browser
- **wabt** (optional) — WebAssembly Binary Toolkit: `wasm2wat`, `wat2wasm`, `wasm-objdump`

## Stack

- Rust (cargo workspace, edition 2021)
- Target: `wasm32-unknown-unknown` (browser/pure WASM), `wasm32-wasip1` (WASI)
- Host: native Rust binary embedding wasmtime

## Theoretical Context

### What is WebAssembly?

WebAssembly (WASM) is a **portable binary instruction format** for a stack-based virtual machine. It was originally designed as a compilation target for the web browser — a low-level, fast alternative to JavaScript — but has since expanded to server-side, edge computing, plugin systems, and embedded runtimes.

Key properties:
- **Binary format** (`.wasm`): compact, fast to parse and compile. Browsers can begin executing WASM before the entire file has downloaded (streaming compilation).
- **Text format** (`.wat`, WebAssembly Text): human-readable S-expression syntax. Every `.wasm` binary has an equivalent `.wat` representation. Tools like `wasm2wat` convert between them.
- **Sandboxed execution**: WASM modules execute in a memory-safe sandbox. They cannot access the host's memory, filesystem, or network unless the host explicitly provides those capabilities.
- **Near-native performance**: WASM is designed to be compiled to native machine code with minimal overhead. JIT and AOT compilation are both supported.
- **Language-agnostic**: Any language that can target LLVM (Rust, C, C++, Go, Zig) can compile to WASM.

Source: [WebAssembly Specification](https://webassembly.github.io/spec/core/), [MDN WebAssembly](https://developer.mozilla.org/en-US/docs/WebAssembly)

### The Stack-Based VM

WASM uses a **stack machine** execution model. Instructions pop operands from the stack, perform an operation, and push the result back. For example, adding two numbers:

```wat
i32.const 3    ;; push 3 onto the stack
i32.const 4    ;; push 4 onto the stack
i32.add        ;; pop both, push 7
```

There are no registers in the abstract model. However, WASM also has **local variables** (scoped to a function) and **global variables** (module-wide). In practice, WASM engines compile the stack operations into efficient register-based machine code — the stack is a compilation artifact, not a runtime cost.

WASM has four value types: `i32`, `i64`, `f32`, `f64`. There are no strings, structs, or objects at the WASM level — all complex data is represented as bytes in linear memory.

Source: [WebAssembly Core Specification](https://webassembly.github.io/spec/core/)

### Linear Memory Model

A WASM module has access to a single **linear memory**: a contiguous, byte-addressable, growable array of bytes starting at address 0. Key characteristics:

- **Separate from host memory**: The WASM module's linear memory is completely isolated from the host process memory. A bug in WASM code cannot corrupt the host. This is fundamentally different from native C/C++ where a buffer overflow can corrupt any process memory.
- **Page-based growth**: Memory is allocated in 64 KiB pages (65,536 bytes). A module declares a minimum size and optional maximum. It can grow at runtime via `memory.grow`, but never shrink.
- **Bounds-checked**: Every memory access is bounds-checked (by the engine or via guard pages). Out-of-bounds access traps immediately — no silent corruption, no UB.
- **Shared via offsets**: To pass complex data (strings, arrays, structs) between host and guest, you write bytes into linear memory and pass the offset (a `u32`/`i32` pointer) and length. There are no pointers in the traditional sense — only integer offsets into the flat memory array.

This model means that a WASM module is **memory-safe by construction**: it can only read/write its own linear memory, and all accesses are bounds-checked.

Source: [WebAssembly Memory — O'Reilly](https://www.oreilly.com/library/view/webassembly-the-definitive/9781492089834/ch04.html), [Practical Guide to WASM Memory](https://radu-matei.com/blog/practical-guide-to-wasm-memory/)

### Security Model: No Ambient Authority

WASM's security model is built on the principle of **no ambient authority**:

- A WASM module starts with **zero capabilities**. It cannot access files, network, clocks, environment variables, or any system resource.
- The **host** (browser, runtime, or embedding application) explicitly grants capabilities by providing **import functions**. A module can only call functions that the host has linked.
- There is no `syscall` instruction. There is no way for a WASM module to "escape" the sandbox or access anything the host didn't explicitly provide.

This makes WASM ideal for running **untrusted code** safely: plugins, user-submitted scripts, third-party libraries in a supply-chain-attack-resistant way.

Source: [WebAssembly Security](https://webassembly.org/docs/security/), [WASI Capability-Based Security](http://www.chikuwa.it/blog/2023/capability/)

### WASI: WebAssembly System Interface

WASI is a **standardized set of system interfaces** for WASM modules running outside the browser. It provides capabilities for:

| Interface | What it provides |
|-----------|-----------------|
| `wasi:filesystem` | File open/read/write/stat (directory-scoped) |
| `wasi:clocks` | Wall clock, monotonic clock |
| `wasi:random` | Cryptographic random bytes |
| `wasi:cli` | Stdio (stdin/stdout/stderr), args, env vars |
| `wasi:sockets` | TCP/UDP networking (Preview 2) |

WASI uses **capability-based security**: instead of ambient filesystem access, the host pre-opens specific directories and passes file descriptors. The module can only access files within those pre-opened directories. This is implemented via `openat`-style APIs (relative to a directory FD) rather than `open` (absolute path).

**WASI versions in Rust:**
- `wasm32-wasip1` (formerly `wasm32-wasi`): WASI Preview 1. The stable, widely-supported target. Uses a flat C-like API.
- `wasm32-wasip2`: WASI Preview 2. Uses the Component Model (WIT interfaces). Tier 2 support in Rust since 1.82.

The old target name `wasm32-wasi` was removed in Rust 1.84 (January 2025). Always use `wasm32-wasip1`.

Source: [WASI Introduction](https://wasi.dev/), [Rust Blog — WASI Target Updates](https://blog.rust-lang.org/2024/04/09/updates-to-rusts-wasi-targets/)

### Browser WASM vs Server-Side WASM

| Aspect | Browser (`wasm32-unknown-unknown`) | Server-side (`wasm32-wasip1`) |
|--------|-------------------------------------|-------------------------------|
| Host | Browser JS engine (V8, SpiderMonkey) | Standalone runtime (wasmtime, wasmer, wazero) |
| System access | None (browser sandbox) | Via WASI capabilities |
| Interop | JavaScript (via wasm-bindgen) | Host functions (wasmtime Linker) |
| Build tool | `wasm-pack` (generates JS glue) | `cargo build --target wasm32-wasip1` |
| Use case | Web apps, compute-heavy frontend | Plugins, serverless, edge computing |
| I/O | DOM, fetch, Web APIs | Files, stdio, sockets |

### Embedding Wasmtime: Core Concepts

When you embed wasmtime in a native Rust application (the "host"), you work with these core types:

- **`Engine`**: Global compilation context. Compiles WASM bytecode to native machine code. Shared across threads, created once.
- **`Module`**: A compiled WASM module. Created from `.wasm` bytes or `.wat` text. Immutable after creation, can be instantiated multiple times.
- **`Store<T>`**: Owns all runtime state for WASM instances. The generic parameter `T` holds your host-side state. Not thread-safe — pinned to one thread.
- **`Linker`**: Registry of host-provided functions (imports). You define functions the WASM module can call, then use the linker to instantiate modules.
- **`Instance`**: A running instantiation of a Module. Has its own linear memory, globals, and table. Created by the Linker.
- **`TypedFunc<Params, Results>`**: A strongly-typed handle to an exported WASM function. Call it with `.call(&mut store, params)`.

The embedding pattern:
1. Create an `Engine`
2. Compile a `Module`
3. Create a `Linker`, define host functions
4. Create a `Store<T>` with host state
5. Instantiate the module via the linker
6. Get exported functions and call them

Source: [Wasmtime Embedding API](https://docs.wasmtime.dev/api/wasmtime/), [Wasmtime Examples](https://docs.wasmtime.dev/examples-minimal.html)

### The Component Model and WIT (Future Direction)

The **Component Model** is the next evolution of WebAssembly, adding:

- **WIT (WebAssembly Interface Types)**: An IDL for defining rich interfaces with strings, records, variants, lists, options, results — not just i32/f64.
- **Cross-language composition**: Components written in different languages (Rust, Go, Python) can be linked together via WIT interfaces, each running in its own sandbox.
- **Virtualization**: Components can intercept and redirect imports, enabling dependency injection and mocking at the WASM level.

WIT replaces the manual "write bytes to linear memory and pass offset" pattern with automatic serialization. This practice focuses on the **core WASM model** (Preview 1), but awareness of the Component Model is important for understanding where the ecosystem is heading.

Source: [Component Model Introduction](https://component-model.bytecodealliance.org/), [Fermyon — WASM, WASI, and the Component Model](https://www.fermyon.com/blog/webassembly-wasi-and-the-component-model)

### wasm-bindgen: Rust to JavaScript Interop

`wasm-bindgen` is a Rust procedural macro and CLI tool that generates the JavaScript "glue code" needed to pass rich data types between Rust/WASM and JavaScript:

- **`#[wasm_bindgen]`** on a Rust function exports it to JavaScript. On an `extern "C"` block, it imports JavaScript functions into Rust.
- **`JsValue`**: Catch-all type representing any JavaScript value. All JS types (strings, objects, arrays) arrive as `JsValue` in Rust.
- **Automatic type conversion**: Strings (`&str`, `String`), numbers, booleans, and `Vec<u8>` are converted automatically. Complex objects use `serde_wasm_bindgen` for serialization.
- **`wasm-pack`** wraps `cargo build` + `wasm-bindgen` CLI + npm package generation into a single command.

Build output from `wasm-pack build --target web`:
- `pkg/<name>_bg.wasm` — the WASM binary
- `pkg/<name>.js` — JS glue code (initialization, memory management)
- `pkg/<name>.d.ts` — TypeScript type definitions
- `pkg/package.json` — npm-compatible package manifest

Source: [The wasm-bindgen Guide](https://rustwasm.github.io/docs/wasm-bindgen/), [wasm-pack Documentation](https://rustwasm.github.io/docs/wasm-pack/)

## Description

This practice covers the full spectrum of WebAssembly from a Rust systems perspective:

1. **Compile and inspect**: Build a Rust function to `.wasm`, convert to `.wat`, understand the binary format
2. **WASI programs**: Write Rust programs targeting `wasm32-wasip1`, run them with wasmtime CLI, explore capability-based file access
3. **Linear memory and data passing**: Understand how host and guest share data via linear memory offsets
4. **Browser interop**: Use `wasm-bindgen` and `wasm-pack` to expose Rust functions to JavaScript
5. **Plugin host**: Build a native Rust application that loads `.wasm` plugins via wasmtime, defining a host API
6. **Performance benchmark**: Compare native Rust vs WASM execution for a compute-intensive task

## Instructions

### Prerequisites

Install the required targets and tools:

```bash
# Add WASM compilation targets
rustup target add wasm32-unknown-unknown
rustup target add wasm32-wasip1

# Install wasmtime CLI (for running WASI programs)
cargo install wasmtime-cli

# Install wasm-pack (for browser WASM builds)
cargo install wasm-pack

# Optional: install wabt for inspecting WASM binaries
# On Windows: download from https://github.com/WebAssembly/wabt/releases
# Or via cargo: cargo install wasm-tools
```

### Workspace Structure

The practice uses a Cargo workspace with three sub-projects:

- **`wasi-hello/`** — A program targeting `wasm32-wasip1` (exercises 1-2)
- **`wasm-lib/`** — A library targeting `wasm32-unknown-unknown` with wasm-bindgen (exercises 3-4)
- **`host/`** — A native Rust binary that embeds wasmtime to load and run WASM modules (exercises 5-6)

---

### Exercise 1: Compile Rust to WASM and Inspect the Binary

**What you learn**: How Rust code becomes WASM bytecode, the structure of a `.wasm` module (sections, function signatures, linear memory), and how to read WAT text format.

**Why it matters**: Understanding the binary format is essential for debugging WASM issues. When something goes wrong at the WASM level, you need to inspect exports, imports, and memory layout.

Open `wasi-hello/src/main.rs`. Implement the `fibonacci` function and the `main` function that calls it. Build for the `wasm32-wasip1` target, then use `wasm-tools` or `wasm2wat` to inspect the output.

Key observations:
- The `.wat` output shows function signatures with WASM types (`i32`, `i64`)
- Rust strings become byte sequences in the data section
- `println!` pulls in WASI imports (`fd_write`)
- The module declares a linear memory section

---

### Exercise 2: WASI Capabilities and File Access

**What you learn**: How WASI's capability-based security works in practice. A WASM module cannot access the filesystem unless the host pre-opens directories for it.

**Why it matters**: This is the core security model of server-side WASM. Understanding directory pre-opening and capability scoping is essential for running untrusted code safely.

In `wasi-hello/src/main.rs`, implement the file-writing function that creates a file in a WASI-accessible directory. Run with wasmtime and observe how `--dir` grants access. Try accessing a path outside the granted directory and observe the error.

---

### Exercise 3: Linear Memory and Host-Guest Data Passing

**What you learn**: How complex data (strings, byte arrays) crosses the WASM boundary via linear memory offsets. The host writes bytes into the guest's memory at an offset, then passes that offset (as an `i32`) to a guest function.

**Why it matters**: This is the fundamental mechanism for all WASM interop before the Component Model. Every WASM plugin system, every WASM-based microservice, and every browser WASM app uses this pattern to pass anything more complex than a number.

In `wasm-lib/src/lib.rs`, implement functions that:
- Allocate memory in the WASM module (expose an allocator function)
- Accept a pointer+length pair to process a string from the host
- Return results by writing to a shared memory region

---

### Exercise 4: wasm-bindgen for Browser Interop

**What you learn**: How `#[wasm_bindgen]` automatically handles the linear memory dance for strings, structs, and return values. How `wasm-pack` generates the JS glue code, `.d.ts` types, and npm package.

**Why it matters**: This is the standard way to use Rust in web applications. wasm-bindgen abstracts away the raw memory management from Exercise 3, but understanding Exercise 3 first reveals what wasm-bindgen does under the hood.

In `wasm-lib/src/lib.rs`, implement functions annotated with `#[wasm_bindgen]` that:
- Accept and return strings
- Process a JSON payload
- Expose a Rust struct to JavaScript

Build with `wasm-pack build --target web` and examine the generated `pkg/` output.

---

### Exercise 5: WASM Plugin Host with wasmtime

**What you learn**: How to build a native application that loads and runs `.wasm` modules dynamically. Define a host API (imports), compile and instantiate modules, call exported functions, and share data via linear memory.

**Why it matters**: This is the plugin architecture pattern used by Envoy, Figma, Zed, Shopify, and many other systems. The host defines a stable API; plugins are compiled to `.wasm` and loaded at runtime without trusting the plugin code.

In `host/src/main.rs`, implement:
- Engine and Store creation
- Linker setup with host-provided functions (logging, data access)
- Module loading from a `.wasm` file
- Calling exported guest functions
- Reading results from the guest's linear memory

---

### Exercise 6: Performance — Native vs WASM

**What you learn**: The realistic performance overhead of WASM compared to native execution. How wasmtime's AOT compilation, Cranelift optimizations, and the memory sandbox affect throughput.

**Why it matters**: Knowing the performance characteristics helps you decide when WASM is appropriate (plugins, untrusted code, portability) vs when native execution is required (tight inner loops, latency-critical paths).

In `host/src/main.rs`, implement a benchmark that:
- Runs a compute-intensive function (matrix multiply, or fibonacci) natively
- Runs the same function via a WASM module loaded in wasmtime
- Compares wall-clock times and reports the overhead ratio

---

## Motivation

**Why learn WebAssembly from the systems side?**

1. **Plugin systems**: The industry standard for safe plugin execution is converging on WASM. Envoy proxies (service mesh filters), Figma (design tool plugins), Zed (editor extensions), Shopify (serverless functions), and Fastly/Cloudflare (edge computing) all use WASM. Building a WASM plugin host is a high-demand skill.

2. **Portable sandboxed execution**: WASM runs identically on any platform with zero setup. Ship a `.wasm` binary that works on Linux, macOS, Windows, ARM, x86 — no recompilation, no containers, no VMs.

3. **Supply chain security**: Running third-party code in a WASM sandbox limits blast radius. A compromised dependency can only access what you explicitly grant. This is increasingly critical as supply-chain attacks grow.

4. **Edge computing**: Cloudflare Workers, Fastly Compute, Fermyon Spin — all run WASM at the edge. Understanding WASM internals is essential for building and optimizing edge functions.

5. **Cross-language interop**: The Component Model enables linking Rust, Go, Python, and C++ components together via WIT interfaces — a universal FFI that's memory-safe and language-agnostic.

6. **Complementary to current skills**: As a Rust programmer with systems experience, you're in the ideal position to build WASM hosts (wasmtime embedding), compile high-performance libraries to WASM, and design plugin APIs.

## Commands

### Setup

| Command | Description |
|---------|-------------|
| `rustup target add wasm32-unknown-unknown` | Add the browser/pure WASM compilation target |
| `rustup target add wasm32-wasip1` | Add the WASI Preview 1 compilation target |
| `cargo install wasmtime-cli` | Install the wasmtime standalone WASM runtime CLI |
| `cargo install wasm-pack` | Install the wasm-pack build tool for browser WASM |
| `cargo install wasm-tools` | Install wasm-tools for inspecting `.wasm` binaries (alternative to wabt) |

### Building

| Command | Description |
|---------|-------------|
| `cargo build -p wasi-hello --target wasm32-wasip1` | Compile the WASI program to `.wasm` |
| `cargo build -p wasm-lib --target wasm32-unknown-unknown` | Compile the WASM library (no WASI, no JS glue) |
| `wasm-pack build wasm-lib --target web` | Build the WASM library with JS glue for browser use |
| `cargo build -p wasm-host` | Build the native host binary that embeds wasmtime |
| `cargo check` | Check the entire workspace (native target only; sub-projects have WASM-specific targets) |

### Running

| Command | Description |
|---------|-------------|
| `wasmtime target/wasm32-wasip1/debug/wasi-hello.wasm` | Run the WASI program with default capabilities |
| `wasmtime --dir=. target/wasm32-wasip1/debug/wasi-hello.wasm` | Run with filesystem access to current directory |
| `wasmtime --dir=./testdata target/wasm32-wasip1/debug/wasi-hello.wasm` | Run with filesystem access scoped to `testdata/` |
| `cargo run -p wasm-host` | Run the native host with embedded wasmtime |
| `cargo run -p wasm-host -- --bench` | Run the native host in benchmark mode (Exercise 6) |

### Inspecting WASM Binaries

| Command | Description |
|---------|-------------|
| `wasm-tools print target/wasm32-wasip1/debug/wasi-hello.wasm` | Disassemble `.wasm` to WAT text format |
| `wasm-tools print target/wasm32-wasip1/debug/wasi-hello.wasm \| head -50` | View first 50 lines of WAT output |
| `wasm-tools validate target/wasm32-wasip1/debug/wasi-hello.wasm` | Validate a `.wasm` binary against the spec |
| `wasm-tools objdump target/wasm32-wasip1/debug/wasi-hello.wasm` | Show WASM module sections (types, functions, memory, data) |

## Notes

*(To be filled during practice.)*
