//! WASM Host — Exercises 5 & 6
//!
//! This is a native Rust binary that embeds the wasmtime WebAssembly runtime.
//! It loads .wasm modules compiled from the other workspace members and executes
//! them in a sandboxed environment.
//!
//! Build: cargo build -p wasm-host
//! Run:   cargo run -p wasm-host
//! Bench: cargo run -p wasm-host -- --bench
//!
//! Before running, compile the guest modules:
//!   cargo build -p wasi-hello --target wasm32-wasip1
//!   cargo build -p wasm-lib --target wasm32-unknown-unknown

use anyhow::{Context, Result};
use std::env;
use std::time::Instant;
use wasmtime::*;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let bench_mode = args.iter().any(|a| a == "--bench");

    println!("=== Exercise 5: WASM Plugin Host with wasmtime ===\n");
    exercise_5_plugin_host()?;

    if bench_mode {
        println!("\n=== Exercise 6: Performance — Native vs WASM ===\n");
        exercise_6_benchmark()?;
    } else {
        println!("\nSkipping Exercise 6 (benchmark). Run with --bench to enable.");
    }

    Ok(())
}

// =============================================================================
// Exercise 5: WASM Plugin Host with wasmtime
// =============================================================================

/// Build a native application that loads and runs WASM modules as plugins.
///
/// This is the exercise where you build the "host" side of a WASM plugin system.
/// The host:
///   1. Creates a wasmtime Engine (compiles WASM to native code)
///   2. Creates a Store (holds runtime state)
///   3. Sets up a Linker with host-provided functions (the plugin API)
///   4. Loads a .wasm module from disk
///   5. Instantiates the module (resolves imports)
///   6. Calls exported guest functions
///   7. Reads results from the guest's linear memory
///
/// This is the exact pattern used by Envoy (proxy plugins), Figma (design plugins),
/// Zed (editor extensions), Shopify (serverless functions), and many other systems.
fn exercise_5_plugin_host() -> Result<()> {
    // We'll use an inline WAT module for Exercise 5 so it works without building
    // the other crates first. This makes the exercise self-contained.
    //
    // The WAT module below defines:
    //   - A linear memory (1 page = 64 KiB)
    //   - An imported host function: "env" "log_message" (ptr: i32, len: i32)
    //   - An exported function: "compute" (a: i32, b: i32) -> i32
    //   - An exported function: "write_greeting" () -> i64 (packed ptr|len)
    //   - An exported memory: "memory"

    // --- Part A: Engine, Store, and Module setup ---
    println!("--- 5a: Engine, Store, and Module setup ---\n");
    run_simple_module()?;

    // --- Part B: Host functions (imports) ---
    println!("\n--- 5b: Host functions and imports ---\n");
    run_with_host_functions()?;

    // --- Part C: Linear memory data exchange ---
    println!("\n--- 5c: Linear memory data exchange ---\n");
    run_memory_exchange()?;

    Ok(())
}

/// Exercise 5a: Create Engine, Store, compile and run a minimal WASM module.
///
/// # What to implement
///
/// 1. Create a wasmtime `Engine` with default settings
/// 2. Create a `Store` with a unit `()` host state
/// 3. Compile a WAT module that exports a pure function `add(i32, i32) -> i32`
/// 4. Instantiate the module (no imports needed for a pure function)
/// 5. Get a typed function reference to the exported `add`
/// 6. Call it and print the result
///
/// # Why this matters
///
/// This is the minimal viable WASM embedding. Understanding Engine/Store/Module/Instance
/// is prerequisite to everything else. Key concepts:
///
/// - **Engine**: compiles WASM bytecode to native machine code using Cranelift.
///   Created once, shared for the lifetime of the application. Thread-safe.
///
/// - **Store<T>**: owns all WASM runtime state (memories, tables, globals).
///   The `T` parameter holds host-side state that host functions can access.
///   NOT thread-safe — pinned to a single thread.
///
/// - **Module**: a compiled WASM module. Immutable after creation. Can be
///   instantiated multiple times (each instance gets its own memory/globals).
///
/// - **Instance**: a running module. Owns its linear memory, globals, and tables.
///   Created by instantiating a Module with resolved imports.
///
/// - **TypedFunc<Params, Results>**: a strongly-typed handle to an exported function.
///   `.call(&mut store, params)` executes the function and returns the result.
///
/// # Hints
///
/// ```text
/// let engine = Engine::default();
/// let module = Module::new(&engine, wat_source)?;
/// let mut store = Store::new(&engine, ());
/// let instance = Instance::new(&mut store, &module, &[])?;
/// let add = instance.get_typed_func::<(i32, i32), i32>(&mut store, "add")?;
/// let result = add.call(&mut store, (2, 3))?;
/// ```
fn run_simple_module() -> Result<()> {
    // This WAT module defines a single pure function: add(a, b) -> a + b
    // No imports, no memory, no side effects — the simplest possible WASM module.
    //
    // WAT syntax explained:
    //   (module ...)        — top-level WASM module
    //   (func ...)          — function definition
    //   (export "add" ...)  — makes the function visible to the host
    //   (param $a i32)      — function parameter (WASM only has i32/i64/f32/f64)
    //   (result i32)        — return type
    //   local.get $a        — push parameter 'a' onto the stack
    //   i32.add             — pop two i32s, push their sum
    let wat = r#"
        (module
            (func (export "add") (param $a i32) (param $b i32) (result i32)
                local.get $a
                local.get $b
                i32.add
            )
        )
    "#;

    // TODO(human): Create Engine, compile Module, create Store, instantiate, call "add".
    //
    // Steps:
    // 1. let engine = Engine::default();
    //    → Creates the global compilation context. Cranelift (wasmtime's code generator)
    //      will compile the WAT/WASM to native x86/ARM machine code.
    //
    // 2. let module = Module::new(&engine, wat)?;
    //    → Parses the WAT text format, validates it against the WASM spec,
    //      and compiles it to native code. This is the expensive step.
    //      In production, you'd cache compiled modules or use Module::deserialize.
    //
    // 3. let mut store = Store::new(&engine, ());
    //    → Creates the runtime state container. The () means no host state.
    //      The store owns all WASM objects (memories, tables, globals) created
    //      during instantiation.
    //
    // 4. let instance = Instance::new(&mut store, &module, &[])?;
    //    → Instantiates the module. The &[] means "no imports" — this module
    //      is self-contained. If the module had imports, you'd provide them
    //      here (or use a Linker instead, which is more ergonomic).
    //
    // 5. let add = instance.get_typed_func::<(i32, i32), i32>(&mut store, "add")?;
    //    → Gets a strongly-typed handle to the exported "add" function.
    //      The type parameters <(i32, i32), i32> must match the WASM function
    //      signature exactly — wasmtime verifies this at runtime and returns
    //      an error if the types don't match.
    //
    // 6. let result = add.call(&mut store, (3, 4))?;
    //    → Calls the WASM function. Execution transitions from native code
    //      into the compiled WASM code, runs the function, and returns.
    //      The overhead per call is very small (~10-50ns) thanks to Cranelift's
    //      optimized trampolines.
    //
    // 7. println!("  add(3, 4) = {}", result);
    //    assert_eq!(result, 7);
    //
    // After implementing, also try:
    //   - Calling with different arguments
    //   - Deliberately using wrong types: get_typed_func::<(i32,), i32> → should error
    //   - Calling a non-existent export name → should error
    todo!("Exercise 5a: Create Engine/Store/Module/Instance, call exported function")
}

/// Exercise 5b: Define host functions that WASM modules can import.
///
/// # What to implement
///
/// 1. Create a `Linker` and register a host function `"env" "log_message"`
///    that the WASM module can call to log messages
/// 2. Register a host function `"env" "get_random"` that returns a pseudo-random i32
/// 3. Compile a WAT module that imports and calls these functions
/// 4. Instantiate via the linker and call the guest's exported function
///
/// # Why this matters
///
/// Host functions are the ONLY way a WASM module can interact with the outside world.
/// The module starts in a complete sandbox — no I/O, no system calls, no network.
/// Every capability must be explicitly provided by the host via imported functions.
///
/// This is the plugin API pattern:
/// - The host defines a set of functions (the API) that plugins can call
/// - Plugins are compiled to .wasm and import these functions by name
/// - The host controls exactly what each plugin can do
///
/// Examples in production:
/// - Envoy: plugins import `proxy_log()`, `proxy_get_header()`, `proxy_send_response()`
/// - Figma: plugins import `figma.currentPage.selection`, `figma.createRectangle()`
/// - Game engines: plugins import `spawn_entity()`, `play_sound()`, `get_input()`
///
/// # Key concept: Caller<T>
///
/// Host functions receive a `Caller<'_, T>` parameter that provides:
/// - `caller.data()` / `caller.data_mut()` — access to the Store's host state (T)
/// - `caller.get_export("memory")` — access to the guest's exports (especially memory)
///
/// This lets host functions read/write the guest's linear memory (to exchange strings,
/// byte arrays, etc.) and access shared host state (database connections, configuration).
///
/// # Hints
///
/// ```text
/// let mut linker = Linker::new(&engine);
///
/// linker.func_wrap("env", "log_value", |caller: Caller<'_, HostState>, value: i32| {
///     println!("  [host] Guest logged: {}", value);
/// })?;
///
/// let instance = linker.instantiate(&mut store, &module)?;
/// ```
fn run_with_host_functions() -> Result<()> {
    // Host state: the host can store arbitrary data that host functions can access.
    // This is how you give plugins access to application-specific resources
    // (database connections, caches, configuration) without exposing raw system APIs.
    struct HostState {
        log_buffer: Vec<String>,
        call_count: u32,
    }

    // WAT module that imports two host functions and calls them.
    //
    // Import declaration syntax:
    //   (import "module" "name" (func $local_name (param types) (result type)))
    //
    // The "module" and "name" strings must match what the Linker registers.
    // "env" is a common convention for the default import namespace.
    let wat = r#"
        (module
            ;; Import host functions
            (import "env" "log_value" (func $log_value (param i32)))
            (import "env" "get_random" (func $get_random (result i32)))

            ;; Export a function that uses host imports
            (func (export "run_plugin") (result i32)
                ;; Log the number 42
                i32.const 42
                call $log_value

                ;; Get a random number and log it
                call $get_random
                call $log_value

                ;; Get another random number and return it
                call $get_random
            )
        )
    "#;

    // TODO(human): Set up Linker with host functions, instantiate, and call "run_plugin".
    //
    // Steps:
    //
    // 1. let engine = Engine::default();
    //
    // 2. let module = Module::new(&engine, wat)?;
    //
    // 3. let mut linker: Linker<HostState> = Linker::new(&engine);
    //    → The Linker is parameterized by the host state type. All host functions
    //      registered on this linker will receive Caller<'_, HostState>.
    //
    // 4. Register "env" "log_value":
    //    linker.func_wrap("env", "log_value", |mut caller: Caller<'_, HostState>, value: i32| {
    //        println!("  [host] Guest logged: {}", value);
    //        caller.data_mut().log_buffer.push(format!("logged: {}", value));
    //        caller.data_mut().call_count += 1;
    //    })?;
    //
    //    → When the guest calls `(call $log_value)`, execution jumps to this closure.
    //      The Caller gives access to HostState, so we can record the log.
    //
    // 5. Register "env" "get_random":
    //    linker.func_wrap("env", "get_random", |mut caller: Caller<'_, HostState>| -> i32 {
    //        caller.data_mut().call_count += 1;
    //        // Deterministic "random" for reproducibility
    //        let count = caller.data().call_count;
    //        ((count * 1103515245 + 12345) % 100) as i32
    //    })?;
    //
    //    → The guest thinks it's getting random numbers, but the host controls
    //      exactly what values are returned. This is the power of the sandbox:
    //      the host can mock, intercept, or modify any capability.
    //
    // 6. let host_state = HostState { log_buffer: Vec::new(), call_count: 0 };
    //    let mut store = Store::new(&engine, host_state);
    //
    // 7. let instance = linker.instantiate(&mut store, &module)?;
    //    → The linker resolves the module's imports against registered functions.
    //      If any import is missing, this returns an error (the module can't run
    //      without all its imports satisfied).
    //
    // 8. let run = instance.get_typed_func::<(), i32>(&mut store, "run_plugin")?;
    //    let result = run.call(&mut store, ())?;
    //
    // 9. Print the result and the host state:
    //    println!("  Plugin returned: {}", result);
    //    println!("  Host call count: {}", store.data().call_count);
    //    println!("  Log buffer: {:?}", store.data().log_buffer);
    todo!("Exercise 5b: Set up Linker with host functions, call guest plugin")
}

/// Exercise 5c: Exchange string data between host and guest via linear memory.
///
/// # What to implement
///
/// 1. Load a WASM module that exports `allocate`, `deallocate`, and `transform_string`
///    (from the wasm-lib crate, or use the inline WAT below)
/// 2. Write a string into the guest's linear memory
/// 3. Call the guest's transform function
/// 4. Read the result string from the guest's linear memory
///
/// # Why this matters
///
/// This is the most important exercise for understanding WASM data exchange.
/// WASM functions can only accept and return scalar types (i32, i64, f32, f64).
/// To pass strings, arrays, structs, or any complex data:
///
///   Host → Guest:
///     1. Host calls guest's allocate(size) → gets an offset
///     2. Host writes bytes at that offset in the guest's memory
///     3. Host calls a guest function, passing (offset, length)
///
///   Guest → Host:
///     1. Guest allocates and fills a buffer in its own memory
///     2. Guest returns (offset, length) to the host (packed in i64 or via multi-return)
///     3. Host reads the bytes from the guest's memory at that offset
///
/// The key insight: **the guest's linear memory is visible to the host** via
/// `instance.get_memory()`. The host can read/write any byte in the guest's
/// memory. But the guest CANNOT access the host's memory — isolation is one-way.
///
/// # Memory safety
///
/// - The host reads/writes via `memory.read()` and `memory.write()` — these
///   are bounds-checked and return errors instead of corrupting memory.
/// - The guest's allocator manages which regions of linear memory are in use.
///   The host must only write to allocated regions (via the guest's allocate function).
///
/// # Hints
///
/// ```text
/// // Get the guest's exported memory
/// let memory = instance.get_memory(&mut store, "memory")
///     .context("module must export memory")?;
///
/// // Write a string into guest memory
/// let input = "hello wasm";
/// let input_bytes = input.as_bytes();
/// let ptr = allocate_fn.call(&mut store, input_bytes.len() as i32)?;
/// memory.write(&mut store, ptr as usize, input_bytes)?;
///
/// // Call the transform function
/// let packed_result = transform_fn.call(&mut store, (ptr, input_bytes.len() as i32))?;
/// let result_ptr = (packed_result >> 32) as i32;
/// let result_len = (packed_result & 0xFFFF_FFFF) as i32;
///
/// // Read the result from guest memory
/// let mut result_buf = vec![0u8; result_len as usize];
/// memory.read(&store, result_ptr as usize, &mut result_buf)?;
/// let result_str = String::from_utf8(result_buf)?;
/// ```
fn run_memory_exchange() -> Result<()> {
    // For this exercise, we use an inline WAT module that demonstrates the
    // memory exchange protocol. This is a simplified version of what wasm-lib
    // does — it uppercases ASCII characters (only handles a-z for simplicity).
    //
    // In a real plugin system, you'd load the compiled wasm-lib.wasm file:
    //   let module = Module::from_file(&engine, "target/wasm32-unknown-unknown/debug/wasm_lib.wasm")?;
    //
    // We use inline WAT here so the exercise works without cross-compilation.
    let wat = r#"
        (module
            ;; Linear memory: 1 page (64 KiB). The host can read/write this.
            (memory (export "memory") 1)

            ;; Simple bump allocator: global tracks next free offset.
            ;; Starts at offset 1024 to leave space for stack and static data.
            (global $bump_offset (mut i32) (i32.const 1024))

            ;; allocate(size: i32) -> offset: i32
            ;; Bump-allocate `size` bytes and return the start offset.
            (func (export "allocate") (param $size i32) (result i32)
                (local $offset i32)
                global.get $bump_offset
                local.set $offset
                global.get $bump_offset
                local.get $size
                i32.add
                global.set $bump_offset
                local.get $offset
            )

            ;; to_upper(ptr: i32, len: i32) -> packed_result: i64
            ;; Read `len` bytes at `ptr`, uppercase ASCII a-z, write result
            ;; to new allocation, return packed (result_ptr << 32 | result_len).
            (func (export "to_upper") (param $ptr i32) (param $len i32) (result i64)
                (local $i i32)
                (local $result_ptr i32)
                (local $byte i32)

                ;; Allocate space for result (same size as input)
                global.get $bump_offset
                local.set $result_ptr
                global.get $bump_offset
                local.get $len
                i32.add
                global.set $bump_offset

                ;; Loop: copy each byte, uppercasing a-z
                (block $break
                    (loop $continue
                        ;; if i >= len, break
                        local.get $i
                        local.get $len
                        i32.ge_u
                        br_if $break

                        ;; Load byte from input
                        local.get $ptr
                        local.get $i
                        i32.add
                        i32.load8_u

                        ;; If byte >= 'a' (97) and byte <= 'z' (122), subtract 32
                        local.set $byte
                        local.get $byte
                        i32.const 97
                        i32.ge_u
                        (if
                            (then
                                local.get $byte
                                i32.const 122
                                i32.le_u
                                (if
                                    (then
                                        local.get $byte
                                        i32.const 32
                                        i32.sub
                                        local.set $byte
                                    )
                                )
                            )
                        )

                        ;; Store byte to result
                        local.get $result_ptr
                        local.get $i
                        i32.add
                        local.get $byte
                        i32.store8

                        ;; i++
                        local.get $i
                        i32.const 1
                        i32.add
                        local.set $i

                        br $continue
                    )
                )

                ;; Return packed (result_ptr << 32) | len
                local.get $result_ptr
                i64.extend_i32_u
                i64.const 32
                i64.shl
                local.get $len
                i64.extend_i32_u
                i64.or
            )
        )
    "#;

    // TODO(human): Load the module, write a string to guest memory, call to_upper, read result.
    //
    // Steps:
    //
    // 1. Create Engine and compile the Module from WAT:
    //    let engine = Engine::default();
    //    let module = Module::new(&engine, wat)?;
    //    let mut store = Store::new(&engine, ());
    //
    // 2. Instantiate (no imports needed for this module):
    //    let instance = Instance::new(&mut store, &module, &[])?;
    //
    // 3. Get the exported functions and memory:
    //    let memory = instance.get_memory(&mut store, "memory")
    //        .context("module must export 'memory'")?;
    //    let allocate_fn = instance.get_typed_func::<i32, i32>(&mut store, "allocate")?;
    //    let to_upper_fn = instance.get_typed_func::<(i32, i32), i64>(&mut store, "to_upper")?;
    //
    // 4. Write input string into guest memory:
    //    let input = "hello from the host!";
    //    let input_bytes = input.as_bytes();
    //    let input_ptr = allocate_fn.call(&mut store, input_bytes.len() as i32)?;
    //    memory.write(&mut store, input_ptr as usize, input_bytes)?;
    //    println!("  Wrote '{}' to guest memory at offset {}", input, input_ptr);
    //
    //    → allocate_fn returns an offset in the guest's linear memory.
    //    → memory.write copies our bytes into the guest's memory at that offset.
    //    → The guest will read these bytes when we call to_upper.
    //
    // 5. Call the transform function:
    //    let packed_result = to_upper_fn.call(&mut store, (input_ptr, input_bytes.len() as i32))?;
    //    let result_ptr = (packed_result >> 32) as i32;
    //    let result_len = (packed_result & 0xFFFF_FFFF) as i32;
    //    println!("  Guest returned: ptr={}, len={}", result_ptr, result_len);
    //
    //    → The guest reads our input, processes it, writes the result to a new
    //      allocation in its own memory, and returns the (offset, length) packed
    //      into an i64.
    //
    // 6. Read the result from guest memory:
    //    let mut result_buf = vec![0u8; result_len as usize];
    //    memory.read(&store, result_ptr as usize, &mut result_buf)?;
    //    let result_str = String::from_utf8(result_buf)?;
    //    println!("  Result: '{}'", result_str);
    //    assert_eq!(result_str, "HELLO FROM THE HOST!");
    //
    //    → memory.read copies bytes FROM the guest's memory into our buffer.
    //    → We then interpret the bytes as a UTF-8 string.
    //
    // 7. Print memory stats:
    //    println!("  Guest memory size: {} bytes ({} pages)",
    //        memory.data_size(&store),
    //        memory.data_size(&store) / 65536);
    //
    // After implementing, experiment:
    //   - Try writing beyond the allocated region (should memory.write fail?)
    //   - Try reading beyond memory size (memory.read returns MemoryAccessError)
    //   - Try growing memory: memory.grow(&mut store, 1)? → adds one 64 KiB page
    todo!("Exercise 5c: Exchange string data via WASM linear memory")
}

// =============================================================================
// Exercise 6: Performance — Native vs WASM
// =============================================================================

/// Benchmark native Rust execution against WASM execution via wasmtime.
///
/// # What to implement
///
/// 1. Implement a compute-intensive function in native Rust (matrix multiply)
/// 2. Write the same function as inline WAT (or compile wasm-lib to .wasm)
/// 3. Run both versions, measure wall-clock time, compute the overhead ratio
///
/// # Why this matters
///
/// Before choosing WASM for a use case, you need to understand its performance
/// characteristics:
///
/// - **Cranelift compilation**: wasmtime uses Cranelift (a fast code generator) to
///   compile WASM to native code. Cranelift optimizes for compilation speed over
///   peak performance, so the generated code is typically 50-80% of LLVM's speed.
///
/// - **Sandboxing overhead**: bounds checks on linear memory access, function call
///   trampolines between host and guest, and guard page setup all add overhead.
///
/// - **Where WASM shines**: CPU-bound computation with infrequent host calls.
///   The overhead per WASM instruction is near-zero; the cost is in boundary crossings.
///
/// - **Where WASM struggles**: I/O-heavy workloads (every I/O requires a host call),
///   workloads with many small function calls across the boundary, and workloads
///   that benefit from SIMD/vectorization (WASM SIMD support is still limited).
///
/// Typical overhead: 1.2x to 2x for compute-bound tasks.
/// Worst case: 5x+ for memory-access-heavy workloads due to bounds checks.
///
/// # Hints
///
/// Use `std::time::Instant` for timing:
/// ```text
/// let start = Instant::now();
/// // ... work ...
/// let elapsed = start.elapsed();
/// ```
///
/// For the WASM version, use a WAT module with the same algorithm.
/// Measure only the function call, not module compilation or instantiation.
fn exercise_6_benchmark() -> Result<()> {
    // --- Part A: Native benchmark ---
    println!("--- 6a: Native Rust benchmark ---\n");
    let n = 30;
    let native_result = native_fibonacci(n);
    let native_time = benchmark_native_fibonacci(n);
    println!("  fibonacci({}) = {} (native)", n, native_result);
    println!("  Native time: {:?}", native_time);

    // --- Part B: WASM benchmark ---
    println!("\n--- 6b: WASM benchmark ---\n");

    // WAT module with the same fibonacci algorithm
    let wat = r#"
        (module
            (func (export "fibonacci") (param $n i32) (result i64)
                (local $a i64)
                (local $b i64)
                (local $i i32)
                (local $temp i64)

                ;; a = 0, b = 1
                i64.const 0
                local.set $a
                i64.const 1
                local.set $b

                ;; for i = 0; i < n; i++
                (block $break
                    (loop $continue
                        local.get $i
                        local.get $n
                        i32.ge_u
                        br_if $break

                        ;; temp = a + b
                        local.get $a
                        local.get $b
                        i64.add
                        local.set $temp

                        ;; a = b
                        local.get $b
                        local.set $a

                        ;; b = temp
                        local.get $temp
                        local.set $b

                        ;; i++
                        local.get $i
                        i32.const 1
                        i32.add
                        local.set $i

                        br $continue
                    )
                )

                local.get $a
            )
        )
    "#;

    // TODO(human): Compile the WAT module, call fibonacci, measure time, compare.
    //
    // Steps:
    //
    // 1. Compile the module (measure compilation time separately):
    //    let engine = Engine::default();
    //    let compile_start = Instant::now();
    //    let module = Module::new(&engine, wat)?;
    //    let compile_time = compile_start.elapsed();
    //    println!("  WASM compilation time: {:?}", compile_time);
    //
    // 2. Instantiate (measure separately):
    //    let mut store = Store::new(&engine, ());
    //    let instance_start = Instant::now();
    //    let instance = Instance::new(&mut store, &module, &[])?;
    //    let instance_time = instance_start.elapsed();
    //    println!("  WASM instantiation time: {:?}", instance_time);
    //
    // 3. Get the function:
    //    let fib = instance.get_typed_func::<i32, i64>(&mut store, "fibonacci")?;
    //
    // 4. Warm up (call once to ensure JIT is ready):
    //    let _ = fib.call(&mut store, 1)?;
    //
    // 5. Benchmark the WASM call:
    //    let wasm_start = Instant::now();
    //    let iterations = 1_000_000;
    //    let mut wasm_result = 0i64;
    //    for _ in 0..iterations {
    //        wasm_result = fib.call(&mut store, n as i32)?;
    //    }
    //    let wasm_time = wasm_start.elapsed();
    //    println!("  fibonacci({}) = {} (wasm)", n, wasm_result);
    //    println!("  WASM time ({} iterations): {:?}", iterations, wasm_time);
    //
    // 6. Compare results:
    //    assert_eq!(wasm_result, native_result as i64);
    //
    // 7. Compute overhead ratio:
    //    let native_ns = native_time.as_nanos() as f64;
    //    let wasm_ns = wasm_time.as_nanos() as f64;
    //    println!("  Overhead ratio (wasm/native): {:.2}x", wasm_ns / native_ns);
    //    println!("  Per-call: native={:.0}ns, wasm={:.0}ns",
    //        native_ns / iterations as f64,
    //        wasm_ns / iterations as f64);
    //
    // Expected results:
    //   - Compilation: ~1-5ms (Cranelift is fast)
    //   - Instantiation: <1ms (no complex init)
    //   - Per-call overhead: fibonacci is so cheap that the host-guest boundary
    //     crossing dominates. You'll see WASM being 2-10x slower per call.
    //   - For expensive functions (matrix multiply, sorting large arrays), the
    //     overhead ratio approaches 1.2-1.5x because the computation dominates.
    //
    // Lesson: WASM overhead is in boundary crossings, not in computation.
    // For plugin systems where plugins do significant work per call, the
    // overhead is negligible. For tight inner loops that call WASM millions
    // of times, consider batching work into fewer, larger calls.
    todo!("Exercise 6: Benchmark native vs WASM fibonacci")
}

/// Native Rust fibonacci for comparison.
fn native_fibonacci(n: u32) -> u64 {
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    a
}

/// Benchmark native fibonacci over many iterations.
fn benchmark_native_fibonacci(n: u32) -> std::time::Duration {
    let iterations = 1_000_000;
    let start = Instant::now();
    let mut result = 0u64;
    for _ in 0..iterations {
        result = native_fibonacci(n);
    }
    let elapsed = start.elapsed();
    // Prevent compiler from optimizing away the loop
    std::hint::black_box(result);
    println!("  ({} iterations)", iterations);
    elapsed
}
