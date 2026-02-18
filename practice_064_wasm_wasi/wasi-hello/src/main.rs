//! WASI Hello — Exercises 1 & 2
//!
//! This is a standalone program targeting `wasm32-wasip1`. When compiled to WASM,
//! it uses WASI system calls for I/O (stdout, filesystem). When compiled natively
//! (for cargo check), it uses the normal OS APIs — the Rust std library abstracts
//! this difference away.
//!
//! Build:  cargo build -p wasi-hello --target wasm32-wasip1
//! Run:    wasmtime target/wasm32-wasip1/debug/wasi-hello.wasm
//!
//! The compiled .wasm binary can be inspected with:
//!   wasm-tools print target/wasm32-wasip1/debug/wasi-hello.wasm | head -80

use std::env;
use std::fs;
use std::io::Write;

fn main() {
    println!("=== Exercise 1: Compile Rust to WASM ===");
    exercise_1_fibonacci();

    println!();
    println!("=== Exercise 2: WASI File Access ===");
    exercise_2_wasi_files();
}

// ─── Exercise 1: Compile Rust to WASM and Inspect the Binary ───────────

/// Compute the n-th Fibonacci number iteratively.
///
/// # What to implement
///
/// Write an iterative Fibonacci function that returns the n-th Fibonacci number
/// (0-indexed: fib(0)=0, fib(1)=1, fib(2)=1, fib(3)=2, ...).
///
/// # Why this matters for WASM
///
/// This is a pure computation — no I/O, no system calls, no memory allocation.
/// When compiled to WASM, this function translates directly to WASM instructions:
///
/// - Rust `i64` maps to WASM `i64`
/// - The loop becomes WASM `loop`/`br_if` control flow
/// - Local variables become WASM `local.get`/`local.set`
/// - No imports needed — this is a self-contained WASM function
///
/// After implementing this, compile to WASM and inspect the output:
///
///   cargo build -p wasi-hello --target wasm32-wasip1
///   wasm-tools print target/wasm32-wasip1/debug/wasi-hello.wasm | head -100
///
/// In the WAT output, look for:
/// - `(func $fibonacci ...)` — your function, translated to stack-based instructions
/// - `(memory ...)` — the linear memory declaration (even though fibonacci doesn't use it,
///   the WASI runtime needs memory for stdio buffers)
/// - `(import "wasi_snapshot_preview1" "fd_write" ...)` — WASI imports for println!
/// - `(export "memory" ...)` — the module exports its linear memory to the host
///
/// # Key WASM insight
///
/// Notice that `fibonacci` itself needs NO imports — it's pure computation.
/// The WASI imports (`fd_write`, `proc_exit`, etc.) come from `println!` in main.
/// A WASM module with only pure functions would have ZERO imports, making it
/// completely self-contained and trivially sandboxed.
///
/// # Hints
///
/// - Use two variables `a` and `b`, starting at 0 and 1
/// - Loop n times, updating: `(a, b) = (b, a + b)`
/// - Return `a` after the loop
fn fibonacci(n: u32) -> u64 {
    // TODO(human): Implement iterative Fibonacci.
    //
    // This is deliberately simple — the point is not the algorithm but what
    // happens when you compile it to WASM. After implementing:
    //
    // 1. Build: cargo build -p wasi-hello --target wasm32-wasip1
    // 2. Inspect: wasm-tools print target/wasm32-wasip1/debug/wasi-hello.wasm > output.wat
    // 3. Open output.wat and find the fibonacci function
    // 4. Count the WASI imports — these come from println!, not from fibonacci
    //
    // WASM observation: the fibonacci function in WAT will use:
    //   - local.get / local.set (read/write local variables)
    //   - i64.add (64-bit addition, your a + b)
    //   - loop / br_if (the loop construct)
    //   - NO memory loads/stores (fibonacci doesn't touch linear memory)
    //
    // This demonstrates that WASM is very close to native — a simple loop with
    // locals compiles to a simple WASM loop with locals, no overhead.
    todo!("Exercise 1: Implement iterative fibonacci(n) -> u64")
}

fn exercise_1_fibonacci() {
    let test_cases = [(0, 0u64), (1, 1), (2, 1), (5, 5), (10, 55), (20, 6765), (50, 12586269025)];

    for (n, expected) in test_cases {
        let result = fibonacci(n);
        assert_eq!(result, expected, "fibonacci({}) should be {}, got {}", n, expected, result);
        println!("  fibonacci({}) = {}", n, result);
    }

    println!("  All fibonacci tests passed!");
}

// ─── Exercise 2: WASI Capabilities and File Access ─────────────────────

/// Write a message to a file using WASI filesystem capabilities.
///
/// # What to implement
///
/// 1. Create (or overwrite) a file at the given `path`
/// 2. Write the `message` string to it
/// 3. Print confirmation to stdout
///
/// # Why this matters for WASM / WASI
///
/// In native Rust, `std::fs::write("any/path", data)` just works — the OS gives
/// your process ambient filesystem access. In WASI, this is NOT the case:
///
/// - A WASM module has NO filesystem access by default
/// - The host (wasmtime) must explicitly grant directory access via `--dir=<path>`
/// - WASI uses `openat`-style APIs: all paths are relative to a pre-opened directory FD
/// - The Rust std library transparently maps `std::fs` calls to WASI syscalls
///
/// When you run:
///   wasmtime target/wasm32-wasip1/debug/wasi-hello.wasm
///
/// The file write will FAIL with a permission error because no directories are granted.
///
/// When you run:
///   wasmtime --dir=./testdata target/wasm32-wasip1/debug/wasi-hello.wasm
///
/// The write succeeds because `./testdata` is pre-opened and accessible.
///
/// # Capability-based security in action
///
/// Try accessing a path OUTSIDE the granted directory:
///   write_file("/etc/passwd", "hacked")  → permission denied (WASI blocks it)
///   write_file("../secret.txt", "data")  → permission denied (path escapes pre-opened dir)
///   write_file("testdata/ok.txt", "hi")  → succeeds (within granted directory)
///
/// This is the principle of **least privilege**: the module can only access
/// what the host explicitly grants. There is no way to "escape" the sandbox —
/// there is no `open()` syscall in WASI, only `openat()` relative to granted FDs.
///
/// # Hints
///
/// - `std::fs::File::create(path)` creates/truncates a file (maps to WASI `path_open`)
/// - `file.write_all(message.as_bytes())` writes the content
/// - Or simply: `std::fs::write(path, message)` does both in one call
/// - Handle errors with `match` or `if let Err(e)` — don't unwrap, because
///   WASI permission errors are expected when directories aren't granted
fn write_file(path: &str, message: &str) {
    // TODO(human): Create/overwrite a file and write the message to it.
    //
    // Use std::fs — Rust's standard library automatically routes these calls
    // through WASI syscalls when compiled for wasm32-wasip1.
    //
    // Important: handle the error case! When running in WASI without --dir,
    // this WILL fail with a "capabilities insufficient" error. Print the error
    // instead of panicking, so the user can see what WASI capability errors
    // look like.
    //
    // Pattern:
    //   match std::fs::write(path, message) {
    //       Ok(()) => println!("  Wrote to {}", path),
    //       Err(e) => println!("  Error writing {}: {} (this is expected without --dir)", path, e),
    //   }
    todo!("Exercise 2a: Write message to file using std::fs (WASI-backed)")
}

/// Read a file and return its contents as a String.
///
/// # What to implement
///
/// Read the file at `path` and return its contents. Handle the error case
/// (file not found, permission denied) by returning an error message.
///
/// # Why this matters for WASM / WASI
///
/// Like file writing, file reading requires WASI capabilities. The host must
/// grant access to the directory containing the file. This function demonstrates
/// the round-trip: write a file (Exercise 2a), then read it back (Exercise 2b).
///
/// Under the hood, `std::fs::read_to_string` calls WASI's `fd_read` syscall,
/// which reads from a file descriptor that was opened via `path_open` relative
/// to a pre-opened directory capability.
///
/// # Hints
///
/// - `std::fs::read_to_string(path)` reads the entire file into a String
/// - Return `Ok(contents)` on success, `Err(error_message)` on failure
fn read_file(path: &str) -> Result<String, String> {
    // TODO(human): Read the file and return its contents.
    //
    // Use std::fs::read_to_string and convert the error to a String.
    //
    // Pattern:
    //   std::fs::read_to_string(path).map_err(|e| format!("Error reading {}: {}", path, e))
    todo!("Exercise 2b: Read file contents using std::fs (WASI-backed)")
}

/// List the contents of a directory.
///
/// # What to implement
///
/// List all entries in the given directory path and print their names.
///
/// # Why this matters for WASM / WASI
///
/// Directory listing requires the same WASI capability as file access — the
/// directory must be pre-opened by the host. WASI's `fd_readdir` syscall
/// returns directory entries relative to the granted directory FD.
///
/// This exercise demonstrates that ALL filesystem operations (not just
/// read/write) are gated by capabilities. A module cannot even list directory
/// contents without explicit permission.
///
/// # Hints
///
/// - `std::fs::read_dir(path)` returns an iterator of directory entries
/// - Each entry has `.file_name()` (OsString) and `.file_type()`
/// - Handle the error case (directory not found, permission denied)
fn list_directory(path: &str) {
    // TODO(human): List directory contents using std::fs::read_dir.
    //
    // Print each entry's name and whether it's a file or directory.
    // Handle errors gracefully (the directory might not be accessible in WASI).
    //
    // Pattern:
    //   match std::fs::read_dir(path) {
    //       Ok(entries) => {
    //           for entry in entries {
    //               if let Ok(entry) = entry {
    //                   let kind = if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
    //                       "dir"
    //                   } else {
    //                       "file"
    //                   };
    //                   println!("    [{}] {:?}", kind, entry.file_name());
    //               }
    //           }
    //       }
    //       Err(e) => println!("  Error listing {}: {}", path, e),
    //   }
    todo!("Exercise 2c: List directory contents using std::fs (WASI-backed)")
}

fn exercise_2_wasi_files() {
    // Print WASI environment info
    println!("  Program args: {:?}", env::args().collect::<Vec<_>>());
    println!("  Current dir: {:?}", env::current_dir().unwrap_or_default());

    // Attempt to write a file — this demonstrates WASI capabilities.
    //
    // When run with: wasmtime target/wasm32-wasip1/debug/wasi-hello.wasm
    //   → FAILS: no directory access granted
    //
    // When run with: wasmtime --dir=./testdata target/wasm32-wasip1/debug/wasi-hello.wasm
    //   → SUCCEEDS: testdata/ is pre-opened
    let test_file = "testdata/hello_wasi.txt";
    let message = "Hello from WebAssembly via WASI!\nWritten by a sandboxed WASM module.\n";

    println!();
    println!("  Attempting to write: {}", test_file);
    write_file(test_file, message);

    println!();
    println!("  Attempting to read back: {}", test_file);
    match read_file(test_file) {
        Ok(contents) => println!("  Contents: {:?}", contents),
        Err(e) => println!("  {}", e),
    }

    println!();
    println!("  Attempting to list directory: testdata/");
    list_directory("testdata");

    // Demonstrate capability boundary — try accessing outside granted dir
    println!();
    println!("  Attempting to write OUTSIDE granted directory (should fail in WASI):");
    write_file("/tmp/escape_attempt.txt", "This should fail in WASI");

    println!();
    println!("  Exercise 2 complete.");
    println!("  Try running with different --dir flags to see capability-based access in action.");
}
