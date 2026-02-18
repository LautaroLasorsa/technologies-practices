//! WASM Library — Exercises 3 & 4
//!
//! This crate compiles to a `.wasm` library targeting `wasm32-unknown-unknown`.
//! It demonstrates two patterns for WASM interop:
//!
//! - **Exercise 3**: Raw linear memory data passing (manual pointer+length protocol)
//! - **Exercise 4**: wasm-bindgen automatic JS interop (#[wasm_bindgen])
//!
//! Build options:
//!   Raw WASM:     cargo build -p wasm-lib --target wasm32-unknown-unknown
//!   With JS glue: wasm-pack build wasm-lib --target web
//!
//! The raw build produces a .wasm file that the host (Exercise 5) can load.
//! The wasm-pack build generates additional JS glue for browser usage.

use wasm_bindgen::prelude::*;

// =============================================================================
// Exercise 3: Linear Memory and Host-Guest Data Passing
// =============================================================================
//
// These functions demonstrate the RAW data-passing protocol between host and guest.
// Before wasm-bindgen existed, this was the ONLY way to pass strings and byte arrays.
// Understanding this protocol is essential because:
//
// 1. It's what wasm-bindgen does under the hood
// 2. It's what you use in non-JS hosts (wasmtime, wasmer)
// 3. It reveals the fundamental constraint of WASM: only i32/i64/f32/f64 cross the boundary
//
// The protocol:
//   Host writes bytes into guest's linear memory at some offset.
//   Host calls a guest function, passing (offset: i32, length: i32).
//   Guest reads the bytes from its own linear memory at that offset.
//   Guest writes results back into linear memory.
//   Guest returns the offset (and possibly length) of the result.
//
// Memory ownership: the guest must expose allocation/deallocation functions
// so the host can reserve space in the guest's linear memory before writing.

/// Allocate `size` bytes in the WASM module's linear memory and return the offset.
///
/// # What to implement
///
/// Use Rust's global allocator to allocate a byte buffer of the given size,
/// then return the pointer as an i32 (WASM address).
///
/// # Why this matters for WASM
///
/// WASM linear memory is a flat byte array. The host can read/write it freely,
/// but the host doesn't know which regions are "in use" by the guest. Without
/// an allocation function, the host might overwrite the guest's data.
///
/// By exposing `allocate` and `deallocate`, the guest tells the host: "Here is
/// a safe region of my memory where you can write data." This is the standard
/// pattern for all WASM plugin systems.
///
/// # How WASM memory allocation works internally
///
/// Rust's `#[global_allocator]` manages a heap inside linear memory. When you
/// call `Vec::with_capacity(n)` or `Box::new(...)`, the allocator carves out
/// space from the linear memory. The pointer returned is an offset into that
/// same linear memory — so it's directly usable by the host.
///
/// Memory layout of a WASM module:
///   [0x0000 .. stack_end]    ← stack (grows downward)
///   [stack_end .. heap_start] ← static data, globals
///   [heap_start .. memory_end] ← heap (managed by allocator, grows via memory.grow)
///
/// # Hints
///
/// - `std::alloc::Layout::from_size_align(size, align)` creates an allocation layout
/// - `std::alloc::alloc(layout)` returns a raw pointer to allocated memory
/// - Cast the pointer to i32: `ptr as i32` (WASM addresses are 32-bit)
/// - Use alignment of 1 (byte alignment) for generic byte buffers
/// - If allocation fails (null pointer), return 0
///
/// # Safety
///
/// This function is `extern "C"` and `#[no_mangle]` so the host can call it by name.
/// The caller (host) must call `deallocate` with the same size when done.
#[no_mangle]
pub extern "C" fn allocate(size: i32) -> i32 {
    // TODO(human): Allocate `size` bytes and return the WASM memory offset.
    //
    // Steps:
    // 1. let layout = std::alloc::Layout::from_size_align(size as usize, 1).unwrap();
    // 2. let ptr = unsafe { std::alloc::alloc(layout) };
    // 3. if ptr.is_null() { return 0; }
    // 4. ptr as i32
    //
    // Why alignment 1? Because the host is writing raw bytes (a serialized string
    // or byte array). The guest will interpret them with its own alignment requirements
    // after reading. For structured data, you'd use a higher alignment.
    //
    // Why return i32? WASM uses 32-bit addressing. Every pointer in WASM linear
    // memory is an i32 offset from the start of the memory array. The host uses
    // this offset to write into the guest's memory via store.memory.write().
    todo!("Exercise 3a: Allocate bytes in WASM linear memory")
}

/// Deallocate a previously allocated buffer.
///
/// # What to implement
///
/// Free the memory at the given WASM offset that was allocated by `allocate`.
///
/// # Why this matters
///
/// Without deallocation, every call to `allocate` leaks memory. In a long-running
/// WASM plugin, this would eventually exhaust linear memory (which can grow but
/// never shrink). The host must call `deallocate` after it's done reading results.
///
/// # Hints
///
/// - Reconstruct the Layout: `Layout::from_size_align(size as usize, 1).unwrap()`
/// - Cast the offset back to a pointer: `offset as *mut u8`
/// - `unsafe { std::alloc::dealloc(ptr, layout) }`
#[no_mangle]
pub extern "C" fn deallocate(offset: i32, size: i32) {
    // TODO(human): Deallocate the buffer at the given WASM memory offset.
    //
    // Steps:
    // 1. let layout = std::alloc::Layout::from_size_align(size as usize, 1).unwrap();
    // 2. let ptr = offset as *mut u8;
    // 3. unsafe { std::alloc::dealloc(ptr, layout); }
    //
    // IMPORTANT: the (offset, size) pair must exactly match a previous `allocate` call.
    // Passing wrong values is undefined behavior (double-free, use-after-free, corruption).
    // In a real plugin system, you'd wrap this in a safer protocol (e.g., arena allocation
    // where you free everything at once after each plugin invocation).
    todo!("Exercise 3b: Deallocate WASM linear memory")
}

/// Transform a string in WASM linear memory: convert to uppercase.
///
/// # What to implement
///
/// Given a pointer and length to a UTF-8 string in linear memory, read the string,
/// convert it to uppercase, write the result back to a NEW allocation, and return
/// the (offset, length) of the result packed into an i64.
///
/// # Why this matters for WASM
///
/// This is the core data-passing pattern for WASM plugins:
///
///   1. Host allocates space in guest memory (calls `allocate`)
///   2. Host writes input data into guest memory (via store.memory.write)
///   3. Host calls guest function with (offset, length)
///   4. Guest reads input from its own memory, processes it
///   5. Guest allocates space for the result, writes it
///   6. Guest returns the result's (offset, length) to the host
///   7. Host reads the result from guest memory (via store.memory.read)
///   8. Host deallocates both buffers (calls `deallocate` twice)
///
/// We pack (offset, length) into a single i64 because WASM functions can only
/// return a single value (in WASM 1.0). The upper 32 bits hold the offset,
/// the lower 32 bits hold the length.
///
/// # Hints
///
/// - Read the string: `let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };`
/// - Convert: `let s = std::str::from_utf8(bytes).unwrap();`
/// - Uppercase: `let upper = s.to_uppercase();`
/// - Allocate for result: `let result_ptr = allocate(upper.len() as i32);`
/// - Write result: `unsafe { std::ptr::copy_nonoverlapping(upper.as_ptr(), result_ptr as *mut u8, upper.len()); }`
/// - Pack return value: `((result_ptr as i64) << 32) | (upper.len() as i64)`
///
/// # What would break
///
/// - If `ptr` doesn't point to valid UTF-8, `from_utf8` panics (WASM trap)
/// - If `len` extends beyond linear memory, the slice creation is UB (bounds check may not catch it)
/// - If the host forgets to deallocate, memory leaks
///
/// In production, you'd use a protocol like MessagePack or Protocol Buffers for
/// serialization, with length-prefixed messages. The raw pointer+length approach
/// shown here is the minimal viable data-passing protocol.
#[no_mangle]
pub extern "C" fn transform_string(ptr: i32, len: i32) -> i64 {
    // TODO(human): Read string from linear memory, uppercase it, write result back.
    //
    // Steps:
    // 1. Read input: unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) }
    // 2. Convert to str: std::str::from_utf8(bytes).unwrap()
    // 3. Transform: s.to_uppercase()
    // 4. Allocate result: allocate(upper.len() as i32)
    // 5. Write result: unsafe { std::ptr::copy_nonoverlapping(...) }
    // 6. Pack and return: ((result_offset as i64) << 32) | (result_len as i64)
    //
    // The pack/unpack pattern for returning two values:
    //   Pack:   ((offset as i64) << 32) | (length as i64)
    //   Unpack: offset = (packed >> 32) as i32;  length = (packed & 0xFFFF_FFFF) as i32;
    //
    // This is necessary because WASM 1.0 functions can only return one value.
    // WASM multi-return (proposal) and the Component Model eliminate this limitation,
    // but most runtimes still use the single-return model for compatibility.
    todo!("Exercise 3c: Transform string via linear memory protocol")
}

/// Compute the sum of an i32 array stored in linear memory.
///
/// # What to implement
///
/// Given a pointer to an array of i32 values in linear memory and the number
/// of elements, compute and return their sum.
///
/// # Why this matters for WASM
///
/// This demonstrates passing structured data (an array of integers) through
/// linear memory. The host writes i32 values into the guest's memory as
/// little-endian 4-byte sequences, then the guest reads them.
///
/// Key insight: the `ptr` parameter is a WASM address (offset into linear memory).
/// Inside the WASM module, dereferencing this pointer reads from the module's own
/// linear memory — it CANNOT access the host's memory. This is the memory safety
/// guarantee of WASM.
///
/// # Hints
///
/// - Cast: `let array = unsafe { std::slice::from_raw_parts(ptr as *const i32, count as usize) };`
/// - Sum: `array.iter().sum()`
///
/// # Memory layout
///
/// If the host writes [1, 2, 3] starting at offset 1024:
///   Memory[1024..1028] = 1i32.to_le_bytes()  → [0x01, 0x00, 0x00, 0x00]
///   Memory[1028..1032] = 2i32.to_le_bytes()  → [0x02, 0x00, 0x00, 0x00]
///   Memory[1032..1036] = 3i32.to_le_bytes()  → [0x03, 0x00, 0x00, 0x00]
///
/// Then calls: sum_array(1024, 3) → reads 3 i32s starting at offset 1024 → returns 6
#[no_mangle]
pub extern "C" fn sum_array(ptr: i32, count: i32) -> i32 {
    // TODO(human): Read an i32 array from linear memory and return the sum.
    //
    // Steps:
    // 1. let slice = unsafe { std::slice::from_raw_parts(ptr as *const i32, count as usize) };
    // 2. slice.iter().sum()
    //
    // Note: this assumes the host wrote the i32s in little-endian format
    // (which is natural on x86 and what WASM uses internally). WASM itself
    // is little-endian for multi-byte memory operations.
    todo!("Exercise 3d: Sum an i32 array from linear memory")
}

// =============================================================================
// Exercise 4: wasm-bindgen for Browser Interop
// =============================================================================
//
// These functions use #[wasm_bindgen] to automatically handle the linear memory
// dance from Exercise 3. wasm-bindgen generates JavaScript glue code that:
//
//   1. Converts JS strings to UTF-8 bytes
//   2. Writes them into WASM linear memory
//   3. Passes the (offset, length) to the Rust function
//   4. Reads the result back from linear memory
//   5. Converts it to a JS string
//
// All of this happens transparently — you just write normal Rust functions
// that accept and return String/&str, and wasm-bindgen handles the rest.
//
// Build with: wasm-pack build wasm-lib --target web
// This generates pkg/ with .wasm, .js, .d.ts files ready for browser use.

/// Greet a person by name.
///
/// # What to implement
///
/// Accept a name string and return a greeting string.
///
/// # Why this matters for WASM
///
/// Compare this with Exercise 3's `transform_string`: there, you manually
/// allocated memory, copied bytes, packed offsets into i64. Here, `#[wasm_bindgen]`
/// does all of that automatically.
///
/// Under the hood, wasm-bindgen generates a JS function that:
///   1. Encodes the JS string to UTF-8 using TextEncoder
///   2. Calls allocate() to get space in WASM memory
///   3. Copies the UTF-8 bytes into WASM memory
///   4. Calls the inner Rust function with (ptr, len)
///   5. Reads the return value (ptr, len of result string) from WASM memory
///   6. Decodes the result back to a JS string using TextDecoder
///   7. Calls deallocate() to free both buffers
///
/// You can verify this by examining the generated `pkg/wasm_lib.js` after building.
///
/// # Hints
///
/// - Just use `format!("Hello, {}! Greetings from WASM.", name)`
/// - The `#[wasm_bindgen]` macro handles String conversion automatically
#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    // TODO(human): Return a greeting string that includes the name.
    //
    // This is intentionally trivial — the learning is in what wasm-bindgen
    // generates, not in the Rust code itself.
    //
    // After implementing, build with wasm-pack and examine:
    //   wasm-pack build wasm-lib --target web
    //   cat wasm-lib/pkg/wasm_lib.js     ← see the generated JS glue
    //   cat wasm-lib/pkg/wasm_lib.d.ts   ← see the TypeScript type definitions
    //
    // In the .js file, look for the greet() function wrapper. You'll see it:
    //   - Uses TextEncoder to convert the JS string to UTF-8 bytes
    //   - Calls __wbindgen_malloc to allocate space in WASM memory
    //   - Copies bytes using new Uint8Array(wasm.memory.buffer, ptr, len)
    //   - Calls the actual WASM function
    //   - Reads the result using getStringFromWasm0() (TextDecoder)
    //   - Calls __wbindgen_free to deallocate
    todo!("Exercise 4a: Greet with wasm-bindgen string interop")
}

/// Process a JSON string and return analysis results.
///
/// # What to implement
///
/// Accept a JSON string (e.g., `{"numbers": [1, 2, 3, 4, 5]}`), parse it manually
/// (we avoid serde to keep the WASM binary small), extract the numbers, compute
/// statistics (sum, count, average), and return a formatted result string.
///
/// # Why this matters for WASM
///
/// Real-world WASM libraries process structured data. JSON is the most common
/// interchange format between JavaScript and WASM. This exercise shows that
/// string-based protocols work seamlessly with wasm-bindgen — the host sends
/// JSON as a string, the guest parses and processes it, returns a string.
///
/// For production use, you'd use `serde_json` + `serde_wasm_bindgen` to pass
/// structured data directly as JS objects. But the string-based approach
/// demonstrates the underlying mechanism.
///
/// # Hints
///
/// - Parse the JSON manually: find the array between [ and ]
/// - Split by commas, parse each number with .trim().parse::<f64>()
/// - Compute sum, count, average, min, max
/// - Return a formatted summary string
///
/// # Simplification
///
/// We do simple string parsing instead of pulling in serde_json because:
/// 1. serde_json adds ~100KB to the .wasm binary
/// 2. The point is string interop, not JSON parsing
/// 3. Keeping .wasm small demonstrates WASM's efficiency
#[wasm_bindgen]
pub fn analyze_json(json: &str) -> String {
    // TODO(human): Parse a simple JSON number array and return statistics.
    //
    // Expected input format: {"numbers": [1, 2, 3, 4, 5]}
    // Expected output: "count=5, sum=15, avg=3.00, min=1, max=5"
    //
    // Steps:
    // 1. Find the '[' and ']' positions in the JSON string
    // 2. Extract the substring between them
    // 3. Split by ',' and parse each element as f64
    // 4. Compute count, sum, average, min, max
    // 5. Return a formatted summary string
    //
    // Handle errors gracefully:
    //   - If no '[' or ']' found: return "Error: no array found in JSON"
    //   - If a number can't be parsed: skip it
    //   - If empty array: return "count=0, sum=0, avg=NaN, min=NaN, max=NaN"
    //
    // This demonstrates that WASM + wasm-bindgen can process text data
    // efficiently. The string crosses the boundary once (JS → WASM), gets
    // processed entirely in WASM, and the result crosses back once (WASM → JS).
    todo!("Exercise 4b: Parse JSON and return statistics via wasm-bindgen")
}

/// A struct exposed to JavaScript via wasm-bindgen.
///
/// # What to implement
///
/// Create a `Counter` struct with methods that JavaScript can call:
/// - `new(initial)` — constructor
/// - `increment()` — add 1
/// - `decrement()` — subtract 1
/// - `value()` — return current count
/// - `reset()` — set back to initial value
///
/// # Why this matters for WASM
///
/// wasm-bindgen can expose Rust structs as JavaScript classes. The struct lives
/// in WASM linear memory (heap-allocated), and JavaScript holds a handle (an i32
/// pointer) to it. Method calls from JavaScript pass this handle back to WASM,
/// which dereferences it to access the struct.
///
/// This is how Rust-based WASM libraries expose stateful APIs to JavaScript:
/// the Figma plugin API, the tree-sitter parser, and other production WASM
/// libraries use this pattern.
///
/// Under the hood:
/// - `#[wasm_bindgen]` on the struct generates a JS class with a constructor
/// - Each `#[wasm_bindgen]` method becomes a method on the JS class
/// - The struct is Box-allocated in WASM memory; JS holds the raw pointer
/// - When the JS object is garbage collected, wasm-bindgen calls `drop` on
///   the Rust struct (via the `free()` method on the JS class)
///
/// # Hints
///
/// - Use `#[wasm_bindgen]` on the struct AND on the `impl` block
/// - The constructor must be named `new` or annotated with `#[wasm_bindgen(constructor)]`
/// - Use `&self` for read methods, `&mut self` for mutating methods
#[wasm_bindgen]
pub struct Counter {
    // TODO(human): Define the Counter struct fields.
    //
    // Fields needed:
    //   count: i32     — current count value
    //   initial: i32   — initial value (for reset)
    //
    // Note: #[wasm_bindgen] structs must have all fields private (no pub).
    // JavaScript accesses the data only through the exposed methods.
    // This is actually good API design — encapsulation at the WASM boundary.
    count: i32,
    initial: i32,
}

#[wasm_bindgen]
impl Counter {
    /// Create a new Counter with the given initial value.
    ///
    /// In JavaScript, this becomes: `const counter = new Counter(10);`
    /// wasm-bindgen translates this to: allocate a Counter struct in WASM memory,
    /// initialize fields, return the pointer as an opaque JS handle.
    #[wasm_bindgen(constructor)]
    pub fn new(initial: i32) -> Counter {
        // TODO(human): Initialize the Counter with both count and initial set.
        //
        // Counter { count: initial, initial }
        todo!("Exercise 4c: Counter constructor")
    }

    /// Increment the counter by 1.
    ///
    /// In JavaScript: `counter.increment();`
    /// Under the hood: JS passes the struct pointer to WASM, which dereferences
    /// it, modifies the count field in linear memory, and returns.
    pub fn increment(&mut self) {
        // TODO(human): self.count += 1
        todo!("Exercise 4c: Counter increment")
    }

    /// Decrement the counter by 1.
    pub fn decrement(&mut self) {
        // TODO(human): self.count -= 1
        todo!("Exercise 4c: Counter decrement")
    }

    /// Get the current value.
    ///
    /// Returns an i32, which wasm-bindgen passes directly as a WASM i32 value.
    /// No memory copy needed — scalar values cross the boundary natively.
    pub fn value(&self) -> i32 {
        // TODO(human): return self.count
        todo!("Exercise 4c: Counter value getter")
    }

    /// Reset to the initial value.
    pub fn reset(&mut self) {
        // TODO(human): self.count = self.initial
        todo!("Exercise 4c: Counter reset")
    }
}
