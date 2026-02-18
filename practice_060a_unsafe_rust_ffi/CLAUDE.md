# Practice 060a: Unsafe Rust & FFI — Raw Pointers, bindgen & repr(C)

## Technologies

- **libc** — Raw POSIX/C type definitions (`c_int`, `c_char`, `c_void`, `size_t`)
- **cc** (build dependency) — Compile C source files from `build.rs`, auto-detects MSVC/gcc/clang
- **bindgen** (optional build dependency, `--features bindgen`) — Generate Rust FFI bindings from C header files automatically; requires LLVM/libclang
- **std::ffi** — `CStr`, `CString` for safe C string interop

## Stack

- Rust (cargo, edition 2024)
- C (compiled via the `cc` crate; MSVC on Windows, gcc/clang on Linux/macOS)

## Theoretical Context

### What is Unsafe Rust and Why Does It Exist?

Rust's safety guarantees are enforced at compile time by the borrow checker: no dangling references, no data races, no use-after-free. But some operations are **impossible to express within the borrow checker's model** — dereferencing a raw pointer from C, writing to a hardware register, implementing a custom allocator. For these cases, Rust provides `unsafe` blocks that tell the compiler: "I, the programmer, have manually verified that this code upholds Rust's invariants."

`unsafe` does **not** disable the borrow checker for surrounding code. It only unlocks five specific capabilities (the "unsafe superpowers") within that block. The rest of Rust's safety checks remain active. The philosophy is to **minimize** the surface area of `unsafe` and wrap it in safe abstractions — so callers never need to think about it.

Sources: [The Rust Programming Language, ch. 20.1](https://doc.rust-lang.org/book/ch20-01-unsafe-rust.html), [The Rustonomicon](https://doc.rust-lang.org/nomicon/)

### The Five Unsafe Superpowers

Inside an `unsafe` block (or `unsafe fn`), you can:

| # | Superpower | Why it's unsafe |
|---|-----------|-----------------|
| 1 | **Dereference a raw pointer** (`*const T`, `*mut T`) | Compiler cannot verify the pointer is valid, aligned, or non-null |
| 2 | **Call an unsafe function or method** | The function has preconditions the compiler cannot check (e.g., pointer validity) |
| 3 | **Access or modify a mutable static variable** | Global mutable state is inherently racy in multi-threaded code |
| 4 | **Implement an unsafe trait** | The trait has invariants the compiler cannot verify (e.g., `Send`, `Sync`) |
| 5 | **Access fields of a `union`** | Only one variant is valid at a time; the compiler cannot know which |

Creating raw pointers is safe — only **dereferencing** them requires `unsafe`. This is a deliberate design: you can pass raw pointers around freely; the danger is only at the point of use.

Source: [Unsafe Rust — The Rust Book](https://doc.rust-lang.org/book/ch20-01-unsafe-rust.html)

### Raw Pointers: `*const T` and `*mut T`

Raw pointers are Rust's equivalent of C pointers:

- `*const T` — immutable raw pointer (like `const T*` in C)
- `*mut T` — mutable raw pointer (like `T*` in C)

Unlike references (`&T`, `&mut T`), raw pointers:
- Are **not guaranteed to point to valid memory** (may be null, dangling, or misaligned)
- **Ignore borrowing rules** — you can have multiple `*mut T` to the same data
- **Have no lifetime** — the compiler does not track when the pointed-to data is freed
- Can be **cast between types** freely (`*const u8` to `*const i32`)

You convert between references and raw pointers:
```rust
let x = 42;
let ptr: *const i32 = &x;        // reference → raw pointer (safe)
let val = unsafe { *ptr };        // dereference raw pointer (unsafe)
```

Source: [The Rustonomicon — Raw Pointers](https://doc.rust-lang.org/nomicon/raw-pointers.html)

### Foreign Function Interface (FFI)

FFI allows Rust to call functions written in other languages (and vice versa). The key mechanism is `extern "C"`:

```rust
// Declaring a C function that Rust can call
// (Rust 2024 edition requires `unsafe` on extern blocks)
unsafe extern "C" {
    fn abs(x: i32) -> i32;
}

// Exposing a Rust function that C can call
// (Rust 2024 edition requires `unsafe(no_mangle)` instead of `no_mangle`)
#[unsafe(no_mangle)]
pub extern "C" fn rust_add(a: i32, b: i32) -> i32 {
    a + b
}
```

**Rust 2024 edition changes:** The `extern "C"` block must be `unsafe extern "C"` (acknowledging that the declarations are a trust boundary). The `#[no_mangle]` attribute must be `#[unsafe(no_mangle)]` (acknowledging that exposing symbols affects linker-level safety).

**Why `extern "C"`?** Every language has a **calling convention** — the rules for how arguments are passed (registers vs. stack), how return values are retrieved, and who cleans up the stack. Rust's default calling convention is unspecified and can change between compiler versions. `extern "C"` forces the **C calling convention** (cdecl on x86, System V AMD64 ABI on x86-64 Linux, Microsoft x64 on Windows), which is the universal lingua franca of inter-language interop.

**Why `#[no_mangle]`?** Rust (like C++) mangles function names to encode type information (e.g., `_ZN4main7rust_add17h...E`). C expects plain symbol names. `#[no_mangle]` preserves the original function name in the compiled binary.

Sources: [The Rustonomicon — FFI](https://doc.rust-lang.org/nomicon/ffi.html), [Effective Rust, Item 34](https://effective-rust.com/ffi.html)

### `repr(C)` — Controlling Memory Layout

Rust's default struct layout (`repr(Rust)`) is **unspecified** — the compiler freely reorders fields and inserts padding for optimal performance. This layout can change between compiler versions.

`#[repr(C)]` forces C-compatible layout: fields are laid out **in declaration order** with padding matching what a C compiler would produce. This is **mandatory** for any struct that crosses the FFI boundary.

```rust
#[repr(C)]
struct Point {
    x: f64,  // offset 0, size 8
    y: f64,  // offset 8, size 8
}            // total size: 16, alignment: 8
```

Without `repr(C)`, Rust might reorder fields or use different padding, causing C code to read garbage when accessing the struct's fields.

Other layout attributes:
- `#[repr(C, packed)]` — no padding (may cause unaligned access)
- `#[repr(C, align(N))]` — force minimum alignment to N bytes
- `#[repr(transparent)]` — single-field struct has the same layout as its field (useful for newtypes over FFI types)

Sources: [The Rustonomicon — repr(C)](https://doc.rust-lang.org/nomicon/other-reprs.html), [Rust Reference — Type Layout](https://doc.rust-lang.org/reference/type-layout.html)

### String Handling Across FFI

C strings (`char*`) and Rust strings (`String`/`&str`) are fundamentally different:

| Property | Rust `String`/`&str` | C `char*` |
|----------|---------------------|-----------|
| Encoding | UTF-8 guaranteed | Arbitrary bytes (usually ASCII) |
| Termination | Length-prefixed (no null terminator) | Null-terminated (`\0`) |
| Interior nulls | Allowed | Impossible (null = end of string) |

Rust provides two bridge types:
- **`CString`** (owned) — Rust string → C string. Adds null terminator, rejects interior nulls. Use `CString::new("hello")?.as_ptr()` to get a `*const c_char`.
- **`CStr`** (borrowed) — C string → Rust. Wraps a `*const c_char`, finds the null terminator. Use `CStr::from_ptr(ptr).to_str()?` to get a `&str`.

**Critical rule:** The `CString` must **outlive** the raw pointer obtained from `.as_ptr()`. If the `CString` is dropped, the pointer dangles.

Sources: [std::ffi::CString docs](https://doc.rust-lang.org/std/ffi/struct.CString.html), [Rust FFI Patterns — Accepting Strings](https://rust-unofficial.github.io/patterns/idioms/ffi/accepting-strings.html)

### bindgen: Auto-Generating Rust Bindings from C Headers

Writing FFI declarations by hand is tedious and error-prone. [bindgen](https://rust-lang.github.io/rust-bindgen/) automates this: given a C header file, it produces Rust `extern "C"` declarations, struct definitions, constants, and type aliases.

Typical workflow in `build.rs`:
1. Use the `cc` crate to compile C source files into a static library
2. Use `bindgen::Builder` to parse the C header and generate `bindings.rs`
3. `include!()` the generated file in your Rust source

bindgen requires **libclang** (the Clang C API). On Windows, install LLVM and set `LIBCLANG_PATH` to the LLVM `bin/` directory.

Source: [The bindgen User Guide](https://rust-lang.github.io/rust-bindgen/)

### Ownership Across FFI: Who Frees the Memory?

The golden rule: **memory allocated by one language must be freed by the same language**. Rust's allocator and C's `malloc`/`free` are different (and mixing them is UB).

Pattern for Rust objects exposed to C:
```rust
#[no_mangle]
pub extern "C" fn widget_new() -> *mut Widget {
    Box::into_raw(Box::new(Widget::default()))  // Rust allocates, gives C a raw ptr
}

#[no_mangle]
pub extern "C" fn widget_free(ptr: *mut Widget) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }    // C returns ptr, Rust frees
    }
}
```

`Box::into_raw` moves the heap allocation out of Rust's ownership system (Rust "forgets" it). `Box::from_raw` reclaims ownership so the destructor runs. Never call C's `free()` on Rust-allocated memory, and never call Rust's `drop` on C-allocated memory.

Sources: [Effective Rust, Item 34](https://effective-rust.com/ffi.html), [Rust FFI Omnibus — Objects](http://jakegoulding.com/rust-ffi-omnibus/objects/)

### Common Undefined Behavior Pitfalls in FFI

| Pitfall | What happens | Prevention |
|---------|-------------|------------|
| **Dangling pointer** | C frees memory, Rust dereferences it | Clear ownership protocol; document who frees |
| **Aliasing violation** | `&mut T` and `*mut T` coexist pointing to same data | Never create `&mut` while a raw pointer alias exists |
| **Null pointer dereference** | C returns NULL, Rust dereferences without checking | Always check for null before `*ptr` or `Box::from_raw` |
| **Type mismatch** | Rust declares `c_int` but C uses `long` | Use `libc` types; verify sizes with `static_assert`/`size_of` |
| **Unwind across FFI** | Rust panic propagates into C code | Wrap callbacks with `std::panic::catch_unwind` |
| **Wrong calling convention** | Omitting `extern "C"` | Always specify `extern "C"` on both sides |
| **Layout mismatch** | Missing `#[repr(C)]` on structs shared with C | Always use `repr(C)` for FFI structs |

**Miri limitation:** Miri (Rust's undefined behavior detector) **cannot analyze the C side** of FFI calls. It only validates Rust code. C-side UB (buffer overflows, use-after-free in C) is invisible to Miri. For full validation, combine Miri with C sanitizers (ASan, UBSan).

Sources: [Rust Reference — Undefined Behavior](https://doc.rust-lang.org/reference/behavior-considered-undefined.html), [A Study of UB Across FFI Boundaries](https://arxiv.org/html/2404.11671v3)

## Description

Build a series of exercises that progressively teach unsafe Rust and FFI:

1. **Raw pointer basics** — Create, cast, and dereference raw pointers; understand the difference from references
2. **Unsafe functions & safe abstractions** — Write unsafe functions, then wrap them in safe APIs
3. **Calling libc from Rust** — Use the `libc` crate to call POSIX/C standard library functions directly
4. **Calling a custom C library from Rust** — Write a C math library, compile it with `cc`, declare `extern "C"` bindings manually, call from Rust
5. **repr(C) structs & complex data across FFI** — Share structs, strings, and arrays between Rust and C
6. **bindgen: Auto-generated bindings** — Use bindgen to auto-generate bindings from a C header instead of writing them by hand
7. **Capstone: Bidirectional FFI** — Expose Rust functions to C, call C from Rust, manage ownership across the boundary

### What you'll learn

- When and why to use `unsafe` (and when not to)
- Raw pointer creation, casting, arithmetic, and dereferencing
- The `extern "C"` ABI and `#[no_mangle]`
- `repr(C)` struct layout vs default Rust layout
- `CString`/`CStr` for string interop
- `Box::into_raw`/`Box::from_raw` for ownership transfer
- Using `cc` to compile C code in `build.rs`
- Using `bindgen` to auto-generate Rust bindings from C headers
- Common UB pitfalls and how to avoid them

## Instructions

### Phase 1: Setup & Build System (~10 min)

1. Review `Cargo.toml` — note the `libc` dependency and `cc`/`bindgen` as build dependencies
2. Review `build.rs` — understand how `cc::Build` compiles C files and how bindgen generates bindings
3. Review `c_src/mathlib.h` and `c_src/mathlib.c` — the C library you'll call from Rust
4. Build the project: `cargo build` — this compiles the C code and generates bindgen bindings
5. **Key question:** Why does `build.rs` run *before* the rest of the crate compiles? What would break if it didn't?

### Phase 2: Raw Pointer Fundamentals (Exercise 1, ~15 min)

1. Open `src/ex1_raw_pointers.rs` — five functions to implement
2. **User implements:** `pointer_from_reference()` — Convert a reference to a raw pointer and back. This is the most basic unsafe operation: you must dereference the raw pointer inside an `unsafe` block. Teaches that creating raw pointers is safe; only dereferencing is unsafe.
3. **User implements:** `pointer_arithmetic()` — Use `.add()` / `.offset()` on raw pointers to walk through an array. This mirrors how C array access works (`*(arr + i)`) and teaches why pointer arithmetic is unsafe (out-of-bounds = UB).
4. **User implements:** `null_pointer_check()` — Accept a `*const i32`, check for null, return `Option<i32>`. Teaches defensive FFI programming: never trust a pointer from foreign code.
5. **User implements:** `cast_pointer_types()` — Reinterpret `*const u8` bytes as `*const u32`. Teaches type punning and alignment requirements.
6. **User implements:** `swap_via_raw_pointers()` — Swap two values using only raw pointers (no `std::mem::swap`). Teaches why aliasing rules matter: with `&mut`, the borrow checker prevents simultaneous mutable access, but with raw pointers you must ensure correctness yourself.
7. Run: `cargo run` (select exercise 1)

### Phase 3: Unsafe Functions & Safe Wrappers (Exercise 2, ~15 min)

1. Open `src/ex2_unsafe_functions.rs` — three functions to implement
2. **User implements:** `unsafe_get_unchecked()` — Write an unsafe function that indexes a slice without bounds checking (like C array access). Then write a safe wrapper that validates the index first. This teaches the core pattern: unsafe implementation + safe API = zero-cost safety.
3. **User implements:** `split_at_mut_raw()` — Reimplement `slice::split_at_mut` using raw pointers. The standard library does exactly this — the borrow checker cannot prove that two non-overlapping mutable slices from the same array don't alias, so `unsafe` is required. This is the canonical example of why `unsafe` exists.
4. **User implements:** `transmute_demo()` — Use `std::mem::transmute` to convert between types of the same size. Demonstrates the most dangerous unsafe operation and why it should be avoided when safer alternatives exist (`as` casts, `from_ne_bytes`, etc.).
5. Run: `cargo run` (select exercise 2)

### Phase 4: Calling libc from Rust (Exercise 3, ~10 min)

1. Open `src/ex3_libc_calls.rs` — two functions to implement
2. **User implements:** `libc_string_length()` — Call `libc::strlen` on a Rust string. This requires converting a Rust `&str` to a C string (`CString`), getting a raw pointer, and passing it to the C function. Teaches the full Rust→C string pipeline.
3. **User implements:** `libc_sort_array()` — Call `libc::qsort` to sort an array of integers. This teaches passing Rust data to C functions that expect raw pointers and function pointers (callbacks). The comparison callback must be `extern "C"` because C will call it.
4. Run: `cargo run` (select exercise 3)

### Phase 5: Custom C Library via Manual FFI (Exercise 4, ~15 min)

1. Open `src/ex4_manual_ffi.rs` — review the `extern "C"` block and implement three functions
2. The `extern "C"` block declares functions from `c_src/mathlib.c` that were compiled by `build.rs`. These declarations are **hand-written** — you must ensure they match the C signatures exactly. A mismatch is silent UB.
3. **User implements:** `call_c_add()` — Call the C `mathlib_add` function. Simplest possible FFI call: pass two integers, get one back. No pointers, no allocation, no ownership issues.
4. **User implements:** `call_c_dot_product()` — Call the C `mathlib_dot_product` function, passing a Rust slice as a C array (pointer + length). Teaches the fundamental pattern for passing Rust collections to C: `.as_ptr()` for the data pointer, `.len()` for the count.
5. **User implements:** `call_c_format_vector()` — Call a C function that returns a heap-allocated string (`char*`). You must take ownership and free the memory using C's `libc::free`. Teaches the ownership boundary: C allocated it, so C must free it (not Rust's allocator).
6. Run: `cargo run` (select exercise 4)

### Phase 6: repr(C) Structs & Complex Data (Exercise 5, ~15 min)

1. Open `src/ex5_repr_c_structs.rs` — review the `#[repr(C)]` structs and implement three functions
2. **User implements:** `create_and_pass_point()` — Create a `#[repr(C)]` `Point3D` struct in Rust, pass it to a C function that computes its magnitude. Teaches that `repr(C)` guarantees layout compatibility — without it, the C function would read garbage.
3. **User implements:** `receive_struct_from_c()` — Call a C function that returns a struct by value. Teaches that `repr(C)` structs can be returned from C functions just like in C; Rust handles the copy.
4. **User implements:** `array_of_structs_to_c()` — Pass a Rust `Vec<Point3D>` to a C function that processes an array of structs. Teaches that `Vec<T>` with `repr(C)` T is layout-compatible with `T[]` in C (contiguous memory, same element layout).
5. Run: `cargo run` (select exercise 5)

### Phase 7: bindgen Auto-Generated Bindings (Exercise 6, ~15 min)

1. Open `src/ex6_bindgen.rs` — uses auto-generated bindings from `bindings.rs`
2. Review the generated `target/debug/build/*/out/bindings.rs` (or `$OUT_DIR/bindings.rs`) to see what bindgen produced. Compare with your hand-written declarations from Exercise 4. Note how bindgen handles types, constants, and structs.
3. **User implements:** `bindgen_call_add()` — Same as Exercise 4, but using bindgen-generated bindings. Teaches that bindgen eliminates hand-written declarations and the risk of type mismatches.
4. **User implements:** `bindgen_use_struct()` — Use a bindgen-generated struct type to call C functions. Note that bindgen generates `#[repr(C)]` automatically.
5. **User implements:** `bindgen_use_constant()` — Access a `#define` constant that bindgen converted to a Rust `const`. Teaches how bindgen maps C preprocessor macros to Rust constants.
6. Run: `cargo run` (select exercise 6)
7. **Key question:** What are the trade-offs between hand-written FFI declarations and bindgen? When would you choose each?

### Phase 8: Capstone — Bidirectional FFI (Exercise 7, ~20 min)

1. Open `src/ex7_capstone.rs` — the full round-trip: Rust→C→Rust
2. **User implements:** `rust_string_processor_for_c()` — Expose a Rust function with `extern "C"` + `#[no_mangle]` that C can call. The function receives a `*const c_char`, processes it in Rust (e.g., converts to uppercase), and returns a new `*mut c_char`. Teaches the Rust→C export pattern and the critical ownership rule: Rust allocates the return string, so Rust must also provide a free function.
3. **User implements:** `rust_free_string()` — The companion free function that C calls to deallocate the string returned by `rust_string_processor_for_c`. Uses `CString::from_raw` to reclaim ownership. Teaches that every `into_raw()` must have a matching `from_raw()`.
4. **User implements:** `round_trip_demo()` — Orchestrate a full round trip: create data in Rust, pass to C for processing, get results back, pass Rust callback to C. Combines everything from previous exercises.
5. Run: `cargo run` (select exercise 7)
6. **Key question:** If you forget to call `rust_free_string`, what happens? How would you detect this in production?

## Motivation

FFI is a **critical professional skill** for systems programmers:

- **OS API access**: All operating system APIs (Win32, POSIX, Linux syscalls) are C interfaces. Even Rust's `std` library uses FFI internally.
- **C library ecosystem**: Thousands of battle-tested C libraries (OpenSSL, SQLite, zlib, BLAS/LAPACK) are accessed via FFI. Writing wrappers is a common task.
- **Performance-critical code**: Sometimes the fastest implementation is in C/C++ or uses CPU-specific intrinsics only available through C headers.
- **Gradual migration**: Many companies adopt Rust by replacing C/C++ modules incrementally. FFI is the boundary between old and new code (Mozilla Firefox, Dropbox, AWS).
- **AutoScheduler.AI relevance**: Optimization engines often have C/C++ cores (solvers, LP algorithms) wrapped in higher-level languages. Understanding FFI is essential for integrating Rust with existing C++ optimization code.

This practice builds the foundation for Practice 060b (advanced FFI patterns) and is prerequisite knowledge for practices involving CUDA (019a/b), HFT C++ interop (020a/b), and WebAssembly (064).

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Build | `cargo build` | Compile C code via `cc` crate, use fallback bindings, build Rust |
| Build (with bindgen) | `cargo build --features bindgen` | Auto-generate bindings from C header (requires LLVM/libclang) |
| Build (check only) | `cargo check` | Type-check without producing binary (faster feedback) |
| Run all | `cargo run` | Run all exercises sequentially |
| Run one exercise | `cargo run -- 1` | Run only exercise 1 (replace `1` with `1`-`7`) |
| Run with output | `cargo run 2>&1` | Capture stderr (build warnings) alongside stdout |
| Clean | `cargo clean` | Remove build artifacts (forces C recompilation on next build) |
| View generated bindings | View file at `target/debug/build/unsafe-ffi-practice-*/out/bindings.rs` | Inspect the bindings used by Exercise 6 (fallback or bindgen-generated) |
| Check for UB (Rust side) | `cargo +nightly miri run` | Run under Miri to detect UB in Rust code (does NOT check C side) |

## Notes

*(To be filled during practice.)*
