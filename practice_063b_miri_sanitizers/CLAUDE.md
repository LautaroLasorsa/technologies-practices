# Practice 063b: Memory Safety Verification — Miri & Sanitizers

## Technologies

- **Miri** — Rust's MIR interpreter for detecting undefined behavior at runtime
- **AddressSanitizer (ASAN)** — LLVM-based runtime detector for memory errors (out-of-bounds, use-after-free, leaks)
- **ThreadSanitizer (TSAN)** — LLVM-based runtime detector for data races

## Stack

- Rust (nightly, required for Miri and `-Zsanitizer` flags)
- cargo miri (installed via `rustup component add miri`)

## Theoretical Context

### What Is Undefined Behavior in Rust?

Safe Rust guarantees memory safety and data-race freedom at compile time through the ownership and borrow checker. However, `unsafe` blocks opt out of these guarantees — the programmer takes responsibility for upholding invariants that the compiler cannot verify. When `unsafe` code violates these invariants, the result is **undefined behavior (UB)**: the compiler is allowed to do literally anything, because it assumes UB never happens and optimizes accordingly.

UB in Rust includes (non-exhaustive):
- **Dereferencing dangling or null pointers** — use-after-free, use-after-scope
- **Misaligned pointer access** — reading a `u32` from an address not divisible by 4
- **Creating invalid references** — `&mut` aliasing, references to uninitialized memory
- **Violating aliasing rules** — two `&mut` to the same data, or `&mut` coexisting with `&`
- **Data races** — unsynchronized concurrent access where at least one is a write
- **Reading uninitialized memory** — `MaybeUninit::assume_init()` on uninitialized data
- **Invalid type values** — a `bool` that is not 0 or 1, an invalid enum discriminant
- **Breaking `unsafe` function preconditions** — e.g., `slice::from_raw_parts` with wrong length

The crucial insight: UB doesn't always crash. It often "works" in debug mode but breaks under optimization, or works on one platform but not another. This is what makes it so dangerous — it's a latent bug that can surface at any time.

### How Miri Works

Miri is an **interpreter for Rust's Mid-level Intermediate Representation (MIR)**. Instead of compiling to machine code and running natively, Miri executes MIR instructions one at a time, tracking metadata that the native CPU doesn't have:

1. **Pointer provenance** — Every pointer carries a tag identifying which allocation it came from. Even if two pointers have the same numeric address, Miri distinguishes them by provenance. This catches use-after-free (pointer to freed allocation) and out-of-bounds (pointer to wrong allocation).

2. **Borrow tracking** — Miri maintains a per-allocation state machine tracking which pointers are allowed to read/write. This enforces the Stacked Borrows or Tree Borrows aliasing model (see below).

3. **Alignment checking** — Every memory access is verified against the type's alignment requirement. By default this uses the numeric address; `-Zmiri-symbolic-alignment-check` uses symbolic tracking to catch cases where alignment was "lucky."

4. **Initialization tracking** — Every byte is tracked as initialized or uninitialized. Reading uninitialized bytes is UB.

5. **Data race detection** — Miri tracks vector clocks per thread and per memory location, detecting unsynchronized concurrent accesses.

6. **Type invariant validation** — On every typed copy, Miri validates that the value is legal for the type (e.g., `bool` must be 0 or 1).

Miri was validated in the POPL 2026 paper ["Miri: Practical Undefined Behavior Detection for Rust"](https://dl.acm.org/doi/10.1145/3776690) (Ralf Jung et al.), which describes it as "the first tool that can find all de-facto Undefined Behavior in deterministic Rust programs." It is integrated into the CI of the Rust standard library and many major crates.

### Stacked Borrows vs Tree Borrows

Rust's type system enforces aliasing rules (no mutable aliasing) for references. But `unsafe` code works with raw pointers (`*mut T`, `*const T`), which have no compile-time aliasing restrictions. The question is: **what aliasing rules must raw pointers follow at runtime?**

**Stacked Borrows** (Ralf Jung, POPL 2020) models each allocation as a **stack** of "borrow items." When you create a reference, it pushes an item onto the stack. Accessing through a pointer is only valid if that pointer's item is on the stack; using a pointer that was "popped off" (invalidated by a newer exclusive borrow) is UB.

Key Stacked Borrows rules:
- Creating `&mut T` pushes `Unique` — invalidates all items above it
- Creating `&T` pushes `SharedReadOnly` — writing through anything above it pops it
- Raw pointers from `&mut` get `Unique` downgraded to `SharedReadWrite`
- Accessing through a pointer not on the stack = UB

**Tree Borrows** (Neven Villani, PLDI 2025) replaces the stack with a **tree** of permissions. Each pointer has a node in the tree, and permissions propagate through parent-child relationships. This is more permissive than Stacked Borrows — it rejects 54% fewer real-world crate tests while maintaining the same optimization guarantees.

Key differences:
- Tree Borrows handles two-phase borrows correctly (Stacked Borrows doesn't)
- Tree Borrows is more permissive with interior mutability patterns
- Both models catch the same fundamental violations (mutable aliasing, use-after-free)
- Miri defaults to Stacked Borrows; use `-Zmiri-tree-borrows` for Tree Borrows

For this practice, we primarily use the default Stacked Borrows model, with one exercise exploring Tree Borrows.

### AddressSanitizer (ASAN)

ASAN is an LLVM compiler instrumentation pass that inserts runtime checks around memory operations. It works by:
1. **Shadow memory** — Maps every 8 bytes of application memory to 1 byte of shadow memory tracking accessibility
2. **Redzone poisoning** — Surrounds allocations with "poisoned" redzones; accessing them triggers an error
3. **Quarantine** — Freed memory is quarantined (not immediately reused), catching use-after-free

ASAN catches: heap/stack/global buffer overflow, use-after-free, use-after-scope, double-free, memory leaks. It runs at ~2x slowdown (much faster than Miri's ~1000x).

In Rust: `RUSTFLAGS="-Zsanitizer=address" cargo run -Zbuild-std --target <target>`. Requires nightly and rebuilding the standard library.

### ThreadSanitizer (TSAN)

TSAN detects data races — unsynchronized concurrent accesses where at least one is a write. It uses a happens-before analysis based on vector clocks (similar to Miri's approach but operating on native code).

In Rust: `RUSTFLAGS="-Zsanitizer=thread" cargo run -Zbuild-std --target <target>`.

### When to Use Each Tool

| Tool | Speed | Catches | Doesn't Catch | Best For |
|------|-------|---------|----------------|----------|
| **Miri** | ~1000x slower | Provenance, aliasing (Stacked/Tree Borrows), alignment, uninitialized reads, type invariants, data races, leaks | FFI code, syscalls, inline asm, platform-specific behavior | Pure Rust `unsafe` code — CI for any crate with `unsafe` |
| **ASAN** | ~2x slower | Buffer overflow, use-after-free, use-after-scope, leaks, C FFI memory errors | Aliasing violations, provenance, uninitialized reads (use MSAN for that) | Mixed Rust/C codebases, FFI boundaries |
| **TSAN** | ~5-15x slower | Data races in native code including C FFI | Everything non-race-related | Concurrent code with FFI |

**Key insight**: Miri and sanitizers are complementary. Miri catches Rust-specific UB (provenance, aliasing) that sanitizers miss. Sanitizers catch C/FFI-side issues that Miri can't execute. A robust CI runs both.

**Limitations of Miri**:
- Cannot execute FFI (`extern "C"` calls to C libraries) — use `cfg(miri)` to skip
- Cannot execute inline assembly
- ~1000x slower than native execution — test with small inputs
- Does not cover all possible execution paths — only the paths actually taken
- Non-deterministic code (random, time-dependent) may need `-Zmiri-seed` or `-Zmiri-many-seeds`
- Platform-specific behavior may differ from native execution

**References**:
- [Miri GitHub repository](https://github.com/rust-lang/miri)
- [Ralf Jung's blog: What's new in Miri (2025)](https://www.ralfj.de/blog/2025/12/22/miri.html)
- [Miri POPL 2026 paper](https://dl.acm.org/doi/10.1145/3776690)
- [Tree Borrows PLDI 2025](https://pldi25.sigplan.org/details/pldi-2025-papers/42/Tree-Borrows)
- [Stacked Borrows paper (POPL 2020)](https://plv.mpi-sws.org/rustbelt/stacked-borrows/)
- [Rust Sanitizers documentation](https://doc.rust-lang.org/beta/unstable-book/compiler-flags/sanitizer.html)
- [Using Miri in CI (GitHub Actions)](https://kflansburg.com/posts/improving-rust-codebase-quality-with-miri-in-ci/)

## Description

This practice teaches you to use Miri and sanitizers to detect undefined behavior in `unsafe` Rust code. You will:

1. Run Miri on safe code to establish a baseline and understand the tooling
2. Diagnose and fix use-after-free and misaligned memory access detected by Miri
3. Understand Stacked Borrows violations through aliasing-rule-breaking examples
4. Detect data races with Miri's concurrency checker
5. Explore Tree Borrows as an alternative aliasing model
6. Write correct `unsafe` code that passes Miri's strict checks
7. Fix a capstone module full of intentional UB bugs

## Instructions

### Setup

1. The `rust-toolchain.toml` pins nightly (required for Miri). Run `rustup component add miri` to install Miri on the nightly toolchain.
2. Verify: `cargo miri --version` should print the Miri version.
3. `cargo check` should pass — all exercises compile. Buggy code lives in test modules that only run under `cargo miri test`.

### Exercise 1: Miri Basics — Safe Code Baseline

**What it teaches**: How to invoke Miri, what its output looks like, and that safe Rust code passes Miri cleanly.

**Why it matters**: Before using Miri to find bugs, you need to know what "clean" looks like. This exercise also introduces `cfg(miri)` for conditional compilation.

- Run `cargo miri test ex1` to execute the Exercise 1 tests under Miri. They should all pass.
- TODO(human): Implement `safe_vector_ops()` — a function doing standard Vec operations (push, index, iterate, collect). This demonstrates that safe Rust passes Miri without issues.
- TODO(human): Implement `cfg_miri_demo()` — use `cfg(miri)` to reduce iteration counts when running under Miri (since Miri is ~1000x slower).

### Exercise 2: Use-After-Free & Dangling Pointers

**What it teaches**: How Miri's provenance tracking catches use-after-free — accessing memory through a pointer whose allocation has been freed.

**Why it matters**: Use-after-free is one of the most common CVEs in C/C++. In Rust, safe code prevents this entirely, but `unsafe` code with raw pointers can recreate it. Miri catches it because it tracks pointer provenance — even if the memory address is reused, the pointer's "tag" is invalidated.

- Run `cargo miri test ex2` — the tests contain INTENTIONALLY BUGGY code that Miri will flag. Read and understand the error messages.
- TODO(human): In `fix_use_after_free()`, fix the buggy version so it passes Miri. The key is to ensure the pointer remains valid for the duration of its use.
- TODO(human): In `fix_misaligned_access()`, fix unaligned pointer casting. Hint: use `ptr::read_unaligned` or ensure proper alignment.

### Exercise 3: Stacked Borrows Violations

**What it teaches**: The Stacked Borrows aliasing model and how Miri enforces it. Creating two `&mut` to the same data via raw pointers is UB even if you "carefully" alternate accesses.

**Why it matters**: Stacked Borrows is the aliasing model that justifies the compiler's alias-based optimizations. Violating it means the compiler may reorder or eliminate your memory accesses based on the assumption that `&mut` is unique. Understanding this is essential for writing sound `unsafe` abstractions.

- Run `cargo miri test ex3` — see Stacked Borrows violations flagged.
- Study the buggy test: two `&mut` pointers to the same data created via raw pointer casting.
- TODO(human): Implement `safe_mutable_split()` — split a slice into two non-overlapping mutable halves using raw pointers, the way `split_at_mut` works internally. This is the canonical example of correct `unsafe` aliasing.
- TODO(human): Implement `interior_mut_pattern()` — use `UnsafeCell` (the correct escape hatch for aliased mutation) to achieve what the buggy example tried to do.

### Exercise 4: Data Race Detection

**What it teaches**: Miri's data race detector using vector clocks. Two threads accessing the same non-atomic memory without synchronization is UB, even if it "works" on x86 (which has a strong memory model).

**Why it matters**: Data races are UB in Rust (and C/C++). They can cause torn reads, stale values, or compiler optimizations that assume single-threaded access. Miri catches them regardless of platform.

- Run `cargo miri test ex4` — see data races flagged.
- TODO(human): Implement `fix_race_with_atomic()` — replace the raw data race with proper `AtomicU64` operations using appropriate memory orderings.
- TODO(human): Implement `fix_race_with_mutex()` — use `Mutex<T>` to synchronize access.

### Exercise 5: Tree Borrows & Aliasing Exploration

**What it teaches**: How Tree Borrows differs from Stacked Borrows. Some code that is UB under Stacked Borrows is valid under Tree Borrows, and vice versa.

**Why it matters**: The Rust aliasing model is still evolving. Tree Borrows (PLDI 2025) is the leading candidate to replace Stacked Borrows. Understanding both helps you write `unsafe` code that is sound under either model.

- Run `cargo miri test ex5` with default flags (Stacked Borrows) — see which tests fail.
- Run `MIRIFLAGS="-Zmiri-tree-borrows" cargo miri test ex5` — see which tests now pass or fail differently.
- TODO(human): Implement `read_through_parent_pointer()` — a pattern that is valid under Tree Borrows but UB under Stacked Borrows (reading from a parent pointer after child borrows).
- TODO(human): Annotate each test function with a comment explaining whether it passes under Stacked Borrows, Tree Borrows, both, or neither, and why.

### Exercise 6: Writing Miri-Clean Unsafe Code

**What it teaches**: How to write `unsafe` code that is correct and passes Miri with maximum strictness (`-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check`).

**Why it matters**: This is the practical skill — not just detecting UB but writing `unsafe` code that avoids it entirely. The patterns here (raw pointer arithmetic, manual Drop, custom collections) are what you encounter when building real-world `unsafe` abstractions.

- TODO(human): Implement `manual_vec_push()` — a simplified `Vec::push` using raw pointer allocation (`alloc::alloc`), reallocation, and `ptr::write`. Must handle alignment, capacity growth, and cleanup.
- TODO(human): Implement `linked_list_node_alloc()` — allocate and link nodes using `Box::into_raw` / `Box::from_raw`, ensuring no leaks and no dangling pointers.
- Run with maximum strictness: `MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" cargo miri test ex6`

### Exercise 7: Capstone — Fix the Bugs

**What it teaches**: Applying everything learned to diagnose and fix a module with 5 intentional UB bugs.

**Why it matters**: Real-world debugging of UB requires reading Miri error messages, understanding the aliasing model, and knowing the correct fix patterns.

- Run `cargo miri test ex7` — Miri will report multiple errors.
- TODO(human): Fix all 5 bugs. Each bug is documented with a comment explaining what kind of UB it triggers. You must understand the root cause and apply the correct fix without changing the function's observable behavior.
- After fixing, `cargo miri test ex7` should pass cleanly.

## Motivation

**Essential CI tool for `unsafe` Rust**: Any crate that uses `unsafe` should run Miri in CI. Major Rust projects (std, crossbeam, rayon, tokio, bytes) all use Miri. It has found dozens of real-world bugs and prevented many more.

**CVE prevention**: Memory safety bugs are the #1 source of security vulnerabilities. Miri catches the Rust-specific subset of these bugs before they ship.

**Professional skill**: Understanding aliasing models (Stacked Borrows, Tree Borrows) is required for writing sound `unsafe` abstractions. This is a distinguishing skill for systems Rust developers.

**Complementary to profiling (063a)**: While 063a focuses on performance, 063b focuses on correctness — both are essential for production-quality Rust.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `rustup component add miri` | Install Miri on the nightly toolchain |
| **Setup** | `cargo miri --version` | Verify Miri installation |
| **Build** | `cargo check` | Verify all exercises compile (no UB runs at check time) |
| **Run all tests** | `cargo miri test` | Run all tests under Miri (Stacked Borrows, default) |
| **Run one exercise** | `cargo miri test ex1` | Run only Exercise 1 tests under Miri |
| **Run one exercise** | `cargo miri test ex2` | Run only Exercise 2 tests (intentional UB — expect errors) |
| **Run one exercise** | `cargo miri test ex3` | Run only Exercise 3 tests (Stacked Borrows violations) |
| **Run one exercise** | `cargo miri test ex4` | Run only Exercise 4 tests (data race detection) |
| **Tree Borrows** | `MIRIFLAGS="-Zmiri-tree-borrows" cargo miri test ex5` | Run Exercise 5 under Tree Borrows model |
| **Strict mode** | `MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" cargo miri test ex6` | Run Exercise 6 with maximum strictness |
| **Capstone** | `cargo miri test ex7` | Run capstone — fix bugs until this passes |
| **Many seeds** | `MIRIFLAGS="-Zmiri-many-seeds=0..64" cargo miri test ex4` | Test concurrency with 64 different thread schedules |
| **Native tests** | `cargo test` | Run tests natively (without Miri) — for comparison |
| **Run binary** | `cargo miri run` | Run main.rs under Miri |
| **Run binary** | `cargo run` | Run main.rs natively |

**Note on Windows**: Sanitizers (ASAN, TSAN) require Linux targets. On Windows, use Miri exclusively. If you have WSL, you can run sanitizers there:

| Phase | Command (Linux/WSL only) | Description |
|-------|--------------------------|-------------|
| **ASAN** | `RUSTFLAGS="-Zsanitizer=address" cargo test -Zbuild-std --target x86_64-unknown-linux-gnu` | Run with AddressSanitizer |
| **TSAN** | `RUSTFLAGS="-Zsanitizer=thread" cargo test -Zbuild-std --target x86_64-unknown-linux-gnu` | Run with ThreadSanitizer |

## Notes

*(To be filled during practice.)*
