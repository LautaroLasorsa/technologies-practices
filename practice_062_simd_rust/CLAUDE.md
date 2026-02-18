# Practice 062: SIMD in Rust — portable_simd & std::arch

## Technologies

- **std::simd** (portable_simd, nightly) — Portable SIMD abstraction: `Simd<T, N>`, masks, reductions, swizzle
- **std::arch** (stable) — Architecture-specific intrinsics: `_mm_add_ps`, `_mm256_fmadd_ps`, `is_x86_feature_detected!`
- **cargo-show-asm** — Inspect generated assembly to verify vectorization

## Stack

- Rust (nightly, via `rust-toolchain.toml`)

## Theoretical Context

### What is SIMD?

**SIMD** (Single Instruction, Multiple Data) is a CPU execution model where one instruction operates on multiple data elements simultaneously. Instead of adding two numbers one pair at a time, a SIMD instruction adds 4, 8, or 16 pairs in a single clock cycle.

Modern CPUs have dedicated **vector registers** — wider than regular registers — that hold multiple values packed side by side. A single SIMD instruction tells the hardware "apply this operation to every element in the register at once."

```
Scalar (one at a time):          SIMD (4 at once):
  a[0] + b[0] = c[0]              [a0, a1, a2, a3]
  a[1] + b[1] = c[1]            + [b0, b1, b2, b3]
  a[2] + b[2] = c[2]            = [c0, c1, c2, c3]  ← one instruction
  a[3] + b[3] = c[3]
  (4 instructions)                 (1 instruction)
```

SIMD is the single biggest performance lever for data-parallel numeric code — delivering 4x to 16x speedups depending on the data type and register width.

### Vector Register Sizes Across x86 ISA Extensions

Each generation of x86 SIMD introduced wider registers and more operations:

| ISA Extension | Register Width | f32 Lanes | f64 Lanes | i32 Lanes | Introduced |
|---------------|----------------|-----------|-----------|-----------|------------|
| **SSE2** | 128-bit (`__m128`) | 4 | 2 | 4 | Pentium 4 (2001) |
| **AVX** | 256-bit (`__m256`) | 8 | 4 | — | Sandy Bridge (2011) |
| **AVX2** | 256-bit (`__m256i`) | 8 | 4 | 8 | Haswell (2013) |
| **AVX-512** | 512-bit (`__m512`) | 16 | 8 | 16 | Skylake-X (2017) |

SSE2 is the baseline — guaranteed on every x86-64 CPU (Rust enables it by default). AVX2 is widely available on modern desktop/server CPUs. AVX-512 has limited desktop support and causes frequency throttling on some Intel CPUs.

ARM has its own SIMD: **NEON** (128-bit, ubiquitous on AArch64) and **SVE/SVE2** (variable-width, server-class).

### Vertical vs Horizontal Operations

SIMD operations come in two flavors:

**Vertical (lane-wise):** Each lane operates independently with the corresponding lane of another vector. These are fast — they map directly to hardware instructions with full throughput.
```
  [a0, a1, a2, a3]
+ [b0, b1, b2, b3]
= [a0+b0, a1+b1, a2+b2, a3+b3]   ← vertical add
```

**Horizontal (cross-lane):** Combine elements within the same vector. These are slower — they require shuffling data between lanes, which costs extra instructions and latency.
```
  [a0, a1, a2, a3]
→ a0 + a1 + a2 + a3 = sum          ← horizontal reduction
```

Rule of thumb: **maximize vertical operations, minimize horizontal ones.** The ideal SIMD algorithm processes entire arrays with vertical ops and only does one horizontal reduction at the very end (e.g., summing an accumulator).

### Auto-Vectorization

The Rust compiler (via LLVM) can automatically convert scalar loops into SIMD instructions — **auto-vectorization**. When it works, you get SIMD performance without writing any explicit SIMD code.

**When auto-vectorization succeeds:**
- Simple loops with predictable bounds and stride-1 access
- No loop-carried dependencies (each iteration is independent)
- Standard arithmetic operations on primitive types
- Separate input/output slices (no aliasing concerns)

**When it fails (common):**
- Loop-carried dependencies (`sum += a[i]` can partially vectorize, but complex chains cannot)
- Complex control flow inside the loop (branches, early exits)
- Function calls that LLVM cannot inline or prove side-effect-free
- Indirect indexing, pointer aliasing, or complex address computation
- Iterator chains that are too complex for the optimizer to see through

**How to check:** Use `cargo-show-asm` to inspect the generated assembly. Look for SIMD mnemonics:
- SSE2: `addps`, `mulps`, `movaps` (packed single-precision, 4 lanes)
- AVX2: `vaddps`, `vmulps`, `vmovaps` (256-bit packed, 8 lanes)
- Scalar fallback: `addss`, `mulss` (single scalar — auto-vectorization failed)

Auto-vectorization is fragile. Small code changes (adding a bound check, changing an iterator adapter) can silently disable it. This is why explicit SIMD exists — to **guarantee** vectorization.

### std::simd (portable_simd) — Portable, Explicit SIMD

`std::simd` is Rust's portable SIMD API (nightly-only, feature gate `portable_simd`). It provides a `Simd<T, N>` type that abstracts over all architectures — the same code compiles to SSE2 on x86, NEON on ARM, and scalar fallback on unsupported targets.

**Core type: `Simd<T, N>`**

A fixed-size vector of `N` elements of type `T`. Supported element types: `f32`, `f64`, `i8`-`i64`, `u8`-`u64`. Lane counts: 1 to 64 (power of two recommended for hardware alignment).

Type aliases for convenience: `f32x4` = `Simd<f32, 4>`, `f32x8` = `Simd<f32, 8>`, etc.

```rust
#![feature(portable_simd)]
use std::simd::f32x4;

let a = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
let b = f32x4::splat(10.0);  // [10.0, 10.0, 10.0, 10.0]
let c = a + b;                // [11.0, 12.0, 13.0, 14.0] — vertical add
```

**Key operations:**

| Category | Methods | Description |
|----------|---------|-------------|
| **Construction** | `splat(v)`, `from_array([...])`, `from_slice(&[T])` | Create vectors |
| **Arithmetic** | `+`, `-`, `*`, `/`, `%` (operator overloads) | Lane-wise arithmetic |
| **Comparison** | `simd_eq`, `simd_lt`, `simd_ge`, etc. | Returns a `Mask` |
| **Reduction** | `reduce_sum()`, `reduce_product()`, `reduce_min()`, `reduce_max()` | Horizontal fold |
| **Bitwise** | `&`, `|`, `^`, `!`, `<<`, `>>` | Lane-wise bitwise ops |
| **Select** | `mask.select(true_vec, false_vec)` | Conditional per-lane |
| **Swizzle** | `simd_swizzle!(v, [indices])` | Rearrange lanes |
| **Shuffle** | `reverse()`, `rotate_elements_left::<N>()`, `interleave()` | Lane permutation |
| **Memory** | `copy_to_slice()`, `load_or_default()`, `gather_or()`, `scatter()` | Load/store |
| **Math** | `simd_abs()`, `simd_min()`, `simd_max()`, `simd_clamp()` | Float/int math |

**Masks:** Comparisons produce `Mask<T, N>` — a per-lane boolean. Use `mask.select(a, b)` for branchless conditional logic: if lane's mask is true, take from `a`; otherwise from `b`. This replaces branches in SIMD code — there is no per-lane branching in SIMD, only masking.

```rust
let values = f32x4::from_array([1.0, -2.0, 3.0, -4.0]);
let zero = f32x4::splat(0.0);
let mask = values.simd_ge(zero);       // [true, false, true, false]
let clamped = mask.select(values, zero); // [1.0, 0.0, 3.0, 0.0]  ← ReLU!
```

### std::arch — Maximum Performance, Architecture-Specific

`std::arch` exposes the raw CPU intrinsics: `_mm_add_ps` (SSE2 128-bit float add), `_mm256_fmadd_ps` (AVX2+FMA 256-bit fused multiply-add), etc. These map 1:1 to machine instructions.

**Advantages over portable_simd:**
- Available on **stable** Rust
- Exact control over which instructions are emitted
- Can use architecture-specific operations that have no portable equivalent
- Performance-critical paths where you need to know exactly what the CPU executes

**Disadvantages:**
- Not portable (x86-only code won't compile on ARM)
- All intrinsics are `unsafe`
- Verbose — must handle feature detection, cfg guards, and type conversions manually

**Usage pattern — runtime dispatch:**

```rust
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2_fma(a, b) };
        }
        if is_x86_feature_detected!("sse2") {
            return unsafe { dot_product_sse2(a, b) };
        }
    }
    dot_product_scalar(a, b)  // fallback
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // ... use _mm256_fmadd_ps, etc.
}
```

The `#[target_feature(enable = "avx2")]` attribute tells the compiler to generate AVX2 instructions for that specific function only, without requiring the whole program to target AVX2. The function must be `unsafe` because calling it on a CPU without AVX2 is undefined behavior. The `is_x86_feature_detected!` macro performs a runtime check (cached after first call).

### Alignment and Memory

SIMD loads/stores are fastest when the data address is aligned to the vector width:
- SSE2 (128-bit): 16-byte alignment
- AVX2 (256-bit): 32-byte alignment

Unaligned loads (`_mm_loadu_ps` / `Simd::from_slice`) work on modern CPUs with minimal penalty, but aligned loads (`_mm_load_ps`) can be slightly faster and are required by some older instructions.

In portable_simd, `Simd::from_slice()` handles unaligned data automatically. For std::arch, use `_mm_loadu_ps` (unaligned) vs `_mm_load_ps` (aligned, UB if misaligned).

### Processing Remainder Elements

When the array length is not a multiple of the SIMD lane count, you must handle the leftover "tail" elements. Common strategies:

1. **Scalar tail:** Process remaining elements one at a time (simplest, fine if N >> lanes)
2. **Masked load:** Use `Simd::load_or_default()` or mask operations to handle partial vectors
3. **Overlap:** Process the last full-width chunk overlapping with the tail (duplicate some work but no branching)

### Common SIMD Patterns

**Accumulator pattern (dot product, sum):**
```
accumulator = zero_vector
for each chunk of N elements:
    accumulator += chunk_a * chunk_b   ← vertical multiply + vertical add
final_sum = accumulator.reduce_sum()   ← one horizontal reduction at the end
```

**Branchless min/max/clamp:**
```
mask = values > threshold
result = mask.select(values, threshold)  ← no branch, no branch misprediction
```

**Structure of Arrays (SoA) vs Array of Structures (AoS):**
SIMD works best on contiguous data of the same type. SoA layout (`[x0,x1,x2,...], [y0,y1,y2,...]`) enables loading 4 x-values at once. AoS layout (`[x0,y0,x1,y1,...]`) requires interleave/deinterleave shuffles to extract lanes. **Prefer SoA for SIMD-heavy code.**

## Description

Build a series of exercises progressing from auto-vectorization observation through portable_simd operations to raw std::arch intrinsics. Each exercise targets a specific SIMD concept: vertical operations, horizontal reductions, masked conditional logic, and runtime feature dispatch. The capstone benchmarks scalar vs auto-vectorized vs explicit SIMD implementations of the same algorithm to quantify real speedups.

### What you'll learn

1. **Auto-vectorization** — Write a loop that LLVM vectorizes, then break it and observe the difference
2. **portable_simd basics** — `Simd<f32, N>`, splat, from_array, arithmetic, reductions
3. **SIMD dot product** — The canonical accumulator pattern with chunked processing and tail handling
4. **Masks and conditional SIMD** — Branchless selection, clamping, filtering with `Mask`
5. **std::arch SSE2/AVX2 intrinsics** — Raw intrinsics with runtime feature detection
6. **Swizzle and data rearrangement** — `simd_swizzle!`, interleave, deinterleave, rotate
7. **Capstone benchmark** — Scalar vs auto-vectorized vs portable_simd vs std::arch, measured with criterion

## Instructions

### Exercise 1: Auto-Vectorization — Observe and Break It (~15 min)

Open `src/ex1_autovec.rs`. This exercise builds intuition for what the compiler can and cannot vectorize automatically.

Auto-vectorization is LLVM's attempt to convert your scalar loops into SIMD instructions without you writing any explicit SIMD. Understanding when it works (and when it silently fails) is the foundation for knowing when you need explicit SIMD.

1. **TODO(human): `sum_array_autovec()`** — Write the simplest possible summation loop over a `&[f32]` slice. This is the "hello world" of auto-vectorization: a reduction with no loop-carried dependency beyond the accumulator, which LLVM can vectorize by unrolling into multiple accumulators.

2. **TODO(human): `add_arrays_autovec()`** — Write an element-wise addition of two slices into an output slice. This is pure vertical operation with no dependencies — the ideal case for auto-vectorization. Use `iter().zip()` or index-based access.

3. **TODO(human): `sum_with_branch_autovec()`** — Write a conditional sum: only add elements greater than a threshold. The branch inside the loop makes auto-vectorization much harder. Compare the assembly output (via cargo-show-asm) with the unconditional sum to see if LLVM managed to use masked operations or fell back to scalar.

### Exercise 2: portable_simd Fundamentals (~15 min)

Open `src/ex2_portable_basics.rs`. This exercise introduces the core `Simd<T, N>` API through simple operations.

The `std::simd` module (feature `portable_simd`) provides a hardware-agnostic SIMD type. `Simd<f32, 4>` on x86-64 compiles to SSE2 operations; the same code on AArch64 compiles to NEON. You write SIMD logic once, and LLVM maps it to the best available instructions per target.

1. **TODO(human): `simd_arithmetic()`** — Create two `f32x4` vectors from arrays, perform addition, subtraction, multiplication, and division. Print the results. This teaches the basic operator overloads on `Simd` — they are vertical (lane-wise) operations, each mapping to a single SIMD instruction.

2. **TODO(human): `simd_reductions()`** — Create a `f32x8` vector and compute `reduce_sum()`, `reduce_product()`, `reduce_min()`, and `reduce_max()`. These are horizontal operations — they combine all lanes into a single scalar. Understand that reductions are the expensive part of SIMD; the goal is to minimize them (ideally one at the end of a pipeline).

3. **TODO(human): `simd_from_slice()`** — Load SIMD vectors from a slice using `Simd::from_slice()`, process them (e.g., multiply by a constant), and write back with `copy_to_slice()`. Handle the case where the slice length is not a multiple of the lane count by processing the remainder with scalar code. This teaches the fundamental SIMD processing loop pattern.

### Exercise 3: SIMD Dot Product (~20 min)

Open `src/ex3_dot_product.rs`. The dot product is the canonical SIMD exercise — it demonstrates the accumulator pattern that is the backbone of matrix multiply, convolution, and correlation.

The key insight: accumulate vertical multiply-add results into a SIMD accumulator vector, then do a single horizontal reduction at the very end. This maximizes vertical (fast) operations and minimizes horizontal (slow) ones.

1. **TODO(human): `dot_product_scalar()`** — Implement a simple scalar dot product as the baseline. `sum(a[i] * b[i])` for all i.

2. **TODO(human): `dot_product_portable_simd()`** — Implement using `std::simd::f32x8`. Process the arrays in chunks of 8, accumulating into a `f32x8` accumulator. After the main loop, handle the tail elements with scalar code, then call `reduce_sum()` on the accumulator. This is the most important SIMD pattern to internalize.

3. **TODO(human): `dot_product_multi_accumulator()`** — Use TWO accumulator vectors instead of one. Process two chunks per iteration, accumulating into `acc0` and `acc1` separately, then combine at the end: `(acc0 + acc1).reduce_sum()`. Multiple accumulators hide instruction latency — while the CPU computes `acc0 += chunk0`, `acc1 += chunk1` can start in parallel (instruction-level parallelism within the SIMD pipeline).

### Exercise 4: Masks and Conditional SIMD (~15 min)

Open `src/ex4_masks.rs`. This exercise teaches branchless conditional logic — the SIMD replacement for `if/else`.

In scalar code, branches (if/else) are handled by the branch predictor. In SIMD, there is no per-lane branch predictor — instead, you compute BOTH outcomes and use a mask to select which result each lane keeps. This is branchless and has predictable, constant-time performance regardless of the data pattern.

1. **TODO(human): `relu_simd()`** — Implement the ReLU activation function: `max(0, x)` for each element. Use `simd_max` or compare + select. ReLU is the most common activation function in neural networks and demonstrates why SIMD is essential for ML inference.

2. **TODO(human): `clamp_simd()`** — Clamp each element to a `[min, max]` range using masks. Compare two approaches: (a) using `simd_clamp()` if available, and (b) using two comparisons + two selects manually. This teaches how compound conditions translate to mask combinations.

3. **TODO(human): `count_threshold_simd()`** — Count how many elements in a slice exceed a threshold. Use comparison to get a mask, then convert the mask to count (via `to_bitmask()` + `count_ones()`, or by selecting 1s and 0s and reducing). This teaches mask-to-scalar conversion patterns.

### Exercise 5: std::arch — SSE2 and AVX2 Intrinsics (~20 min)

Open `src/ex5_stdarch.rs`. This exercise drops to the lowest level: raw CPU intrinsics via `std::arch`. You control exactly which instructions the CPU executes.

std::arch intrinsics are `unsafe` because calling an SSE2 function on a CPU without SSE2 is undefined behavior. The `#[target_feature(enable = "...")]` attribute and `is_x86_feature_detected!` macro provide the safety mechanism: compile the specialized function for a specific ISA, then dispatch at runtime based on CPU capabilities.

1. **TODO(human): `dot_product_sse2()`** — Implement a dot product using SSE2 intrinsics: `_mm_loadu_ps` (load 4 floats), `_mm_mul_ps` (multiply), `_mm_add_ps` (add to accumulator), and a horizontal sum at the end. The function must be `#[target_feature(enable = "sse2")]` and `unsafe`. SSE2 is baseline on x86-64, so this always works.

2. **TODO(human): `dot_product_avx2_fma()`** — Implement using AVX2+FMA: `_mm256_loadu_ps` (load 8 floats), `_mm256_fmadd_ps` (fused multiply-add: `a * b + c` in one instruction with one rounding). FMA is both faster (one instruction instead of two) and more accurate (single rounding instead of double). Requires `#[target_feature(enable = "avx2,fma")]`.

3. **TODO(human): `dispatch_dot_product()`** — Write a dispatch function that uses `is_x86_feature_detected!` to select the best available implementation at runtime. Fall back gracefully: AVX2+FMA -> SSE2 -> scalar. This is the standard pattern for shipping portable high-performance binaries.

### Exercise 6: Swizzle and Data Rearrangement (~10 min)

Open `src/ex6_swizzle.rs`. This exercise teaches lane permutation — rearranging elements within and across SIMD vectors.

Swizzle operations rearrange which data sits in which lane. They are essential for transposing matrices, interleaving/deinterleaving packed data (RGB pixels, complex numbers), and implementing algorithms that need cross-lane communication.

1. **TODO(human): `transpose_2x2_simd()`** — Given two `f32x4` vectors representing rows of a 2x2 matrix stored as `[r0c0, r0c1, _, _]` and `[r1c0, r1c1, _, _]`, use `simd_swizzle!` or `interleave`/`deinterleave` to produce the transposed columns. This teaches the fundamental matrix transpose building block.

2. **TODO(human): `deinterleave_rgb()`** — Given a slice of interleaved RGB data `[R0,G0,B0, R1,G1,B1, ...]`, extract separate R, G, B channels into three separate vectors using SIMD operations. This teaches the AoS-to-SoA conversion that is critical for image processing.

### Exercise 7: Capstone Benchmark — Scalar vs SIMD (~15 min)

Open `benches/simd_benchmark.rs`. This is a criterion benchmark that measures the real-world speedup of each approach.

The purpose is to build empirical intuition: how much does explicit SIMD actually help vs auto-vectorization? When is the overhead of explicit SIMD not worth it? The answers depend on the workload size, the operation, and the target CPU.

1. **TODO(human): Add criterion benchmark groups** that compare:
   - Scalar dot product vs portable_simd dot product vs std::arch SSE2 vs std::arch AVX2+FMA
   - For array sizes: 64, 1024, 16384, 131072 elements
   - Report throughput in elements/second

   This teaches benchmarking methodology and reveals the crossover points where SIMD overhead pays off.

## Motivation

- **4-16x speedups** for data-parallel numerical code — the single biggest performance lever after algorithmic improvements
- **Essential for HPC, ML inference, signal processing, and game engines** — every high-performance library (BLAS, FFTW, PyTorch, TensorFlow) is built on SIMD
- **Complements Practice 019a/b (CUDA)** — SIMD is the CPU counterpart to GPU parallelism; together they cover the full performance stack
- **Bridges C++ SIMD knowledge** — x86 intrinsics are identical across C++ and Rust; `std::arch` uses the same `_mm_*` / `_mm256_*` naming
- **portable_simd is Rust's future** — understanding it now positions you to use it when it stabilizes, and the concepts transfer directly to any SIMD API

## Commands

### Setup

| Command | Description |
|---------|-------------|
| `rustup toolchain install nightly` | Install nightly toolchain (required for `portable_simd` feature) |
| `cargo install cargo-show-asm` | Install assembly viewer (optional, for inspecting vectorization) |

### Build & Run

| Command | Description |
|---------|-------------|
| `cargo build` | Compile all exercises on nightly (verifies TODO stubs compile) |
| `cargo run` | Run the exercise runner — executes all exercises sequentially |
| `cargo run -- 1` | Run only Exercise 1 (auto-vectorization) |
| `cargo run -- 2` | Run only Exercise 2 (portable_simd basics) |
| `cargo run -- 3` | Run only Exercise 3 (SIMD dot product) |
| `cargo run -- 4` | Run only Exercise 4 (masks and conditionals) |
| `cargo run -- 5` | Run only Exercise 5 (std::arch intrinsics) |
| `cargo run -- 6` | Run only Exercise 6 (swizzle and rearrangement) |

### Benchmarks

| Command | Description |
|---------|-------------|
| `cargo bench` | Run all criterion benchmarks (Exercise 7) |
| `cargo bench -- dot_product` | Run only dot product benchmarks |
| `cargo bench -- --sample-size 50` | Run with fewer samples (faster, less precise) |

### Assembly Inspection

| Command | Description |
|---------|-------------|
| `cargo asm --lib "sum_array_autovec"` | View assembly for the auto-vectorizable sum function |
| `cargo asm --lib "sum_with_branch_autovec"` | View assembly for the branch-containing sum (compare with above) |
| `cargo asm --lib "dot_product_portable_simd"` | View assembly for the portable_simd dot product |
| `RUSTFLAGS="-C target-cpu=native" cargo asm --lib "add_arrays_autovec"` | View assembly with native CPU features enabled (may show AVX2) |

### Development

| Command | Description |
|---------|-------------|
| `cargo check` | Fast type-check without codegen (use while implementing) |
| `cargo test` | Run unit tests (each exercise has verification tests) |
| `cargo clippy` | Run linter for idiomatic Rust suggestions |
| `cargo build --release` | Optimized build (use for meaningful performance comparison) |
| `cargo run --release` | Run exercises with optimizations (IMPORTANT: debug builds do not auto-vectorize) |

## References

- [std::simd documentation (nightly)](https://doc.rust-lang.org/std/simd/index.html)
- [Simd struct API](https://doc.rust-lang.org/std/simd/struct.Simd.html)
- [std::arch documentation](https://doc.rust-lang.org/std/arch/index.html)
- [portable-simd GitHub repository](https://github.com/rust-lang/portable-simd)
- [The State of SIMD in Rust (2025)](https://shnatsel.medium.com/the-state-of-simd-in-rust-in-2025-32c263e5f53d)
- [Taking Advantage of Auto-Vectorization in Rust](https://www.nickwilcox.com/blog/autovec/)
- [Nine Rules for SIMD Acceleration of Your Rust Code](https://towardsdatascience.com/nine-rules-for-simd-acceleration-of-your-rust-code-part-1-c16fe639ce21/)
- [cargo-show-asm](https://github.com/pacak/cargo-show-asm)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) — Reference for all x86 SIMD intrinsics

## Notes

*(To be filled during practice.)*
