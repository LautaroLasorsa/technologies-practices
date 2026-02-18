//! Exercise 1: Building a Tracking Allocator
//!
//! The `GlobalAlloc` trait is Rust's abstraction over heap allocation. Every call to
//! `Box::new()`, `Vec::push()`, `String::from()`, and `HashMap::insert()` ultimately
//! routes through the global allocator's `alloc()` and `dealloc()` methods.
//!
//! By implementing a wrapper around `System` (the default allocator), we can intercept
//! every allocation and count them. This is the same technique used by profiling tools
//! like `dhat` and `heaptrack` — but here you build it from scratch.
//!
//! **Why atomic counters?** The global allocator is called from ALL threads simultaneously.
//! Using `Mutex` for the counters would be catastrophic — it would serialize all allocations
//! (only one thread can allocate at a time), and worse, `Mutex` itself may allocate internally,
//! causing infinite recursion. Atomic operations (`fetch_add`, `load`) are lock-free and
//! never allocate, making them safe to use inside an allocator.

#[allow(unused_imports)]
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Tracking allocator — global state
// ---------------------------------------------------------------------------

/// Atomic counters for tracking allocations. These are `static` because the global
/// allocator is a `static` and can only reference other statics.
///
/// We use `Ordering::Relaxed` for all counter operations because:
/// 1. We only need atomicity (no torn reads/writes), not ordering guarantees.
/// 2. The counters are independent — no "happens-before" relationship needed between
///    ALLOC_COUNT and ALLOC_BYTES. Each is a simple monotonic counter.
/// 3. Relaxed is the cheapest ordering — on x86 it compiles to a plain `lock xadd`
///    without any memory fence instructions.
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

/// A tracking allocator that wraps `System` and counts all allocations/deallocations.
///
/// This struct has no fields — all state lives in the static atomics above.
/// `GlobalAlloc` requires the allocator to be a `static`, and statics cannot contain
/// non-const-constructible fields. Atomics with `AtomicU64::new(0)` are const, so
/// they work as separate statics.
pub struct TrackingAllocator;

// TODO(human): Implement `GlobalAlloc` for `TrackingAllocator`.
//
// The `GlobalAlloc` trait has two required methods and two optional (provided) methods:
//
// Required:
//   unsafe fn alloc(&self, layout: Layout) -> *mut u8
//   unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout)
//
// Optional (have default implementations but you should override for accurate tracking):
//   unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8
//   unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8
//
// For each method:
//
// 1. `alloc`: Increment ALLOC_COUNT by 1 and ALLOC_BYTES by layout.size().
//    Then delegate to `System.alloc(layout)` and return the pointer.
//    layout.size() is the number of bytes requested. layout.align() is the required
//    alignment (e.g., 8 for u64). The System allocator handles alignment internally.
//
//    IMPORTANT: Do NOT panic or allocate on failure — return null. The caller (Box, Vec)
//    handles null by calling the out-of-memory handler. Panicking in an allocator is UB.
//
// 2. `dealloc`: Increment DEALLOC_COUNT by 1 and DEALLOC_BYTES by layout.size().
//    Then delegate to `System.dealloc(ptr, layout)`.
//    The layout MUST match the one used for the original alloc — this is the caller's
//    responsibility (and Rust's standard library always gets this right).
//
// 3. `alloc_zeroed`: Same as alloc but the memory must be zeroed. Increment the same
//    counters as alloc, then delegate to `System.alloc_zeroed(layout)`.
//    This is called by `vec![0u8; n]` and `Box::new([0; N])` — the allocator can
//    sometimes get zeroed pages from the OS for free (via mmap), which is faster
//    than alloc + memset.
//
// 4. `realloc`: This is the tricky one. When Vec grows, it calls realloc to resize
//    the buffer. The old allocation (layout.size() bytes) is being replaced by a new
//    one (new_size bytes). You should:
//    - Increment DEALLOC_BYTES by layout.size() (old allocation freed)
//    - Increment ALLOC_BYTES by new_size (new allocation created)
//    - Increment both ALLOC_COUNT and DEALLOC_COUNT by 1 (one alloc + one dealloc)
//    - Delegate to `System.realloc(ptr, layout, new_size)`
//    Note: The system's realloc may extend the existing block in-place (no copy needed)
//    or allocate a new block and copy. Either way, from our tracking perspective,
//    the old size is gone and the new size exists.
//
// Remember: `unsafe impl` is needed because GlobalAlloc is an unsafe trait —
// the implementor promises the alloc/dealloc contract is upheld.
//
// Example skeleton:
//
//   unsafe impl GlobalAlloc for TrackingAllocator {
//       unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
//           // track...
//           // delegate to System
//       }
//       unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
//           // track...
//           // delegate to System
//       }
//       // ... alloc_zeroed, realloc ...
//   }
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        todo!("Exercise 1: Implement alloc — increment ALLOC_COUNT and ALLOC_BYTES, delegate to System")
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        todo!("Exercise 1: Implement dealloc — increment DEALLOC_COUNT and DEALLOC_BYTES, delegate to System")
    }

    unsafe fn alloc_zeroed(&self, _layout: Layout) -> *mut u8 {
        todo!("Exercise 1: Implement alloc_zeroed — same counters as alloc, delegate to System.alloc_zeroed")
    }

    unsafe fn realloc(&self, _ptr: *mut u8, _layout: Layout, _new_size: usize) -> *mut u8 {
        todo!("Exercise 1: Implement realloc — track old dealloc + new alloc bytes, delegate to System.realloc")
    }
}

/// Register the tracking allocator as the global allocator for this binary.
///
/// IMPORTANT: There can be only ONE `#[global_allocator]` per binary. This attribute
/// replaces the default `System` allocator for ALL allocations — including those made
/// by the standard library itself, third-party crates, and even `println!()` (which
/// allocates a `String` buffer internally).
///
/// The allocator is initialized at program startup, before `main()` runs. The static
/// atomics start at 0, so counter values in main() reflect only allocations made during
/// program execution (not allocator initialization).
#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

// ---------------------------------------------------------------------------
// Snapshot helper
// ---------------------------------------------------------------------------

/// A snapshot of allocation counters at a point in time.
/// Used to measure allocations within a specific code region by taking
/// a "before" snapshot and a "after" snapshot, then computing the delta.
#[derive(Debug, Clone, Copy)]
pub struct AllocSnapshot {
    pub alloc_count: u64,
    pub dealloc_count: u64,
    pub alloc_bytes: u64,
    pub dealloc_bytes: u64,
}

impl AllocSnapshot {
    /// Capture the current counter values.
    pub fn now() -> Self {
        Self {
            alloc_count: ALLOC_COUNT.load(Ordering::Relaxed),
            dealloc_count: DEALLOC_COUNT.load(Ordering::Relaxed),
            alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
            dealloc_bytes: DEALLOC_BYTES.load(Ordering::Relaxed),
        }
    }

    /// Compute the delta between two snapshots.
    /// `self` is the "before" snapshot, `after` is the "after" snapshot.
    pub fn delta_since(&self, after: &AllocSnapshot) -> AllocSnapshot {
        AllocSnapshot {
            alloc_count: after.alloc_count.saturating_sub(self.alloc_count),
            dealloc_count: after.dealloc_count.saturating_sub(self.dealloc_count),
            alloc_bytes: after.alloc_bytes.saturating_sub(self.alloc_bytes),
            dealloc_bytes: after.dealloc_bytes.saturating_sub(self.dealloc_bytes),
        }
    }

    /// Print a human-readable summary.
    pub fn print(&self, label: &str) {
        println!("  [{}] allocs: {}, deallocs: {}, bytes allocated: {}, bytes freed: {}, net: {}",
            label,
            self.alloc_count,
            self.dealloc_count,
            self.alloc_bytes,
            self.dealloc_bytes,
            self.alloc_bytes as i64 - self.dealloc_bytes as i64,
        );
    }
}

// ---------------------------------------------------------------------------
// Exercise functions
// ---------------------------------------------------------------------------

/// Demonstrate that the tracking allocator is active by showing counter values
/// for simple operations.
pub fn demo_tracking_allocator() {
    let before = AllocSnapshot::now();

    // A simple Box allocation: exactly 1 alloc of size_of::<i32>() = 4 bytes.
    let _boxed = Box::new(42i32);

    let after = AllocSnapshot::now();
    let delta = before.delta_since(&after);
    delta.print("Box::new(42i32)");

    // When _boxed goes out of scope here, it will be deallocated.
    // But since we already captured `after`, the dealloc won't show in delta.
    // This is intentional — it demonstrates the snapshot measurement pattern.
}

/// Measure how many allocations `Vec` makes as it grows.
///
/// `Vec` uses a growth strategy: when capacity is exceeded, it allocates a new buffer
/// with roughly double the capacity, copies elements, and frees the old buffer.
/// This means pushing N elements into an empty Vec causes O(log N) allocations, not N.
///
/// Contrast with `Vec::with_capacity(n)`: pre-allocating avoids ALL growth reallocations,
/// resulting in exactly 1 allocation regardless of how many elements you push (up to n).
pub fn measure_vec_allocations() {
    // TODO(human): Measure Vec allocation behavior.
    //
    // Understanding Vec's growth strategy is essential for writing allocation-efficient Rust.
    // Vec starts with capacity 0 (no allocation). The first push allocates a small buffer
    // (usually capacity 4 for small types). Each subsequent growth roughly doubles capacity:
    // 4 → 8 → 16 → 32 → 64 → 128 → ...
    //
    // Each growth is a realloc (or alloc + copy + dealloc), and each abandoned buffer is freed.
    // So pushing 1000 elements into an empty Vec makes ~10 allocations (log2(1000) ≈ 10),
    // not 1000.
    //
    // Steps:
    //
    // 1. Take a snapshot with `AllocSnapshot::now()`.
    // 2. Create an empty `Vec::<u64>::new()`.
    // 3. Push 1000 elements into it in a loop.
    // 4. Take another snapshot and compute the delta.
    // 5. Print the delta with label "Vec::new() + 1000 pushes".
    //    You should see ~10-12 allocations (growth doublings) and the total bytes
    //    will be roughly the sum of capacities: 8*4 + 8*8 + 8*16 + ... + 8*1024.
    //
    // 6. Now repeat with `Vec::<u64>::with_capacity(1000)`:
    //    - Snapshot before, create vec with capacity, push 1000 elements, snapshot after.
    //    - Print delta with label "Vec::with_capacity(1000) + 1000 pushes".
    //    - You should see exactly 1 allocation of 8000 bytes (1000 * size_of::<u64>()).
    //
    // 7. Print a comparison summary showing the difference.
    //    This demonstrates why `with_capacity` matters in hot paths:
    //    - 1 allocation vs ~10
    //    - No wasted intermediate buffers
    //    - No copying of data during growth
    //
    // The lesson: In performance-critical code, ALWAYS pre-allocate when you know the
    // size. `collect::<Vec<_>>()` on an `ExactSizeIterator` does this automatically.

    todo!("Exercise 1b: Measure Vec allocation behavior with and without pre-allocation")
}

/// Measure allocations from string operations.
///
/// String formatting is one of the most allocation-heavy operations in Rust:
/// - `format!("{}", x)` allocates a new String every time
/// - `String::push_str()` may reallocate (same growth strategy as Vec<u8>)
/// - `to_string()` on &str allocates a new String
pub fn measure_string_allocations() {
    // TODO(human): Measure String allocation behavior.
    //
    // Strings in Rust are `Vec<u8>` under the hood (with UTF-8 validation). They follow
    // the same growth strategy: push_str() may trigger a realloc if the capacity is
    // exceeded. But unlike Vec where you often know the final size, strings are frequently
    // built dynamically — making them a major source of hidden allocations.
    //
    // Steps:
    //
    // 1. Measure `format!()` allocations:
    //    - Snapshot before.
    //    - Create 100 strings with `format!("item_{}", i)` in a loop, collecting into a Vec.
    //    - Snapshot after and print delta with label "100x format!()".
    //    - Each `format!()` allocates a new String: expect ~100 allocs for the strings
    //      plus ~7 allocs for Vec growth (to store the 100 Strings).
    //
    // 2. Measure `String::push_str()` growth:
    //    - Snapshot before.
    //    - Create an empty `String::new()`.
    //    - Push 100 short strings into it with `push_str("hello world ")`.
    //    - Snapshot after and print delta with label "String::new() + 100 push_str".
    //    - String grows like Vec: expect ~7-8 allocs (doublings from initial capacity).
    //
    // 3. Measure `String::with_capacity()`:
    //    - Snapshot before.
    //    - Create `String::with_capacity(1200)` (100 * 12 bytes per "hello world ").
    //    - Push 100 short strings into it.
    //    - Snapshot after and print delta with label "String::with_capacity(1200) + 100 push_str".
    //    - Expect exactly 1 allocation.
    //
    // 4. Print a summary comparing all three approaches.
    //    Key insight: `format!()` is convenient but allocates on every call.
    //    In hot loops, prefer `write!()` into a pre-allocated String, or use
    //    `String::with_capacity()` + `push_str()`.

    todo!("Exercise 1c: Measure String allocation behavior across different patterns")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracking_allocator_counts() {
        let before = AllocSnapshot::now();
        let _v: Vec<u8> = Vec::with_capacity(100);
        let after = AllocSnapshot::now();
        let delta = before.delta_since(&after);
        // with_capacity should produce exactly 1 allocation
        assert!(delta.alloc_count >= 1, "Expected at least 1 allocation");
        assert!(delta.alloc_bytes >= 100, "Expected at least 100 bytes allocated");
    }

    #[test]
    fn test_demo_tracking_allocator() {
        demo_tracking_allocator();
    }

    #[test]
    fn test_measure_vec_allocations() {
        measure_vec_allocations();
    }

    #[test]
    fn test_measure_string_allocations() {
        measure_string_allocations();
    }
}
