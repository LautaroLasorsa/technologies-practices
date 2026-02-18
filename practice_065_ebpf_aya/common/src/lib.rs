// ============================================================================
// Common types shared between BPF (kernel-side) and userspace programs
//
// This crate is #![no_std] because it must compile for bpfel-unknown-none
// (the BPF target has no standard library, no heap, no allocator).
//
// All types here MUST be:
// - #[repr(C)]: Guarantees C-compatible memory layout so the kernel and
//   userspace agree on struct field offsets. Without this, Rust may reorder
//   fields for optimization, causing misaligned reads across the BPF/user boundary.
// - Copy + Clone: BPF maps store values by copying bytes. No heap pointers.
// - Fixed-size: No Vec, String, or dynamically-sized types. Use fixed-size
//   arrays (e.g., [u8; 256]) for variable-length data like filenames.
// ============================================================================

#![no_std]

// ---------------------------------------------------------------------------
// Exercise 3: ExecEvent — sent from tracepoint to userspace via PerfEventArray
// ---------------------------------------------------------------------------

/// Event emitted when a process calls execve (sched_process_exec tracepoint).
///
/// This struct is written by the BPF program into a PerfEventArray and read
/// by the userspace program. Both sides include this crate, so the struct
/// layout is guaranteed to match.
///
/// Field sizes are chosen to match kernel data:
/// - pid: u32 (kernel pid_t is 32-bit)
/// - filename: fixed-size buffer (kernel comm is max 16 bytes, but we use 256
///   to capture the full executable path from the tracepoint data)
///
// TODO(human): Define the ExecEvent struct
//
// Requirements:
// 1. Add #[repr(C)] attribute — this ensures the struct has C memory layout,
//    which is critical because the BPF program (compiled for bpfel-unknown-none)
//    and the userspace program (compiled for x86_64-linux-gnu) must agree on
//    exactly where each field sits in memory. Without #[repr(C)], Rust may
//    reorder fields differently on each target.
//
// 2. Add #[derive(Clone, Copy)] — BPF maps transfer data by copying raw bytes,
//    so the type must implement Copy. No heap allocation is possible in BPF.
//
// 3. Define these fields:
//    - pid: u32            — the process ID that called execve
//    - filename_len: u32   — how many bytes of `filename` are valid
//    - filename: [u8; 256] — the executable path (null-padded fixed buffer)
//
// Why a fixed-size [u8; 256] instead of String?
// BPF programs run in the kernel with only 512 bytes of stack and zero heap.
// All data must be fixed-size. The filename buffer is filled by the BPF program
// using bpf_probe_read_kernel_str_bytes(), which copies up to N bytes from
// kernel memory into the buffer. The userspace side reads filename_len to know
// how many bytes are valid.
//
// Example:
//   #[repr(C)]
//   #[derive(Clone, Copy)]
//   pub struct ExecEvent {
//       pub pid: u32,
//       pub filename_len: u32,
//       pub filename: [u8; 256],
//   }

// ---------------------------------------------------------------------------
// Exercise 5 (Capstone): ActivityEvent — unified event for the dashboard
// ---------------------------------------------------------------------------

/// Tag byte for discriminating event types in the RingBuf.
///
/// Since BPF programs cannot use Rust enums with data (no heap, no vtable),
/// we use a manual discriminant: a tag byte at the start of the struct
/// followed by a union-like payload.
///
/// In practice, eBPF observability tools use one of two patterns:
/// 1. Separate maps per event type (simpler, but more map management)
/// 2. Tagged union in a shared map (one RingBuf, discriminant field)
///
/// We use pattern 2 here to practice the tagged-union approach.

// TODO(human): Define the ActivityEvent struct for the capstone dashboard
//
// This struct is sent from BOTH the kprobe (file opens) and the tracepoint
// (process execs) into a single shared RingBuf. The userspace consumer reads
// events and dispatches based on the `kind` tag.
//
// Requirements:
// 1. Add #[repr(C)] and #[derive(Clone, Copy)]
//
// 2. Define an event kind discriminant. You can use plain constants:
//      pub const EVENT_OPEN: u32 = 1;
//      pub const EVENT_EXEC: u32 = 2;
//
// 3. Define the ActivityEvent struct with these fields:
//    - kind: u32         — EVENT_OPEN or EVENT_EXEC
//    - pid: u32          — process ID
//    - timestamp_ns: u64 — event timestamp from bpf_ktime_get_ns()
//    - name_len: u32     — valid bytes in `name`
//    - _pad: u32         — padding for 8-byte alignment (BPF requires aligned access)
//    - name: [u8; 256]   — filename (for OPEN) or executable path (for EXEC)
//
// Why include a timestamp?
// bpf_ktime_get_ns() returns a monotonic nanosecond clock from inside the kernel.
// This lets userspace compute inter-event latencies and display relative timestamps.
// It's the same clock used by perf and ftrace.
//
// Why manual padding?
// The BPF verifier and some architectures require naturally-aligned field access.
// A u64 field must start at an 8-byte-aligned offset. If preceding fields total
// an odd number of 4-byte words, we need explicit padding. #[repr(C)] respects
// C alignment rules, but being explicit about padding makes the layout obvious.
//
// Example:
//   pub const EVENT_OPEN: u32 = 1;
//   pub const EVENT_EXEC: u32 = 2;
//
//   #[repr(C)]
//   #[derive(Clone, Copy)]
//   pub struct ActivityEvent {
//       pub kind: u32,
//       pub pid: u32,
//       pub timestamp_ns: u64,
//       pub name_len: u32,
//       pub _pad: u32,
//       pub name: [u8; 256],
//   }

// ---------------------------------------------------------------------------
// Utility: Safe conversion from byte buffer to str (for userspace display)
// ---------------------------------------------------------------------------

/// Convert a fixed-size byte buffer to a &str, using `len` valid bytes.
/// Returns "<invalid utf8>" if the bytes aren't valid UTF-8.
///
/// This is used by userspace binaries to display filenames/paths from events.
/// BPF programs store raw kernel bytes which are almost always valid ASCII/UTF-8
/// paths, but we handle the error case gracefully.
#[cfg(feature = "user")]
pub fn bytes_to_str(buf: &[u8], len: usize) -> &str {
    let len = len.min(buf.len());
    core::str::from_utf8(&buf[..len]).unwrap_or("<invalid utf8>")
}
