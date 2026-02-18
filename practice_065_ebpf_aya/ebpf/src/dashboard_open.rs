// ============================================================================
// Exercise 5 (Capstone): Kprobe for file opens → RingBuf
//
// This BPF program is one half of the capstone dashboard. It attaches to
// sys_openat and pushes ActivityEvent (kind=EVENT_OPEN) into a shared RingBuf.
//
// Key concepts:
// - RingBuf (Linux 5.8+): A single shared ring buffer for all CPUs.
//   Unlike PerfEventArray (per-CPU buffers), RingBuf provides:
//   * Strong ordering: events from all CPUs are ordered by commit time
//   * No per-CPU waste: one buffer shared efficiently
//   * Precise notifications: userspace is woken exactly when data is available
//   * No sampling: every event is delivered (if buffer has space)
//
// - Multiple programs sharing a map: Both dashboard_open (this file) and
//   dashboard_exec (the tracepoint) write to the SAME RingBuf. This works
//   because BPF maps are identified by name — when the userspace loader
//   loads both programs, they reference the same map by name.
//
// - bpf_probe_read_user_str_bytes: sys_openat's second argument is a
//   userspace pointer to the filename. We must use bpf_probe_read_user
//   (not bpf_probe_read_kernel) because the pointer is in userspace memory.
//
// RingBuf API (BPF side):
//   1. ringbuf.reserve::<T>(0) — Reserve sizeof::<T>() bytes in the ring buffer
//      Returns Option<RingBufEntry<T>> — None if the buffer is full
//   2. Write data into the reserved entry
//   3. entry.submit(0) — Commit the entry and notify userspace
//   OR entry.discard(0) — Discard the reservation without committing
//
// This reserve-then-submit pattern avoids copying: you write directly into
// the ring buffer memory, then atomically make it visible to userspace.
// ============================================================================

#![no_std]
#![no_main]

use aya_ebpf::{
    helpers::{bpf_get_current_pid_tgid, bpf_ktime_get_ns, bpf_probe_read_user_str_bytes},
    macros::{kprobe, map},
    maps::RingBuf,
    programs::ProbeContext,
};
use practice_065_common::ActivityEvent;

// TODO(human): Declare a RingBuf map for streaming events to userspace
//
// RingBuf is declared similarly to other maps, but with a byte capacity
// instead of entry count:
//
//   #[map]
//   static ACTIVITY_EVENTS: RingBuf = RingBuf::with_byte_size(256 * 1024, 0);
//
// with_byte_size(256 * 1024, 0):
//   - 256 KB: Total buffer size in bytes. Must be a power of 2 (kernel
//     requirement for ring buffer internals). 256 KB = enough for ~900
//     ActivityEvent structs (each ~280 bytes). Adjust based on expected
//     event rate.
//   - 0: Flags (default).
//
// WHY byte_size instead of max_entries?
// Unlike HashMap/Array (which have discrete entries), RingBuf is a raw byte
// buffer. Each .reserve::<T>() allocates sizeof::<T>() + header bytes from
// this pool. The buffer wraps around when it reaches the end.
//
// NOTE: This map has the SAME name as the one in dashboard_exec.rs. When
// both programs are loaded by the same userspace Ebpf instance, they will
// share the same physical map. The loader matches maps by section name.
// To make this work, both BPF programs must declare the map with the same
// name and type.

// TODO(human): Implement the kprobe entry point for file open events
//
//   #[kprobe]
//   pub fn dashboard_open(ctx: ProbeContext) -> u32 {
//       match try_dashboard_open(ctx) {
//           Ok(ret) => ret,
//           Err(ret) => ret,
//       }
//   }
//
//   fn try_dashboard_open(ctx: ProbeContext) -> Result<u32, u32> {
//       let pid = (bpf_get_current_pid_tgid() >> 32) as u32;
//       let ts = unsafe { bpf_ktime_get_ns() };
//
//       // Reserve space in the ring buffer for one ActivityEvent.
//       //
//       // .reserve::<ActivityEvent>(0) attempts to allocate sizeof(ActivityEvent)
//       // bytes from the ring buffer. Returns None if the buffer is full.
//       //
//       // The returned RingBufEntry<ActivityEvent> is a smart pointer into
//       // the ring buffer memory. You write directly into it — no intermediate
//       // stack copy needed (important given the 512-byte stack limit).
//       let mut entry = match ACTIVITY_EVENTS.reserve::<ActivityEvent>(0) {
//           Some(entry) => entry,
//           None => return Err(0), // Buffer full, drop event
//       };
//
//       let event = entry.as_mut_ptr();
//       unsafe {
//           // Write the event header fields.
//           (*event).kind = practice_065_common::EVENT_OPEN;
//           (*event).pid = pid;
//           (*event).timestamp_ns = ts;
//           (*event)._pad = 0;
//
//           // Read the filename from sys_openat's second argument.
//           //
//           // ctx.arg::<usize>(1) gets the second argument to sys_openat,
//           // which is a const char __user *filename pointer.
//           //
//           // bpf_probe_read_user_str_bytes() copies the string from userspace
//           // memory into our buffer. It returns a Result<&[u8], _> with the
//           // actual bytes read (including null terminator).
//           //
//           // WHY bpf_probe_read_user and not just reading the pointer directly?
//           // The filename pointer is in USERSPACE memory. BPF programs run in
//           // kernel context and cannot directly dereference userspace pointers.
//           // bpf_probe_read_user handles the page fault / copy safely.
//           let filename_ptr: usize = ctx.arg(1).ok_or(1u32)?;
//           match bpf_probe_read_user_str_bytes(
//               filename_ptr as *const u8,
//               &mut (*event).name,
//           ) {
//               Ok(name_bytes) => {
//                   (*event).name_len = name_bytes.len() as u32;
//               }
//               Err(_) => {
//                   (*event).name_len = 0;
//               }
//           }
//       }
//
//       // Submit the entry to make it visible to userspace.
//       // After submit(), the entry is committed to the ring buffer and
//       // userspace will be notified (via epoll/poll) that data is available.
//       entry.submit(0);
//
//       Ok(0)
//   }
//
// KEY LEARNING: The reserve → write → submit pattern of RingBuf is zero-copy.
// You write directly into the ring buffer memory. This is more efficient than
// PerfEventArray's output() which copies the entire struct from the BPF stack
// into the buffer. With large event structs (like our 280-byte ActivityEvent),
// this matters because the BPF stack is only 512 bytes.

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
