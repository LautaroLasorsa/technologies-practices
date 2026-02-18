// ============================================================================
// Exercise 5 (Capstone): Tracepoint for process exec → RingBuf
//
// This BPF program is the second half of the capstone dashboard. It attaches
// to sched_process_exec and pushes ActivityEvent (kind=EVENT_EXEC) into the
// SAME RingBuf shared with dashboard_open.
//
// Key concepts:
// - Shared maps across programs: This program declares ACTIVITY_EVENTS with
//   the same name and type as dashboard_open.rs. When loaded by the same
//   Ebpf instance in userspace, they share the same kernel map.
// - bpf_get_current_comm(): Returns the current task's "comm" name — the
//   16-byte executable name (e.g., "bash", "curl", "python3"). This is
//   faster and simpler than parsing the tracepoint's __data_loc filename.
// - Combining program types: Having a kprobe (file opens) and a tracepoint
//   (process execs) write to the same event stream lets userspace correlate
//   events and build a unified view of system activity.
// ============================================================================

#![no_std]
#![no_main]

use aya_ebpf::{
    helpers::{bpf_get_current_comm, bpf_get_current_pid_tgid, bpf_ktime_get_ns},
    macros::{map, tracepoint},
    maps::RingBuf,
    programs::TracePointContext,
};
use practice_065_common::ActivityEvent;

// TODO(human): Declare the shared RingBuf map (same name as in dashboard_open.rs)
//
// IMPORTANT: The map name in the BPF ELF section must match exactly for
// sharing to work. Since both programs use:
//   static ACTIVITY_EVENTS: RingBuf = RingBuf::with_byte_size(256 * 1024, 0);
// with the same name "ACTIVITY_EVENTS", aya's loader will reuse the same
// kernel map for both programs.
//
// Declaration (identical to dashboard_open.rs):
//
//   #[map]
//   static ACTIVITY_EVENTS: RingBuf = RingBuf::with_byte_size(256 * 1024, 0);

// TODO(human): Implement the tracepoint entry point for process exec events
//
//   #[tracepoint]
//   pub fn dashboard_exec(ctx: TracePointContext) -> u32 {
//       match try_dashboard_exec(ctx) {
//           Ok(ret) => ret,
//           Err(ret) => ret,
//       }
//   }
//
//   fn try_dashboard_exec(ctx: TracePointContext) -> Result<u32, u32> {
//       let pid = (bpf_get_current_pid_tgid() >> 32) as u32;
//       let ts = unsafe { bpf_ktime_get_ns() };
//
//       // Reserve space in the ring buffer.
//       let mut entry = match ACTIVITY_EVENTS.reserve::<ActivityEvent>(0) {
//           Some(entry) => entry,
//           None => return Err(0),
//       };
//
//       let event = entry.as_mut_ptr();
//       unsafe {
//           (*event).kind = practice_065_common::EVENT_EXEC;
//           (*event).pid = pid;
//           (*event).timestamp_ns = ts;
//           (*event)._pad = 0;
//
//           // Get the current task's comm name (executable name, max 16 bytes).
//           //
//           // bpf_get_current_comm() returns Result<[u8; 16], _>.
//           // This is the task's "comm" field — a short name truncated to 16 chars.
//           // For example: "bash", "python3", "cat", "systemd".
//           //
//           // It's not the full path (/usr/bin/python3), just the basename.
//           // For the full path, you'd need to parse the tracepoint's __data_loc
//           // filename field, which is more complex.
//           match bpf_get_current_comm() {
//               Ok(comm) => {
//                   // Find the actual length (up to first null byte).
//                   let len = comm.iter().position(|&c| c == 0).unwrap_or(comm.len());
//                   (*event).name[..len].copy_from_slice(&comm[..len]);
//                   (*event).name_len = len as u32;
//               }
//               Err(_) => {
//                   (*event).name_len = 0;
//               }
//           }
//       }
//
//       entry.submit(0);
//       Ok(0)
//   }
//
// KEY LEARNING: By having two BPF programs write to the same RingBuf,
// the userspace consumer sees a unified, time-ordered stream of events
// from different sources. This is how production observability agents
// (Datadog, Falco) build comprehensive system views — they load many
// BPF programs (sometimes 50+) that all feed into shared event streams.
//
// The tag field (kind = EVENT_OPEN vs EVENT_EXEC) lets the consumer
// dispatch events to different handlers without needing separate maps
// or consumers per event type.

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
