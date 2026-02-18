// ============================================================================
// Exercise 5 (Capstone): Process Activity Dashboard
//
// This binary loads TWO BPF programs and reads events from a shared RingBuf:
// 1. dashboard_open (kprobe on sys_openat) — emits EVENT_OPEN events
// 2. dashboard_exec (tracepoint on sched_process_exec) — emits EVENT_EXEC events
//
// Both programs write ActivityEvent structs to the same "ACTIVITY_EVENTS" RingBuf.
// This binary consumes the events and displays a unified activity dashboard.
//
// Key concepts:
// - Loading multiple BPF programs: You can load multiple BPF ELF objects
//   into the same Ebpf instance, or load them from separate Ebpf instances
//   and share maps by pinning. Here we load them separately (simpler).
// - RingBuf reading (userspace): aya::maps::RingBuf provides a poll-based
//   interface. You call ring_buf.next() to get the next event, or use
//   async with tokio to wait for events.
// - Event dispatching: The ActivityEvent has a `kind` field that tells us
//   whether it's a file open or a process exec. We match on it.
// - Map sharing between programs: When two BPF programs loaded by different
//   Ebpf instances need to share a map, one option is BPF filesystem pinning
//   (/sys/fs/bpf/). For simplicity here, we load the dashboard_exec program
//   from the same Ebpf object that has the map, so sharing is automatic.
//
// Architecture:
//   [kernel] dashboard_open (kprobe) ──┐
//                                       ├──→ ACTIVITY_EVENTS (RingBuf) ──→ [userspace] dashboard display
//   [kernel] dashboard_exec (tp)    ───┘
// ============================================================================

use std::time::{Duration, Instant};

use anyhow::Context;
use aya::maps::RingBuf;
use aya::programs::{KProbe, TracePoint};
use aya_log::EbpfLogger;
use log::{info, warn};
use practice_065_common::{bytes_to_str, ActivityEvent, EVENT_EXEC, EVENT_OPEN};
use tokio::signal;
use tokio::io::unix::AsyncFd;

const BPF_OPEN: &[u8] = aya::include_bytes_aligned!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/bpfel-unknown-none/release/dashboard-open"
));

const BPF_EXEC: &[u8] = aya::include_bytes_aligned!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/bpfel-unknown-none/release/dashboard-exec"
));

// TODO(human): Implement the main function for the capstone dashboard
//
//   #[tokio::main]
//   async fn main() -> anyhow::Result<()> {
//       env_logger::init();
//
//       // --- Step 1: Load the first BPF object (kprobe for file opens) ---
//       //
//       // This creates the ACTIVITY_EVENTS RingBuf map.
//       let mut ebpf_open = aya::Ebpf::load(BPF_OPEN)
//           .context("Failed to load dashboard-open BPF object")?;
//
//       if let Err(e) = EbpfLogger::init(&mut ebpf_open) {
//           warn!("Failed to init eBPF logger for open: {}", e);
//       }
//
//       // Load and attach the kprobe.
//       let program: &mut KProbe = ebpf_open
//           .program_mut("dashboard_open")
//           .context("Program 'dashboard_open' not found")?
//           .try_into()?;
//       program.load()?;
//       program.attach("sys_openat", 0)
//           .context("Failed to attach dashboard_open kprobe")?;
//       info!("dashboard_open kprobe attached to sys_openat");
//
//       // --- Step 2: Load the second BPF object (tracepoint for execs) ---
//       //
//       // IMPORTANT: This creates a SEPARATE ACTIVITY_EVENTS map. The two
//       // programs do NOT automatically share the map because they're loaded
//       // from different Ebpf instances.
//       //
//       // For true map sharing, you would need to either:
//       // a) Pin the map to /sys/fs/bpf/ and reuse it (production approach)
//       // b) Compile both programs into a single BPF ELF (simpler)
//       // c) Use aya's map reuse API
//       //
//       // For this exercise, we read from BOTH RingBufs separately.
//       // This demonstrates the concept even without perfect sharing.
//       let mut ebpf_exec = aya::Ebpf::load(BPF_EXEC)
//           .context("Failed to load dashboard-exec BPF object")?;
//
//       if let Err(e) = EbpfLogger::init(&mut ebpf_exec) {
//           warn!("Failed to init eBPF logger for exec: {}", e);
//       }
//
//       let program: &mut TracePoint = ebpf_exec
//           .program_mut("dashboard_exec")
//           .context("Program 'dashboard_exec' not found")?
//           .try_into()?;
//       program.load()?;
//       program.attach("sched", "sched_process_exec")
//           .context("Failed to attach dashboard_exec tracepoint")?;
//       info!("dashboard_exec tracepoint attached to sched/sched_process_exec");
//
//       // --- Step 3: Take the RingBuf maps from both Ebpf instances ---
//       //
//       // RingBuf is consumed (take_map) because we need ownership to poll it.
//       let ring_open = ebpf_open
//           .take_map("ACTIVITY_EVENTS")
//           .context("Map 'ACTIVITY_EVENTS' not found in open")?;
//       let mut ring_open = RingBuf::try_from(ring_open)?;
//
//       let ring_exec = ebpf_exec
//           .take_map("ACTIVITY_EVENTS")
//           .context("Map 'ACTIVITY_EVENTS' not found in exec")?;
//       let mut ring_exec = RingBuf::try_from(ring_exec)?;
//
//       info!("Dashboard running. Watching file opens and process execs...");
//       info!("Generate activity in another terminal.");
//       println!();
//       println!("{:<6} {:<8} {:<8} {}", "TYPE", "PID", "TIME(ms)", "NAME");
//       println!("{}", "-".repeat(60));
//
//       let start = Instant::now();
//
//       // --- Step 4: Poll both RingBufs in a loop ---
//       //
//       // We poll both RingBufs in a simple loop with a small sleep.
//       // A more production-ready approach would use epoll/AsyncFd to
//       // wake up only when data is available.
//       //
//       // ring_buf.next() returns Option<RingBufItem>:
//       // - Some(item): An event is available. item.as_ref() gives &[u8].
//       // - None: No events pending.
//       //
//       // We parse the raw bytes as ActivityEvent by casting the pointer.
//
//       let event_loop = tokio::spawn(async move {
//           loop {
//               // Poll the "open" RingBuf.
//               while let Some(item) = ring_open.next() {
//                   let data = item.as_ref();
//                   if data.len() >= std::mem::size_of::<ActivityEvent>() {
//                       let event = unsafe {
//                           (data.as_ptr() as *const ActivityEvent).read_unaligned()
//                       };
//                       print_event(&event, &start);
//                   }
//               }
//
//               // Poll the "exec" RingBuf.
//               while let Some(item) = ring_exec.next() {
//                   let data = item.as_ref();
//                   if data.len() >= std::mem::size_of::<ActivityEvent>() {
//                       let event = unsafe {
//                           (data.as_ptr() as *const ActivityEvent).read_unaligned()
//                       };
//                       print_event(&event, &start);
//                   }
//               }
//
//               // Small sleep to avoid busy-looping.
//               // In production, use epoll/AsyncFd instead.
//               tokio::time::sleep(Duration::from_millis(100)).await;
//           }
//       });
//
//       signal::ctrl_c().await?;
//       event_loop.abort();
//       info!("\nDashboard stopped.");
//
//       Ok(())
//   }
//
// TODO(human): Implement the print_event display function
//
//   fn print_event(event: &ActivityEvent, start: &Instant) {
//       let elapsed = start.elapsed().as_millis();
//       let name = bytes_to_str(&event.name, event.name_len as usize);
//
//       let kind_str = match event.kind {
//           EVENT_OPEN => "OPEN",
//           EVENT_EXEC => "EXEC",
//           _ => "???",
//       };
//
//       // Display format:
//       // TYPE    PID      TIME(ms)  NAME
//       // OPEN       1234     1523  /etc/hostname
//       // EXEC       1235     1524  bash
//       // OPEN       1235     1525  /lib/x86_64-linux-gnu/libc.so.6
//       println!("{:<6} {:>8} {:>8}  {}", kind_str, event.pid, elapsed, name);
//   }
//
// WHAT TO OBSERVE:
// The dashboard shows a unified stream of system activity:
//   TYPE    PID      TIME(ms)  NAME
//   EXEC       1234      100  bash
//   OPEN       1234      101  /etc/bash.bashrc
//   OPEN       1234      102  /lib/x86_64-linux-gnu/libtinfo.so.6
//   EXEC       1235      200  ls
//   OPEN       1235      201  /lib/x86_64-linux-gnu/libc.so.6
//   OPEN       1235      202  /etc/nsswitch.conf
//
// You can see the relationship between execs and opens — when a process
// starts (EXEC), it immediately opens shared libraries and config files (OPEN).
//
// KEY INSIGHT: This is a simplified version of what production tools like
// Datadog's system-probe do. They load dozens of BPF programs (kprobes,
// tracepoints, XDP, cgroup programs) and feed everything into a unified
// event pipeline. The RingBuf/PerfEventArray is the transport layer.
//
// EXTENSION IDEAS (not required):
// 1. Add a kretprobe to capture open() return values (fd or error code)
// 2. Add a uprobe to trace a specific user-space function
// 3. Filter events by PID (pass a target PID via a BPF Array map)
// 4. Add network events from the XDP counter
// 5. Use BPF filesystem pinning for true cross-program map sharing
