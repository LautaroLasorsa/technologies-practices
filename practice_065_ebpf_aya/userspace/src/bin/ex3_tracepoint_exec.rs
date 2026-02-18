// ============================================================================
// Exercise 3: Userspace loader for the tracepoint_exec BPF program
//
// This binary:
// 1. Loads the tracepoint_exec BPF program
// 2. Attaches it to the sched/sched_process_exec tracepoint
// 3. Reads ExecEvent structs from the PerfEventArray and displays them
//
// Key concepts:
// - TracePoint::attach("sched", "sched_process_exec"): Attaches to a
//   tracepoint by category and name. The tracepoint must exist in
//   /sys/kernel/debug/tracing/events/<category>/<name>/
// - AsyncPerfEventArray: Async wrapper around PerfEventArray that uses
//   tokio/epoll to efficiently wait for events without busy-looping.
// - PerfEventArray reading: You open one buffer per CPU (up to num_cpus),
//   then poll each buffer asynchronously. Events arrive in per-CPU order.
// - bytes::BytesMut: The buffer for reading raw event bytes from the
//   perf buffer. Events are copied into this buffer, then parsed.
// ============================================================================

use anyhow::Context;
use aya::maps::AsyncPerfEventArray;
use aya::programs::TracePoint;
use aya::util::online_cpus;
use aya_log::EbpfLogger;
use bytes::BytesMut;
use log::{info, warn};
use practice_065_common::{bytes_to_str, ExecEvent};
use tokio::signal;

const BPF_BYTES: &[u8] = aya::include_bytes_aligned!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/bpfel-unknown-none/release/tracepoint-exec"
));

// TODO(human): Implement the main function to load the tracepoint and read events
//
//   #[tokio::main]
//   async fn main() -> anyhow::Result<()> {
//       env_logger::init();
//
//       let mut ebpf = aya::Ebpf::load(BPF_BYTES)
//           .context("Failed to load BPF object")?;
//
//       if let Err(e) = EbpfLogger::init(&mut ebpf) {
//           warn!("Failed to init eBPF logger: {}", e);
//       }
//
//       // --- Load and attach the tracepoint program ---
//       //
//       // Unlike kprobes (which use function names), tracepoints use a two-part
//       // name: (category, name). The category groups related events:
//       //   - "sched": scheduler events (exec, fork, switch, etc.)
//       //   - "syscalls": syscall enter/exit events
//       //   - "net": network events
//       //   - "block": block I/O events
//       //
//       // You can list all tracepoints with:
//       //   ls /sys/kernel/debug/tracing/events/
//       //   ls /sys/kernel/debug/tracing/events/sched/
//       let program: &mut TracePoint = ebpf
//           .program_mut("tracepoint_exec")
//           .context("Program 'tracepoint_exec' not found")?
//           .try_into()?;
//       program.load()?;
//       program.attach("sched", "sched_process_exec")
//           .context("Failed to attach tracepoint")?;
//
//       info!("Tracepoint attached to sched/sched_process_exec.");
//       info!("Run commands in another terminal to see execve events.");
//
//       // --- Open the PerfEventArray for reading ---
//       //
//       // The BPF program declared a map named "EXEC_EVENTS" of type PerfEventArray.
//       // We access it by name and convert it to AsyncPerfEventArray for async reading.
//       //
//       // AsyncPerfEventArray<MapData> wraps the map and provides .open() to get
//       // per-CPU buffers that support async reading via tokio.
//       let perf_array = ebpf
//           .take_map("EXEC_EVENTS")
//           .context("Map 'EXEC_EVENTS' not found")?;
//       let mut perf_array = AsyncPerfEventArray::try_from(perf_array)?;
//
//       // --- Spawn a reader task per online CPU ---
//       //
//       // PerfEventArray has one ring buffer per CPU. We need to read from ALL
//       // of them because BPF programs can run on any CPU. Each reader task
//       // polls its CPU's buffer and prints events as they arrive.
//       //
//       // online_cpus() returns the set of currently online CPU IDs.
//       // On a 4-core system, this might return [0, 1, 2, 3].
//       let cpus = online_cpus().context("Failed to get online CPUs")?;
//       for cpu_id in cpus {
//           // Open the perf buffer for this CPU.
//           //
//           // The buffer size (in pages) determines how many events can be
//           // queued before they're lost. 64 pages = 256KB per CPU.
//           let mut buf = perf_array.open(cpu_id, Some(64))?;
//
//           tokio::spawn(async move {
//               // Allocate a BytesMut buffer for reading events.
//               // Each read_events() call fills this buffer with one or more events.
//               //
//               // The buffer must be large enough for at least one ExecEvent.
//               // We use 10 * sizeof(ExecEvent) to batch-read multiple events.
//               let mut buffers = (0..10)
//                   .map(|_| BytesMut::with_capacity(std::mem::size_of::<ExecEvent>()))
//                   .collect::<Vec<_>>();
//
//               loop {
//                   // read_events() waits (async) until at least one event is
//                   // available, then reads as many as fit in the provided buffers.
//                   //
//                   // events.read contains the number of events read.
//                   // events.lost contains the number of events lost due to
//                   // the per-CPU buffer being full (userspace too slow).
//                   let events = buf
//                       .read_events(&mut buffers)
//                       .await
//                       .expect("Failed to read perf events");
//
//                   if events.lost > 0 {
//                       warn!("Lost {} events on CPU {}", events.lost, cpu_id);
//                   }
//
//                   // Parse each event from the raw bytes.
//                   for i in 0..events.read {
//                       let buf = &buffers[i];
//                       // Safety: we know the BPF program writes ExecEvent structs.
//                       // The buffer contains exactly sizeof(ExecEvent) bytes per event.
//                       let event = unsafe {
//                           (buf.as_ptr() as *const ExecEvent).read_unaligned()
//                       };
//
//                       let name = bytes_to_str(&event.filename, event.filename_len as usize);
//                       println!("[EXEC] PID {:>6} -> {}", event.pid, name);
//                   }
//               }
//           });
//       }
//
//       // Wait for Ctrl+C.
//       signal::ctrl_c().await?;
//       info!("Exiting...");
//
//       Ok(())
//   }
//
// WHAT TO OBSERVE:
// Every time you run a command in another terminal, you'll see:
//   [EXEC] PID   1234 -> bash
//   [EXEC] PID   1235 -> ls
//   [EXEC] PID   1236 -> cat
//
// Even simple commands spawn multiple processes. For example, running
// `ls | grep foo` produces at least: bash (the shell), ls, grep.
//
// KEY INSIGHT: This is essentially what `execsnoop` from BCC does. The
// tracepoint approach is preferred over kprobes for exec monitoring because
// sched_process_exec is a stable kernel ABI — it won't break across versions.
//
// COMPARE WITH EXERCISE 1: Exercise 1 traced file opens (sys_openat).
// This exercise traces process executions (execve). In Exercise 5, we'll
// combine both into a unified dashboard.
