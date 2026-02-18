// ============================================================================
// Exercise 2: Userspace loader for the kprobe_counter BPF program
//
// This binary:
// 1. Loads the kprobe_counter BPF program
// 2. Attaches it to sys_openat
// 3. Periodically reads the SYSCALL_COUNTS HashMap and displays per-PID counts
//
// Key concepts:
// - Reading BPF maps from userspace: The HashMap created by the BPF program
//   is accessible from userspace via aya::maps::HashMap. You get a reference
//   to it from the loaded Ebpf object.
// - Map iteration: HashMap::iter() yields (key, value) pairs. For PerCpuHashMap,
//   values would be PerCpuValues<T> instead.
// - Polling pattern: We read the map every 2 seconds in a loop. This is the
//   simplest approach. For real-time streaming, use PerfEventArray or RingBuf.
// ============================================================================

use std::collections::BTreeMap;

use anyhow::Context;
use aya::maps::HashMap;
use aya::programs::KProbe;
use aya_log::EbpfLogger;
use log::{info, warn};
use tokio::{signal, time};

const BPF_BYTES: &[u8] = aya::include_bytes_aligned!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/bpfel-unknown-none/release/kprobe-counter"
));

// TODO(human): Implement the main function to load the kprobe and read the HashMap
//
//   #[tokio::main]
//   async fn main() -> anyhow::Result<()> {
//       env_logger::init();
//
//       // Load the BPF object (creates maps + programs).
//       let mut ebpf = aya::Ebpf::load(BPF_BYTES)
//           .context("Failed to load BPF object")?;
//
//       if let Err(e) = EbpfLogger::init(&mut ebpf) {
//           warn!("Failed to init eBPF logger: {}", e);
//       }
//
//       // Load and attach the kprobe.
//       let program: &mut KProbe = ebpf
//           .program_mut("kprobe_counter")
//           .context("Program 'kprobe_counter' not found")?
//           .try_into()?;
//       program.load()?;
//       program.attach("sys_openat", 0)?;
//
//       info!("Kprobe attached. Displaying per-PID syscall counts every 2 seconds...");
//       info!("Generate activity in another terminal (ls, cat, etc.)");
//
//       // --- Read the map in a loop ---
//       //
//       // We spawn a task that reads the HashMap every 2 seconds.
//       // The main task waits for Ctrl+C.
//       //
//       // IMPORTANT: We need to get the map AFTER loading the BPF object.
//       // The map is created during Ebpf::load() and we access it by name.
//       //
//       // aya::maps::HashMap::try_from() converts the generic MapData into
//       // a typed HashMap<&MapData, u32, u64>. The key and value types must
//       // match what the BPF program declared.
//
//       let map_task = tokio::spawn(async move {
//           // We need to get the map from the ebpf object.
//           // The HashMap is accessed via ebpf.map("SYSCALL_COUNTS").
//           //
//           // NOTE: We move ebpf into this task because we need a &mut reference
//           // to iterate the map. In a more complex program, you'd use Arc<Mutex<>>
//           // or separate the map handle before spawning tasks.
//           let mut interval = time::interval(time::Duration::from_secs(2));
//
//           loop {
//               interval.tick().await;
//
//               // Get a reference to the map.
//               //
//               // HashMap::try_from() borrows the map data from the Ebpf object.
//               // The turbofish types <_, u32, u64> specify the key/value types.
//               //
//               // .map() returns Option<&MapData>, which we convert to HashMap.
//               let map_data = ebpf.map("SYSCALL_COUNTS");
//               if let Some(map_data) = map_data {
//                   let counts: HashMap<_, u32, u64> =
//                       HashMap::try_from(map_data).expect("Failed to parse map");
//
//                   // Iterate all entries and collect into a sorted BTreeMap for display.
//                   //
//                   // HashMap::iter() returns Result<(u32, u64), _> for each entry.
//                   // We collect into a BTreeMap<u32, u64> to display PIDs in order.
//                   let entries: BTreeMap<u32, u64> = counts
//                       .iter()
//                       .filter_map(|result| result.ok())
//                       .collect();
//
//                   if !entries.is_empty() {
//                       println!("\n--- Syscall counts per PID ---");
//                       for (pid, count) in &entries {
//                           println!("  PID {:>6}: {} openat calls", pid, count);
//                       }
//                       println!("  Total PIDs tracked: {}", entries.len());
//                   }
//               }
//           }
//       });
//
//       // Wait for Ctrl+C, then cancel the map reading task.
//       signal::ctrl_c().await?;
//       map_task.abort();
//       info!("Exiting...");
//
//       Ok(())
//   }
//
// WHAT TO OBSERVE:
// Every 2 seconds, you'll see a table like:
//   --- Syscall counts per PID ---
//     PID    123: 47 openat calls
//     PID    456: 12 openat calls
//     PID    789: 3 openat calls
//     Total PIDs tracked: 3
//
// The counts increase over time as processes open files. System daemons
// (like systemd, snapd) will show high counts because they constantly
// access files for monitoring and configuration.
//
// KEY INSIGHT: This is how tools like `opensnoop` (from BCC) work — they
// attach a kprobe to sys_openat and aggregate in-kernel. The difference
// is that `opensnoop` also reads the filename argument (ctx.arg(1)) and
// streams individual events via PerfEventArray. You already have the
// building blocks from Exercise 1 and will combine them in Exercise 5.
