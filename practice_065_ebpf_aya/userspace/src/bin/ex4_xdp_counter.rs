// ============================================================================
// Exercise 4: Userspace loader for the XDP packet counter
//
// This binary:
// 1. Loads the xdp_counter BPF program
// 2. Attaches it to the loopback interface (lo) using XDP
// 3. Periodically reads the PerCpuArray counters and displays per-protocol totals
//
// Key concepts:
// - Xdp::attach(&iface, XdpFlags): Attaches the XDP program to a network
//   interface. XdpFlags control the attachment mode:
//   * default() — Use native XDP if the driver supports it (fastest)
//   * SKB_MODE  — Fall back to generic/skb mode (works everywhere, slower)
//   Native mode processes packets before sk_buff allocation; SKB mode
//   processes after. Most virtual interfaces (like Docker's veth) only
//   support SKB mode.
// - PerCpuArray reading: Each array element has one value per CPU.
//   PerCpuArray::get(&index, 0) returns PerCpuValues<u64>, which is
//   essentially a Vec<u64> with one entry per CPU. Sum them for the total.
// - Loopback interface (lo): We attach to lo because it's always available
//   inside Docker. Ping 127.0.0.1 generates ICMP traffic on lo.
// ============================================================================

use std::time::Duration;

use anyhow::Context;
use aya::maps::PerCpuArray;
use aya::programs::{Xdp, XdpFlags};
use aya_log::EbpfLogger;
use log::{info, warn};
use tokio::{signal, time};

const BPF_BYTES: &[u8] = aya::include_bytes_aligned!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/bpfel-unknown-none/release/xdp-counter"
));

/// Protocol counter indices (must match the BPF program's constants).
const IDX_TCP: u32 = 0;
const IDX_UDP: u32 = 1;
const IDX_ICMP: u32 = 2;
const IDX_OTHER: u32 = 3;

// TODO(human): Implement the main function to load XDP and display packet counts
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
//       // --- Load and attach the XDP program ---
//       //
//       // XDP programs are attached to a SPECIFIC network interface.
//       // We use "lo" (loopback) because it's always available.
//       //
//       // XdpFlags::default() tries native XDP first.
//       // If that fails (common in Docker/virtual environments), fall back
//       // to SKB_MODE which works everywhere.
//       let program: &mut Xdp = ebpf
//           .program_mut("xdp_counter")
//           .context("Program 'xdp_counter' not found")?
//           .try_into()?;
//       program.load()?;
//
//       // Try default (native) XDP first, fall back to SKB mode.
//       //
//       // WHY the fallback?
//       // Native XDP requires driver support. The loopback interface and
//       // Docker's virtual interfaces (veth) typically don't have native
//       // XDP support. SKB mode works with any interface but is slower
//       // because the packet has already been wrapped in an sk_buff.
//       // For observability (counting), the performance difference doesn't matter.
//       if let Err(e) = program.attach("lo", XdpFlags::default()) {
//           warn!("Native XDP failed ({}), trying SKB mode...", e);
//           program.attach("lo", XdpFlags::SKB_MODE)
//               .context("Failed to attach XDP in SKB mode")?;
//           info!("XDP attached to 'lo' in SKB mode.");
//       } else {
//           info!("XDP attached to 'lo' in native mode.");
//       }
//
//       info!("Displaying packet counts every 2 seconds...");
//       info!("Generate traffic: ping -c 5 127.0.0.1 or curl http://127.0.0.1");
//
//       // --- Read PerCpuArray in a loop ---
//       //
//       // We take the map from the Ebpf object (consuming it) so we can
//       // move it into the spawned task.
//       let map_data = ebpf.take_map("PACKET_COUNTS")
//           .context("Map 'PACKET_COUNTS' not found")?;
//       let counts_map = PerCpuArray::<_, u64>::try_from(map_data)?;
//
//       let counter_task = tokio::spawn(async move {
//           let mut interval = time::interval(Duration::from_secs(2));
//
//           loop {
//               interval.tick().await;
//
//               // Read each protocol's per-CPU counters and sum them.
//               //
//               // PerCpuArray::get(&index, flags) returns PerCpuValues<u64>,
//               // which contains one u64 per online CPU.
//               //
//               // .iter().sum() gives the total across all CPUs.
//               //
//               // This pattern (per-CPU counting in kernel, sum in userspace)
//               // is the standard approach for high-frequency counters because:
//               // 1. No lock contention in the kernel (each CPU has its own copy)
//               // 2. Summing in userspace is cheap (done once per display interval)
//               let tcp = sum_per_cpu(&counts_map, IDX_TCP);
//               let udp = sum_per_cpu(&counts_map, IDX_UDP);
//               let icmp = sum_per_cpu(&counts_map, IDX_ICMP);
//               let other = sum_per_cpu(&counts_map, IDX_OTHER);
//
//               let total = tcp + udp + icmp + other;
//               if total > 0 {
//                   println!("\n--- Packet counts on lo ---");
//                   println!("  TCP:   {:>8}", tcp);
//                   println!("  UDP:   {:>8}", udp);
//                   println!("  ICMP:  {:>8}", icmp);
//                   println!("  Other: {:>8}", other);
//                   println!("  Total: {:>8}", total);
//               }
//           }
//       });
//
//       signal::ctrl_c().await?;
//       counter_task.abort();
//       info!("Exiting...");
//
//       Ok(())
//   }
//
// TODO(human): Implement the sum_per_cpu helper
//
// This function reads one entry from the PerCpuArray and sums all per-CPU values.
//
//   fn sum_per_cpu(map: &PerCpuArray<aya::maps::MapData, u64>, idx: u32) -> u64 {
//       match map.get(&idx, 0) {
//           Ok(values) => values.iter().sum(),
//           Err(_) => 0,
//       }
//   }
//
// WHAT TO OBSERVE:
// 1. Start the program and see all zeros.
// 2. In another terminal: `ping -c 5 127.0.0.1`
//    → ICMP count increases (by 10: 5 echo requests + 5 echo replies)
// 3. In another terminal: `curl http://127.0.0.1` (if a server is running)
//    → TCP count increases
// 4. DNS lookups generate UDP traffic on some configurations.
//
// KEY INSIGHT: XDP is the fastest packet processing path in Linux. Cloudflare
// uses XDP to drop ~10M DDoS packets/sec per server. Facebook uses XDP for
// load balancing (Katran). You're using it for counting (read-only), but
// the same infrastructure supports packet modification, forwarding, and dropping.
//
// PERFORMANCE NOTE: On the loopback interface, you won't see the full XDP
// performance benefit because lo doesn't have a real network driver. On a
// physical NIC with native XDP support, processing happens in the driver
// before the kernel allocates any memory for the packet.
