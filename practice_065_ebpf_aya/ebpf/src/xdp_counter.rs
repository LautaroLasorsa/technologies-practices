// ============================================================================
// Exercise 4: XDP — Packet counter by protocol
//
// This BPF program attaches to a network interface at the XDP (eXpress Data
// Path) hook point, the EARLIEST point where you can process packets in Linux.
//
// Key concepts:
// - XDP runs at the network DRIVER level, before the kernel allocates an
//   sk_buff (socket buffer). This makes it extremely fast — packets that are
//   dropped by XDP never enter the kernel network stack at all.
// - XDP programs return an action:
//   * XDP_PASS  — Let the packet continue to the kernel stack (normal path)
//   * XDP_DROP  — Drop the packet immediately (DDoS mitigation)
//   * XDP_TX    — Bounce the packet back out the same interface
//   * XDP_REDIRECT — Send to a different interface or CPU
//   * XDP_ABORTED — Error, drop and log (for debugging)
// - XdpContext gives you raw pointers to packet data: ctx.data() (start)
//   and ctx.data_end() (end). You parse headers manually.
// - The BPF VERIFIER requires bounds checks before every packet data access.
//   If you read a byte at offset X, you must prove X < data_end. The
//   ptr_at<T>() helper pattern handles this.
//
// Protocol classification:
// - Ethernet header (14 bytes) → check EtherType (0x0800 = IPv4)
// - IPv4 header (20+ bytes) → protocol field: 1=ICMP, 6=TCP, 17=UDP
// - We count packets in 4 buckets: [TCP, UDP, ICMP, OTHER] using PerCpuArray
//
// PerCpuArray:
// - Each array element is replicated per CPU — no lock contention.
// - The BPF program increments the counter for its current CPU.
// - Userspace reads all per-CPU values and sums them.
// - This is the highest-performance counting pattern in BPF.
// ============================================================================

#![no_std]
#![no_main]

use core::mem;

use aya_ebpf::{
    bindings::xdp_action,
    macros::{map, xdp},
    maps::PerCpuArray,
    programs::XdpContext,
};
use aya_log_ebpf::info;
use network_types::{
    eth::{EthHdr, EtherType},
    ip::Ipv4Hdr,
};

/// Protocol counter indices in the PerCpuArray.
/// Using named constants instead of magic numbers.
const IDX_TCP: u32 = 0;
const IDX_UDP: u32 = 1;
const IDX_ICMP: u32 = 2;
const IDX_OTHER: u32 = 3;
const NUM_COUNTERS: u32 = 4;

/// IP protocol numbers from IANA assignment.
const IPPROTO_ICMP: u8 = 1;
const IPPROTO_TCP: u8 = 6;
const IPPROTO_UDP: u8 = 17;

// TODO(human): Declare a PerCpuArray map for packet counters
//
// PerCpuArray<T> allocates one T per CPU per array index. This means:
// - No lock contention: each CPU increments its own copy
// - Userspace reads: PerCpuValues<T> (a Vec of T, one per CPU), then sums
// - Perfect for high-frequency counters (packets can arrive on any CPU)
//
// Declaration:
//
//   #[map]
//   static PACKET_COUNTS: PerCpuArray<u64> =
//       PerCpuArray::with_max_entries(NUM_COUNTERS, 0);
//
// with_max_entries(4, 0):
//   - 4 entries: indices 0-3 for TCP, UDP, ICMP, OTHER
//   - 0: flags (default)
//
// Each entry is a u64 counter, replicated once per CPU.
// Total memory: 4 entries * 8 bytes * num_cpus.

// TODO(human): Implement the ptr_at<T> bounds-checking helper
//
// This function is the CRITICAL safety pattern for all XDP packet parsing.
// The BPF verifier requires proof that every memory access is within the
// packet boundaries [data, data_end).
//
// Implementation:
//
//   #[inline(always)]
//   fn ptr_at<T>(ctx: &XdpContext, offset: usize) -> Result<*const T, ()> {
//       let start = ctx.data();
//       let end = ctx.data_end();
//       let len = mem::size_of::<T>();
//
//       // This bounds check is MANDATORY. Without it, the verifier rejects
//       // the program. The verifier tracks pointer ranges and requires an
//       // explicit comparison proving start + offset + len <= end before
//       // any dereference.
//       if start + offset + len > end {
//           return Err(());
//       }
//
//       // Cast the offset pointer to *const T.
//       // Safety: we just proved the access is within bounds.
//       Ok((start + offset) as *const T)
//   }
//
// WHY #[inline(always)]?
// BPF has a limited instruction count and doesn't support function calls in
// older kernels (< 5.10). #[inline(always)] ensures this helper is inlined
// at every call site, avoiding BPF function call overhead and ensuring the
// verifier can trace bounds through the inlined code.
//
// WHY return *const T instead of &T?
// BPF pointers are not Rust references — they don't have lifetime or aliasing
// guarantees. Using raw pointers makes this explicit. You'll dereference with
// unsafe { *ptr } or use read_unaligned() for potentially misaligned fields.

// TODO(human): Implement the XDP entry point
//
//   #[xdp]
//   pub fn xdp_counter(ctx: XdpContext) -> u32 {
//       match try_xdp_counter(ctx) {
//           Ok(ret) => ret,
//           Err(_) => xdp_action::XDP_ABORTED,
//       }
//   }
//
//   fn try_xdp_counter(ctx: XdpContext) -> Result<u32, ()> {
//       // Step 1: Parse the Ethernet header at offset 0.
//       //
//       // EthHdr is 14 bytes: [dst_mac: 6][src_mac: 6][ether_type: 2]
//       // ptr_at performs the bounds check the verifier requires.
//       let ethhdr: *const EthHdr = ptr_at(&ctx, 0)?;
//
//       // Step 2: Check if this is an IPv4 packet.
//       //
//       // EtherType::Ipv4 = 0x0800 (in network byte order).
//       // The network_types crate handles byte order conversion.
//       // If it's not IPv4, we still pass the packet but count it as OTHER.
//       let ether_type = unsafe { (*ethhdr).ether_type };
//       if ether_type != EtherType::Ipv4 {
//           increment_counter(IDX_OTHER)?;
//           return Ok(xdp_action::XDP_PASS);
//       }
//
//       // Step 3: Parse the IPv4 header starting after the Ethernet header.
//       //
//       // Ipv4Hdr starts at offset EthHdr::LEN (14 bytes).
//       // The protocol field tells us TCP(6), UDP(17), or ICMP(1).
//       let ipv4hdr: *const Ipv4Hdr = ptr_at(&ctx, EthHdr::LEN)?;
//       let protocol = unsafe { (*ipv4hdr).proto };
//
//       // Step 4: Map protocol to counter index.
//       let idx = match protocol {
//           IPPROTO_TCP => IDX_TCP,
//           IPPROTO_UDP => IDX_UDP,
//           IPPROTO_ICMP => IDX_ICMP,
//           _ => IDX_OTHER,
//       };
//
//       // Step 5: Increment the per-CPU counter.
//       increment_counter(idx)?;
//
//       // Always pass the packet — we're just counting, not filtering.
//       Ok(xdp_action::XDP_PASS)
//   }
//
// TODO(human): Implement the increment_counter helper
//
//   #[inline(always)]
//   fn increment_counter(idx: u32) -> Result<(), ()> {
//       // PerCpuArray::get_ptr_mut() returns a *mut u64 to the current CPU's
//       // value for the given index. The verifier ensures the index is in bounds.
//       //
//       // IMPORTANT: get_ptr_mut() returns Option<*mut u64>. You MUST check
//       // for None before dereferencing — the verifier will reject the program
//       // if you skip the null check.
//       //
//       // The pointer is to the CURRENT CPU's copy of the value — no other CPU
//       // can access this memory simultaneously, so no atomics needed.
//       let ptr = unsafe { PACKET_COUNTS.get_ptr_mut(idx).ok_or(())? };
//       unsafe { *ptr += 1 };
//       Ok(())
//   }
//
// WHY PerCpuArray instead of HashMap?
// For high-frequency counters (potentially millions of packets/sec per CPU),
// HashMap has hash computation overhead and potential collisions. PerCpuArray
// is O(1) with zero contention — it's literally an array indexed by a constant,
// with one copy per CPU. This is the pattern used by production packet counters.
//
// WHY XDP_PASS and not XDP_DROP?
// We're building an observer, not a firewall. XDP_PASS lets all packets through
// to the normal kernel network stack. If you return XDP_DROP, the packet is
// silently discarded — useful for DDoS mitigation but not for counting.

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
