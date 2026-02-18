// ============================================================================
// Exercise 2: Kprobe with BPF HashMap — Count sys_openat calls per PID
//
// This BPF program extends Exercise 1 by using a BPF HashMap to aggregate
// data in the kernel. Instead of logging every event (expensive), it maintains
// a counter per PID in a HashMap<u32, u64>.
//
// Key concepts:
// - BPF Maps: The primary mechanism for kernel <-> userspace communication.
//   Maps are created by the userspace loader (via aya::Ebpf::load()), but
//   the BPF program accesses them via static references.
// - HashMap<u32, u64>: Key = PID (u32), Value = count (u64).
//   The map is declared with #[map] attribute and a max_entries capacity.
// - Atomic-like updates: BPF HashMap operations (get, insert) are atomic
//   per-entry. No explicit locking needed for single-key updates.
// - The userspace side periodically iterates the map to display counts.
//
// Pattern: "Aggregate in kernel, read in userspace" — this is the fundamental
// pattern for all eBPF observability tools. By counting in the kernel, you
// avoid the overhead of sending an event per syscall to userspace.
// ============================================================================

#![no_std]
#![no_main]

use aya_ebpf::{
    helpers::bpf_get_current_pid_tgid,
    macros::{kprobe, map},
    maps::HashMap,
    programs::ProbeContext,
};
use aya_log_ebpf::info;

// TODO(human): Declare a BPF HashMap for counting syscalls per PID
//
// Declare a static HashMap using the #[map] attribute:
//
//   #[map]
//   static SYSCALL_COUNTS: HashMap<u32, u64> = HashMap::with_max_entries(1024, 0);
//
// Explanation of the declaration:
//
// #[map] — This attribute tells aya-ebpf that this static is a BPF map.
//   During loading, aya's userspace library finds this map definition in the
//   BPF ELF's .maps section and creates the actual kernel map via bpf() syscall.
//
// HashMap<u32, u64> — Generic BPF hash map. The kernel hashes the key (u32 PID)
//   to find the bucket, then stores/retrieves the value (u64 count).
//   BPF HashMaps are O(1) average for lookup and insert.
//
// with_max_entries(1024, 0):
//   - 1024: Maximum number of entries. BPF maps have a fixed capacity declared
//     at creation time. If the map is full, inserts fail. 1024 is enough for
//     tracking active PIDs. Production tools use 10K-100K.
//   - 0: Map flags. 0 means default behavior. Other flags include:
//     BPF_F_NO_PREALLOC (don't pre-allocate all entries — saves memory but
//     slower inserts) and BPF_F_NUMA_NODE (NUMA-aware allocation).
//
// WHY a global static?
// BPF programs have no heap and no way to "create" objects at runtime.
// Maps must be declared as statics with known size at compile time. The
// kernel allocates the actual memory when the map is created during load.

// TODO(human): Implement the kprobe entry point
//
// Follow the same pattern as Exercise 1:
//
//   #[kprobe]
//   pub fn kprobe_counter(ctx: ProbeContext) -> u32 { ... }
//
//   fn try_kprobe_counter(ctx: ProbeContext) -> Result<u32, u32> {
//       let pid = (bpf_get_current_pid_tgid() >> 32) as u32;
//
//       // Look up the current count for this PID.
//       // HashMap::get() returns Option<&u64> — it may be None if this PID
//       // hasn't been seen before.
//       //
//       // IMPORTANT: The pointer returned by get() points into the map's
//       // kernel memory. The verifier ensures you only read through valid
//       // pointers. Dereferencing a None would be a null pointer dereference,
//       // which the verifier catches at load time if you skip the check.
//       //
//       // unsafe { SYSCALL_COUNTS.get(&pid) } returns Option<&u64>
//       let count = unsafe { SYSCALL_COUNTS.get(&pid) }
//           .map(|c| *c)
//           .unwrap_or(0);
//
//       // Insert the incremented count.
//       // HashMap::insert() overwrites the value for the given key.
//       // The third argument (0) is flags: 0 = BPF_ANY (insert or update).
//       // Other options: BPF_NOEXIST (insert only), BPF_EXIST (update only).
//       //
//       // insert() can fail if the map is full and this is a new key.
//       // We use .map_err(|_| 1u32)? to propagate the error.
//       unsafe { SYSCALL_COUNTS.insert(&pid, &(count + 1), 0) }
//           .map_err(|_| 1u32)?;
//
//       Ok(0)
//   }
//
// WHY get() then insert() instead of an atomic increment?
// BPF HashMaps don't have a native "increment" operation (unlike PerCpuArray
// which supports per-CPU atomic updates). The get-then-insert pattern is
// effectively atomic for a single key because BPF programs run with preemption
// disabled on the current CPU. However, on multi-CPU systems, two CPUs could
// race on the same key. For precise counting, use PerCpuHashMap instead.
// For this exercise, HashMap is fine — the race is benign (slightly inaccurate
// counts) and teaches the fundamental map access pattern.

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
