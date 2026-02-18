// ============================================================================
// Exercise 1: Kprobe — Trace sys_openat syscalls
//
// This BPF program attaches to the kernel's sys_openat function, which is
// called every time a process opens a file (open(), openat(), fopen(), etc.).
//
// When the kprobe fires, the program:
// 1. Gets the current process's PID via bpf_get_current_pid_tgid()
// 2. Logs the PID using aya-log-ebpf's info!() macro
//
// Key concepts:
// - #![no_std]: BPF programs cannot use Rust's standard library. Only `core`
//   is available (no heap, no I/O, no threads, no String, no Vec).
// - #![no_main]: There is no main() function. The kernel calls the function
//   annotated with #[kprobe] directly.
// - #[panic_handler]: Required by #![no_std]. BPF programs cannot panic
//   (the verifier ensures they always terminate), but Rust requires a handler.
// - ProbeContext: Provides access to the probed function's arguments. For
//   sys_openat, arg(0) is the directory fd, arg(1) is the filename pointer.
//
// The userspace loader (ex1_kprobe_open.rs) loads this program, attaches it
// to sys_openat, and displays the log output.
// ============================================================================

#![no_std]
#![no_main]

use aya_ebpf::{
    helpers::bpf_get_current_pid_tgid,
    macros::kprobe,
    programs::ProbeContext,
};
use aya_log_ebpf::info;

// TODO(human): Implement the kprobe entry point for sys_openat
//
// Step-by-step guide:
//
// 1. Create a function annotated with #[kprobe] that takes a ProbeContext:
//
//      #[kprobe]
//      pub fn kprobe_open(ctx: ProbeContext) -> u32 {
//          match try_kprobe_open(ctx) {
//              Ok(ret) => ret,
//              Err(ret) => ret,
//          }
//      }
//
//    WHY this pattern?
//    BPF programs must return a u32 status code (0 = success). The match
//    wrapper converts Result<u32, u32> into a plain u32. This is the standard
//    aya pattern because it lets you use ? (try operator) in the inner function
//    for cleaner error handling, while the outer function always returns u32.
//
// 2. Implement the inner function `try_kprobe_open`:
//
//      fn try_kprobe_open(ctx: ProbeContext) -> Result<u32, u32> {
//          // Get the PID of the process that called sys_openat.
//          //
//          // bpf_get_current_pid_tgid() returns a u64 where:
//          // - Upper 32 bits = TGID (thread group ID = process ID in userspace terms)
//          // - Lower 32 bits = PID (thread ID in kernel terms)
//          //
//          // We right-shift by 32 to extract the TGID (process ID).
//          // This is the same value you see in `ps` or `top`.
//          let pid = (bpf_get_current_pid_tgid() >> 32) as u32;
//
//          // Log the event using aya-log-ebpf.
//          // This sends a message to userspace via a hidden PerfEventArray.
//          // The userspace program displays it via aya-log + env_logger.
//          info!(&ctx, "sys_openat called by PID: {}", pid);
//
//          Ok(0)
//      }
//
//    WHY bpf_get_current_pid_tgid()?
//    This is a BPF helper function — a kernel-provided function callable from
//    BPF programs. The kernel exposes ~200 helpers (bpf_map_lookup_elem,
//    bpf_probe_read_kernel, bpf_ktime_get_ns, etc.). Each program type has
//    access to a specific subset of helpers. kprobes can call most of them.
//
//    WHY info!() instead of println!()?
//    BPF programs cannot do I/O. info!() from aya-log-ebpf serializes the
//    log message into a PerfEventArray that the userspace side reads and
//    prints. This is purely a development/debugging aid — production BPF
//    programs use maps for structured data instead of log strings.

// The panic handler is required by #![no_std].
// BPF programs never actually panic (the verifier prevents it), but
// Rust requires a panic handler to be defined for compilation.
#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
