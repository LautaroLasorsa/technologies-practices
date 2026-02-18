// ============================================================================
// Exercise 1: Userspace loader for the kprobe_open BPF program
//
// This binary:
// 1. Loads the BPF ELF object (compiled by cargo build-ebpf)
// 2. Initializes the eBPF logger (to receive info!() messages from BPF)
// 3. Finds the "kprobe_open" program in the loaded ELF
// 4. Attaches it to the "sys_openat" kernel function
// 5. Waits for Ctrl+C, displaying BPF log messages as they arrive
//
// Key concepts:
// - aya::Ebpf::load(): Parses the BPF ELF, creates kernel maps, but does
//   NOT yet load the programs into the kernel. The programs are loaded
//   individually via program.load().
// - include_bytes_aligned!(): Embeds the BPF ELF as a byte array in the
//   userspace binary at compile time. At runtime, Ebpf::load() parses this.
// - KProbe::attach("sys_openat", 0): Attaches the loaded program to the
//   kernel function "sys_openat". The second argument (0) is the offset
//   within the function (0 = entry point).
// - EbpfLogger: Bridges BPF-side aya-log-ebpf with userspace env_logger.
//   It reads log events from a hidden PerfEventArray and forwards them.
// ============================================================================

use anyhow::Context;
use aya::programs::KProbe;
use aya_log::EbpfLogger;
use log::{info, warn};
use tokio::signal;

/// Path to the compiled BPF ELF object for this exercise.
/// The BPF program is compiled separately by `cargo build-ebpf` and the
/// output lands in target/bpfel-unknown-none/release/.
///
/// include_bytes_aligned! reads the file at compile time and embeds it as
/// a &[u8] in the binary. The "aligned" part ensures 8-byte alignment,
/// which is required for parsing the ELF headers.
const BPF_BYTES: &[u8] = aya::include_bytes_aligned!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/bpfel-unknown-none/release/kprobe-open"
));

// TODO(human): Implement the main function to load and attach the kprobe
//
// The standard pattern for an aya userspace loader:
//
//   #[tokio::main]
//   async fn main() -> anyhow::Result<()> {
//       // Initialize the Rust logger. RUST_LOG=info enables info-level output.
//       // This receives both Rust-side log messages AND BPF-side aya-log messages.
//       env_logger::init();
//
//       // --- Step 1: Load the BPF object ---
//       //
//       // Ebpf::load() parses the ELF, extracts programs and maps, and creates
//       // the kernel maps (via bpf() syscall). The programs are NOT yet loaded
//       // into the kernel — that happens in step 3.
//       //
//       // If this fails with "Operation not permitted", you need to run with
//       // sudo or inside a privileged container.
//       let mut ebpf = aya::Ebpf::load(BPF_BYTES)
//           .context("Failed to load BPF object. Are you running as root?")?;
//
//       // --- Step 2: Initialize the BPF logger ---
//       //
//       // EbpfLogger connects the BPF-side aya-log-ebpf with userspace env_logger.
//       // It reads from a hidden PerfEventArray map that aya-log-ebpf writes to.
//       // If initialization fails (e.g., the BPF program doesn't use aya-log-ebpf),
//       // we just warn and continue — logging is optional.
//       if let Err(e) = EbpfLogger::init(&mut ebpf) {
//           warn!("Failed to initialize eBPF logger: {}", e);
//       }
//
//       // --- Step 3: Get the kprobe program and load it ---
//       //
//       // ebpf.program_mut("kprobe_open") finds the program by its function name
//       // (the name of the #[kprobe] function in the BPF code).
//       //
//       // .try_into()? converts it from a generic Program to a KProbe type.
//       // This is how aya provides type-safe program handling.
//       //
//       // program.load()? sends the BPF bytecode to the kernel verifier.
//       // If verification passes, the program is JIT-compiled and ready to attach.
//       // If verification FAILS, you get a detailed error with the verifier log.
//       let program: &mut KProbe = ebpf
//           .program_mut("kprobe_open")
//           .context("BPF program 'kprobe_open' not found in ELF")?
//           .try_into()?;
//
//       program.load()
//           .context("Failed to load kprobe into kernel (verifier rejected?)")?;
//
//       // --- Step 4: Attach to sys_openat ---
//       //
//       // This creates a kernel kprobe on the "sys_openat" function.
//       // Every time ANY process calls open()/openat()/fopen(), the BPF program
//       // will fire and execute.
//       //
//       // The second argument (0) is the instruction offset within the function.
//       // 0 means "attach at function entry" (most common). A non-zero offset
//       // would attach mid-function, which is rarely needed.
//       //
//       // NOTE: The function name must match exactly. On some kernels, the
//       // actual function is "__x64_sys_openat" or "do_sys_openat2". If
//       // "sys_openat" doesn't work, check with:
//       //   cat /proc/kallsyms | grep openat
//       program.attach("sys_openat", 0)
//           .context("Failed to attach kprobe. Try '__x64_sys_openat' or 'do_sys_openat2'")?;
//
//       info!("Kprobe attached to sys_openat. Waiting for events... (Ctrl+C to exit)");
//       info!("Try running 'cat /etc/hostname' or 'ls /' in another terminal.");
//
//       // --- Step 5: Wait for Ctrl+C ---
//       //
//       // The BPF program is now running in the kernel. Every sys_openat call
//       // will trigger it, and aya-log messages will appear in our terminal.
//       // We just wait here until the user presses Ctrl+C.
//       //
//       // When main() returns, the Ebpf object is dropped, which:
//       // - Detaches all attached programs
//       // - Closes all map file descriptors
//       // - The kernel garbage-collects the BPF programs and maps
//       signal::ctrl_c().await?;
//       info!("Exiting...");
//
//       Ok(())
//   }
//
// WHAT TO OBSERVE:
// When you run this and then open files in another terminal (ls, cat, etc.),
// you should see log lines like:
//   [INFO  kprobe_open] sys_openat called by PID: 1234
//   [INFO  kprobe_open] sys_openat called by PID: 1234
//   [INFO  kprobe_open] sys_openat called by PID: 5678
//
// You'll be surprised how many files are opened — even a simple `ls` opens
// several shared libraries, locale files, and directory entries.
//
// TROUBLESHOOTING:
// - "Operation not permitted" → Run inside the privileged Docker container
// - "Failed to attach kprobe" → The function name varies by kernel. Try:
//     cat /proc/kallsyms | grep sys_openat
//   and use the exact name you find.
// - No output → Make sure RUST_LOG=info is set
