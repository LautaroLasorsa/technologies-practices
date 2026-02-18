// ============================================================================
// Build script for the userspace crate
//
// This build.rs does NOT compile the BPF programs — that's done separately
// via `cargo build-ebpf` (see .cargo/config.toml).
//
// Instead, this script:
// 1. Sets the OUT_DIR environment variable so that include_bytes_aligned!()
//    can find the compiled BPF ELF files.
// 2. Tells cargo to re-run if the BPF binaries change.
//
// The BPF ELF files are compiled to:
//   target/bpfel-unknown-none/release/<program-name>
//
// The userspace binary includes them at compile time using:
//   aya::include_bytes_aligned!(concat!(env!("CARGO_MANIFEST_DIR"), "/../target/bpfel-unknown-none/release/kprobe-open"))
//
// This approach (compile BPF separately, include as bytes) is simpler than
// using aya-build in build.rs, and gives you more control over the build
// process. The trade-off is that you must remember to run `cargo build-ebpf`
// before `cargo build`.
// ============================================================================

fn main() {
    // Re-run this build script if any BPF binary changes.
    // This ensures the userspace binary re-embeds the latest BPF code.
    println!("cargo:rerun-if-changed=../target/bpfel-unknown-none/release/");

    // Verify BPF programs exist (helpful error message if you forgot to build them).
    let bpf_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../target/bpfel-unknown-none/release");

    if !bpf_dir.exists() {
        println!(
            "cargo:warning=BPF programs not found at {:?}. Run `cargo build-ebpf` first!",
            bpf_dir
        );
    }
}
