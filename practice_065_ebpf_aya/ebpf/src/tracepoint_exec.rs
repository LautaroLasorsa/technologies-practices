// ============================================================================
// Exercise 3: Tracepoint — Monitor process executions (sched_process_exec)
//
// This BPF program attaches to the sched_process_exec tracepoint, which fires
// every time a process calls execve() to execute a new program.
//
// Key concepts:
// - Tracepoints vs Kprobes:
//   * Kprobes attach to ANY kernel function but are UNSTABLE — function
//     signatures and even names can change across kernel versions.
//   * Tracepoints are pre-defined, STABLE hook points in the kernel source.
//     They have a documented format (fields and offsets) that the kernel
//     maintainers commit to not changing without notice.
//   * Rule of thumb: prefer tracepoints when available; use kprobes only
//     for functions without tracepoints.
//
// - TracePointContext:
//   * Unlike ProbeContext (which gives you function arguments), TracePointContext
//     gives you access to a structured data blob.
//   * The fields and their offsets are defined in:
//     /sys/kernel/debug/tracing/events/sched/sched_process_exec/format
//   * You read fields using ctx.read_at::<T>(offset) where offset is the
//     byte offset from the format file.
//
// - PerfEventArray:
//   * Instead of logging (slow, unstructured), this exercise sends structured
//     ExecEvent structs to userspace via a PerfEventArray.
//   * Each CPU has its own ring buffer. The BPF program writes to the current
//     CPU's buffer. Userspace reads from all CPU buffers.
//   * Events can be lost if the buffer is full (userspace reads too slowly).
//     PerfEventArray.output() returns an error in that case.
//
// Tracepoint format (inspect inside container):
//   $ cat /sys/kernel/debug/tracing/events/sched/sched_process_exec/format
//
//   Typical output:
//     name: sched_process_exec
//     ...
//     field:unsigned short common_type;    offset:0;  size:2; signed:0;
//     field:unsigned char common_flags;    offset:2;  size:1; signed:0;
//     field:unsigned char common_preempt;  offset:3;  size:1; signed:0;
//     field:int common_pid;                offset:4;  size:4; signed:1;
//     field:__data_loc char[] filename;    offset:8;  size:4; signed:0;
//     field:pid_t pid;                     offset:12; size:4; signed:1;
//     field:pid_t old_pid;                 offset:16; size:4; signed:1;
//
// IMPORTANT: Offsets vary by kernel version! Always check the format file
// inside the container. The offsets in the TODO below are typical for 5.15+.
// ============================================================================

#![no_std]
#![no_main]

use aya_ebpf::{
    helpers::{bpf_get_current_pid_tgid, bpf_probe_read_kernel_str_bytes},
    macros::{map, tracepoint},
    maps::PerfEventArray,
    programs::TracePointContext,
};
use practice_065_common::ExecEvent;

// TODO(human): Declare a PerfEventArray map for sending ExecEvent to userspace
//
// PerfEventArray is a per-CPU ring buffer array. Each CPU has its own buffer,
// and the BPF program writes to the buffer of the CPU it's currently running on.
// Userspace reads from all CPU buffers using epoll/async.
//
// Declaration:
//
//   #[map]
//   static EXEC_EVENTS: PerfEventArray<ExecEvent> =
//       PerfEventArray::with_max_entries(1024, 0);
//
// with_max_entries(1024, 0):
//   - 1024: Number of per-CPU buffer slots. This is the TOTAL across all CPUs.
//     Each CPU gets 1024 / num_cpus slots. If a CPU's buffer is full when the
//     BPF program tries to write, the event is dropped.
//   - 0: Flags (0 = default).
//
// WHY PerfEventArray instead of HashMap?
// HashMap is for aggregated state (counts, last-seen timestamps). PerfEventArray
// is for streaming individual events to userspace — each execve() produces one
// event that userspace consumes. PerfEventArray preserves per-CPU ordering and
// doesn't overwrite previous events (it's a ring buffer, not a key-value store).
//
// WHY not RingBuf?
// RingBuf (Linux 5.8+) is the modern replacement with better properties
// (single shared buffer, strong ordering). We use PerfEventArray here to teach
// it, and switch to RingBuf in Exercise 5 to compare the two approaches.

// TODO(human): Implement the tracepoint entry point
//
//   #[tracepoint]
//   pub fn tracepoint_exec(ctx: TracePointContext) -> u32 {
//       match try_tracepoint_exec(ctx) {
//           Ok(ret) => ret,
//           Err(ret) => ret,
//       }
//   }
//
//   fn try_tracepoint_exec(ctx: TracePointContext) -> Result<u32, u32> {
//       // Read the PID from the tracepoint context.
//       //
//       // ctx.read_at::<T>(offset) reads sizeof::<T>() bytes starting at
//       // `offset` in the tracepoint's data structure.
//       //
//       // The pid field is at offset 12 (see format above) and is a 4-byte i32.
//       // NOTE: Check the actual offset in YOUR kernel by reading the format file!
//       //
//       // IMPORTANT: read_at returns Result<T, _>. The BPF verifier requires
//       // you to handle the error case — if you unwrap() without checking,
//       // the verifier may reject the program.
//       let pid: i32 = unsafe { ctx.read_at(12) }.map_err(|_| 1u32)?;
//
//       // Create the event struct to send to userspace.
//       // We initialize it with zeroed filename — the BPF program will fill
//       // it in the next step.
//       let mut event = ExecEvent {
//           pid: pid as u32,
//           filename_len: 0,
//           filename: [0u8; 256],
//       };
//
//       // Read the filename from the tracepoint data.
//       //
//       // The filename field uses __data_loc encoding: the 4 bytes at offset 8
//       // contain (length << 16 | offset_from_start). We need to:
//       // 1. Read the __data_loc u32 at offset 8
//       // 2. Extract the actual offset: data_loc & 0xFFFF
//       // 3. Read the string from that offset using bpf_probe_read_kernel_str_bytes
//       //
//       // However, for simplicity, we can use bpf_get_current_comm() to get the
//       // task's comm name (max 16 chars), which is simpler but less precise than
//       // the full filename. For a more complete implementation, you would parse
//       // the __data_loc field.
//       //
//       // Simple approach using the task comm:
//       //   let comm = bpf_get_current_comm().map_err(|_| 1u32)?;
//       //   let len = comm.iter().position(|&c| c == 0).unwrap_or(comm.len());
//       //   event.filename[..len].copy_from_slice(&comm[..len]);
//       //   event.filename_len = len as u32;
//
//       // Alternative: Use bpf_get_current_comm() from aya_ebpf::helpers
//       // which returns [u8; 16] (the kernel's TASK_COMM_LEN).
//       // You'll need to add: use aya_ebpf::helpers::bpf_get_current_comm;
//
//       // Send the event to userspace via PerfEventArray.
//       //
//       // EXEC_EVENTS.output(&ctx, &event, 0) writes the event to the current
//       // CPU's ring buffer. The third argument (0) is flags:
//       //   0 = use sizeof(event) as the data size
//       //   Or pass a specific size to send only part of the struct.
//       //
//       // This can fail if the per-CPU buffer is full (events are being
//       // produced faster than userspace consumes them). In production, you'd
//       // track dropped events via a separate counter map.
//       EXEC_EVENTS.output(&ctx, &event, 0);
//
//       Ok(0)
//   }
//
// KEY LEARNING: Tracepoints provide stable, structured data at fixed offsets.
// Unlike kprobes where you get raw function arguments (which change between
// kernel versions), tracepoints give you a well-defined data layout that the
// kernel team maintains backward compatibility for.
//
// DEBUGGING TIP: If the verifier rejects your program, the error message will
// reference BPF instruction numbers. Use `bpftool prog dump xlated id <N>` to
// see the translated BPF instructions and identify which code path failed
// verification.

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
