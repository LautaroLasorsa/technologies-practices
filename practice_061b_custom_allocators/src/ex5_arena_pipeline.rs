//! Exercise 5: Phase-Based Arena Pipeline
//!
//! The arena-per-phase pattern is a powerful architectural technique used in:
//!
//! - **Compilers** (rustc, GCC, LLVM): Each compilation phase (parsing, type-checking,
//!   optimization, code generation) allocates into its own arena. When a phase completes,
//!   its arena is dropped, freeing all intermediate data structures in one operation.
//!   rustc's `TypedArena` and `DroplessArena` follow exactly this pattern.
//!
//! - **Game engines** (Unreal, Unity internals): Per-frame arenas hold temporary data
//!   (render commands, physics contacts, AI decisions) that are valid only for one frame.
//!   At the end of each frame, the arena resets — no per-object deallocation needed.
//!
//! - **Batch data processing** (ETL pipelines, log processors): Each batch of records
//!   is parsed into an arena, transformed, and output. The arena is reset between batches.
//!
//! - **Request handling** (web servers): Each HTTP request allocates temporaries (parsed
//!   headers, body buffers, response builders) into a per-request arena. When the
//!   response is sent, the arena resets.
//!
//! **Why this pattern works:**
//!
//! 1. **Aligned lifetimes**: Phase data naturally has a uniform lifetime — it's created
//!    at phase start and discarded at phase end. Arenas model this directly.
//!
//! 2. **Zero per-object overhead**: No destructors, no refcounting, no free-list updates
//!    for individual objects. Just reset the bump pointer.
//!
//! 3. **Predictable memory usage**: Each phase's peak memory is bounded by its arena.
//!    Dropping the arena guarantees the memory is reclaimed (unlike general allocation
//!    where freed memory may be retained by the allocator).
//!
//! 4. **Reduced allocator contention**: In multi-threaded pipelines, each phase/thread
//!    can have its own arena — no shared allocator locks needed.

#[allow(unused_imports)]
use bumpalo::Bump;
#[allow(unused_imports)]
use bumpalo::collections::Vec as BumpVec;
#[allow(unused_imports)]
use bumpalo::collections::String as BumpString;

// ---------------------------------------------------------------------------
// Data types for the pipeline
// ---------------------------------------------------------------------------

/// A raw record as "parsed" from input data.
/// In a real pipeline, this would come from parsing CSV, JSON, protobuf, etc.
#[derive(Debug)]
#[allow(dead_code)]
pub struct RawRecord<'bump> {
    pub id: u64,
    pub name: &'bump str,
    pub category: &'bump str,
    pub value: f64,
}

/// A transformed record with computed fields.
/// This is the output of the transform phase — derived from RawRecord but with
/// additional computed data.
#[derive(Debug)]
#[allow(dead_code)]
pub struct TransformedRecord<'bump> {
    pub id: u64,
    pub label: BumpString<'bump>,
    pub normalized_value: f64,
    pub category_code: u32,
}

/// Final output record — a simple owned struct that outlives all arenas.
/// This is what gets written to the output (database, file, network).
#[derive(Debug)]
#[allow(dead_code)]
pub struct OutputRecord {
    pub id: u64,
    pub label: String,
    pub normalized_value: f64,
    pub category_code: u32,
}

// ---------------------------------------------------------------------------
// Pipeline implementation
// ---------------------------------------------------------------------------

/// Run the three-phase arena pipeline.
///
/// Architecture:
///
/// ```text
///   [Raw Input]
///       |
///       v
///   +-- Parse Phase (arena_parse) ----------+
///   |  Parse raw bytes into RawRecord<'a>    |
///   |  All strings/data live in arena_parse  |
///   +----------------------------------------+
///       |  (read RawRecords)
///       v
///   +-- Transform Phase (arena_transform) --+
///   |  Compute derived fields                |
///   |  TransformedRecord<'b> in new arena    |
///   +----------------------------------------+
///       |  (read TransformedRecords)
///       v
///   +-- Output Phase ---+
///   |  Copy final data   |
///   |  into owned Strings |
///   |  Drop both arenas   |
///   +--------------------+
///       |
///       v
///   [Vec<OutputRecord>] (heap-allocated, outlives arenas)
/// ```
///
/// The key insight: the parse arena is dropped BEFORE the output phase, freeing
/// all parse-phase memory. The transform arena is dropped after copying results
/// to OutputRecords. At no point do we hold both arenas simultaneously longer
/// than necessary.
pub fn run_arena_pipeline() {
    // TODO(human): Implement the three-phase arena pipeline.
    //
    // This is the core design exercise. You'll build a data processing pipeline
    // where each phase allocates into its own arena, and arenas are dropped
    // eagerly to reclaim memory between phases.
    //
    // Steps:
    //
    // 1. Define simulated raw input data. Create a function or constant that returns
    //    a Vec of tuples representing raw records:
    //    ```
    //    fn raw_input() -> Vec<(u64, &'static str, &'static str, f64)> {
    //        (0..1000).map(|i| {
    //            let name = match i % 3 {
    //                0 => "alpha",
    //                1 => "beta",
    //                _ => "gamma",
    //            };
    //            let category = match i % 4 {
    //                0 => "electronics",
    //                1 => "clothing",
    //                2 => "food",
    //                _ => "tools",
    //            };
    //            (i as u64, name, category, (i as f64) * 1.5 + 0.5)
    //        }).collect()
    //    }
    //    ```
    //
    // 2. PARSE PHASE:
    //    Create `arena_parse = Bump::new()`.
    //    Parse the raw input into `BumpVec<'_, RawRecord<'_>>`:
    //    ```
    //    let parse_arena = Bump::new();
    //    let raw_records = {
    //        let mut records = BumpVec::new_in(&parse_arena);
    //        for (id, name, category, value) in raw_input() {
    //            // alloc_str copies the &str into the arena, making it 'bump-lived.
    //            // In a real pipeline, you'd parse bytes from a file/network buffer here.
    //            let name = parse_arena.alloc_str(name);
    //            let category = parse_arena.alloc_str(category);
    //            records.push(RawRecord { id, name, category, value });
    //        }
    //        records
    //    };
    //    println!("  Parse phase: {} records, arena: {} bytes",
    //        raw_records.len(), parse_arena.allocated_bytes());
    //    ```
    //
    //    Note: In a real parser, the raw bytes would come from a file or network buffer.
    //    alloc_str() copies them into the arena, ensuring the parsed strings outlive
    //    the input buffer but not the arena.
    //
    // 3. TRANSFORM PHASE:
    //    Create `arena_transform = Bump::new()`.
    //    Transform each RawRecord into a TransformedRecord in the new arena:
    //    ```
    //    let transform_arena = Bump::new();
    //    let transformed = {
    //        let mut records = BumpVec::new_in(&transform_arena);
    //        for raw in raw_records.iter() {
    //            // Compute derived fields
    //            let label = bumpalo::format!(in &transform_arena, "{}_{}", raw.name, raw.id);
    //            let normalized_value = raw.value / 1500.0;  // normalize to [0, 1]
    //            let category_code = match raw.category {
    //                "electronics" => 1,
    //                "clothing" => 2,
    //                "food" => 3,
    //                "tools" => 4,
    //                _ => 0,
    //            };
    //            records.push(TransformedRecord {
    //                id: raw.id,
    //                label,
    //                normalized_value,
    //                category_code,
    //            });
    //        }
    //        records
    //    };
    //    println!("  Transform phase: {} records, arena: {} bytes",
    //        transformed.len(), transform_arena.allocated_bytes());
    //    ```
    //
    // 4. DROP THE PARSE ARENA:
    //    ```
    //    drop(raw_records);  // must drop references before the arena
    //    drop(parse_arena);
    //    println!("  Parse arena dropped — all parse-phase memory freed");
    //    ```
    //    At this point, all RawRecords and their arena-allocated strings are gone.
    //    Only the TransformedRecords (in transform_arena) remain.
    //
    //    KEY: This is the power of phase-based arenas. You don't need to track which
    //    individual records are still referenced. The entire phase's data has a uniform
    //    lifetime, and dropping the arena frees it all.
    //
    // 5. OUTPUT PHASE:
    //    Convert TransformedRecords to OutputRecords (heap-allocated, arena-independent):
    //    ```
    //    let output: Vec<OutputRecord> = transformed.iter().map(|t| {
    //        OutputRecord {
    //            id: t.id,
    //            label: t.label.to_string(),  // copies from arena to heap String
    //            normalized_value: t.normalized_value,
    //            category_code: t.category_code,
    //        }
    //    }).collect();
    //    ```
    //
    // 6. DROP THE TRANSFORM ARENA:
    //    ```
    //    drop(transformed);
    //    drop(transform_arena);
    //    println!("  Transform arena dropped — all transform-phase memory freed");
    //    ```
    //
    // 7. Print summary:
    //    ```
    //    println!("  Output: {} records (heap-allocated, arena-free)", output.len());
    //    println!("  Sample: {:?}", &output[..3.min(output.len())]);
    //    ```
    //
    // Design principle: Each arena lives only as long as needed. The parse arena is
    // dropped before the output phase starts, keeping peak memory = max(parse, transform)
    // rather than parse + transform. In a memory-constrained environment, this matters.

    todo!("Exercise 5a: Implement the three-phase arena pipeline")
}

/// Measure memory usage across pipeline phases.
///
/// This function runs the same pipeline but instruments each phase's memory usage,
/// showing how arena drops reclaim memory between phases.
pub fn measure_arena_pipeline_memory() {
    // TODO(human): Instrument the pipeline with memory measurements.
    //
    // This exercise demonstrates the memory advantage of phase-based arenas.
    // In a general-purpose allocation model, freed memory is returned to the allocator
    // but may not be returned to the OS — the process RSS stays high. With arenas,
    // dropping the arena returns its chunks to the system allocator, which can then
    // return them to the OS (or reuse them for the next arena).
    //
    // Steps:
    //
    // 1. Import `AllocSnapshot` from `crate::ex1_tracking_allocator`.
    //
    // 2. Take a baseline snapshot before the pipeline starts.
    //
    // 3. Run each phase of the pipeline (same as run_arena_pipeline above),
    //    but take snapshots at key points:
    //
    //    a) After parse phase allocation (before transform):
    //       Record snapshot. Compute delta from baseline.
    //       Print: "After parse: +{} allocs, +{} bytes net"
    //
    //    b) After transform phase allocation (both arenas alive):
    //       Record snapshot. Compute delta from baseline.
    //       Print: "After transform (both arenas live): +{} allocs, +{} bytes net"
    //       This is the PEAK memory point — both arenas are alive.
    //
    //    c) After dropping parse arena:
    //       Record snapshot. Compute delta from baseline.
    //       Print: "After parse arena drop: +{} allocs, +{} bytes net"
    //       The net bytes should DROP significantly — the parse arena's chunks
    //       were returned to the system allocator.
    //
    //    d) After output phase (transform arena still alive):
    //       Record snapshot.
    //       Print: "After output copy: +{} allocs, +{} bytes net"
    //
    //    e) After dropping transform arena:
    //       Record snapshot. Compute delta from baseline.
    //       Print: "After transform arena drop: +{} allocs, +{} bytes net"
    //       Net bytes should be just the output Vec<OutputRecord>.
    //
    // 4. Print a summary table:
    //    ```
    //    Phase                     | Net Bytes | Notes
    //    ========================= | ========= | =====
    //    After parse               | ...       | Parse arena alive
    //    After transform (peak)    | ...       | Both arenas alive
    //    After parse arena drop    | ...       | Only transform arena
    //    After output copy         | ...       | Transform arena + output Vec
    //    After transform drop      | ...       | Only output Vec (final)
    //    ```
    //
    //    Key insight: Peak memory = max(parse_arena + transform_arena).
    //    Final memory = output Vec only.
    //    Without arenas, peak = parse_data + transform_data + output_data
    //    (because freed parse data may not be reused by the allocator immediately).
    //
    //    With arenas, you have explicit control over when memory is reclaimed,
    //    leading to lower and more predictable peak memory usage.

    todo!("Exercise 5b: Measure memory across pipeline phases with arena drops")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_arena_pipeline() {
        run_arena_pipeline();
    }

    #[test]
    fn test_measure_arena_pipeline_memory() {
        measure_arena_pipeline_memory();
    }
}
