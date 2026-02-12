# Practice 025b: ML Compilers — Operator Fusion & Graph Rewrites

## Technologies

- **PyTorch** — Deep learning framework providing the neural network modules and tensor operations we'll fuse
- **torch.fx** — Graph capture and transformation framework for writing compiler passes over PyTorch models
- **torch.fx.subgraph_rewriter** — Pattern matching and replacement API that finds subgraphs in an FX graph and swaps them with optimized replacements

## Stack

Python 3.12+ (uv)

## Theoretical Context

### What is Operator Fusion?

**Operator fusion** is a compiler optimization that combines multiple operations into a single kernel, eliminating intermediate memory reads/writes. In deep learning, most operations are **memory-bound** rather than compute-bound — GPUs can compute teraflops per second but are limited by memory bandwidth (typically ~1 TB/s on high-end GPUs). When operations run as separate kernels, each kernel writes its output to global memory and the next kernel reads it back. Fusion removes these roundtrips by computing multiple operations in registers/shared memory before writing the final result.

The **roofline model** visualizes this: for a given operation, performance is limited by either compute throughput (FLOPS) or memory bandwidth (GB/s), whichever is lower. Most DL operations (element-wise activations, small matrix multiplies, normalization) have low **arithmetic intensity** (FLOPS per byte loaded) and hit the memory bandwidth ceiling. Fusion increases arithmetic intensity by doing more compute per memory access, pushing performance toward the compute ceiling.

### How Fusion Works Internally

Fusion operates on the **computation graph IR**. The compiler identifies fusible patterns — subgraphs where consecutive operations can be merged — and replaces them with fused nodes. Two main fusion types:

1. **Vertical fusion** (producer-consumer): When one operation's output feeds directly into another (e.g., MatMul → Add → ReLU), fuse them into a single kernel that computes all three without intermediate writes. This is the most common and impactful fusion.

2. **Horizontal fusion** (sibling operations): When multiple independent operations share inputs and can be computed together (e.g., two separate element-wise ops on the same tensor), fuse them to amortize memory access costs.

At the **graph level**, fusion rewrites the IR: a chain of nodes `[linear, relu]` becomes a single `[fused_linear_relu]` node. The backend code generator then emits a fused GPU kernel (or CPU vectorized code) for that single node. The key insight: **fusion is a graph transformation** (changing the IR) that enables backend optimizations (generating better kernels).

For **torch.fx-based fusion**, the process is:
1. **Pattern matching**: Traverse the FX graph to find fusible subgraphs (e.g., call_module(Linear) → call_module(ReLU)).
2. **Node replacement**: Create a fused module/function, insert a new node calling it, and redirect consumers to use the fused node.
3. **Dead code elimination**: Remove the original nodes (now unused).
4. **Recompile**: Regenerate the forward() code from the modified graph.

torch.fx's `subgraph_rewriter.replace_pattern()` automates this: you define pattern and replacement functions, and it performs subgraph isomorphism matching + replacement automatically.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Memory-Bound Operations** | Ops limited by memory bandwidth, not compute. Most DL ops fall here. |
| **Arithmetic Intensity** | FLOPS per byte of memory accessed. Fusion increases this. |
| **Roofline Model** | Performance model showing compute vs memory bandwidth limits. |
| **Vertical Fusion** | Fuse producer-consumer chains (A → B → C). Most common and impactful. |
| **Horizontal Fusion** | Fuse sibling operations (A and B both use X) to amortize memory access. |
| **Kernel Launch Overhead** | Cost of launching a GPU kernel (~1-10 μs). Fusion reduces # of launches. |
| **Intermediate Tensors** | Temporary results between ops. Fusion eliminates their memory traffic. |
| **Pattern Matching** | Identifying fusible subgraphs in the IR via graph traversal or isomorphism. |
| **Graph Rewriting** | Replacing matched subgraphs with fused equivalents (insert + rewire + erase). |
| **Conv+BN Folding** | Algebraic fusion: fold BatchNorm params into Conv weights (exact, not approx). |

### Ecosystem Context

Every production ML compiler performs operator fusion aggressively:

- **TorchInductor** (torch.compile's backend): Fuses element-wise ops, MatMul+Bias+Activation patterns. Uses Triton for GPU code generation.
- **TensorRT** (NVIDIA's inference optimizer): Highly aggressive fusion, especially Conv+BN+ReLU. Proprietary fused kernel library.
- **XLA** (used by JAX, TensorFlow): Fusion via HLO IR passes. Supports cross-device fusion (CPU/GPU/TPU).
- **TVM**: User-defined fusion via scheduling primitives. Flexible but requires manual tuning.
- **ONNX Runtime**: Graph-level fusion passes for exported models. Conv+BN, element-wise chains.

**Trade-offs**: Fusion increases arithmetic intensity (good) but may increase register pressure or reduce parallelism (bad if it limits occupancy). Compilers use heuristics and cost models to decide which fusions are profitable. torch.compile's Inductor backend can achieve **1.5-3x speedups** on typical models primarily via fusion.

The **alternative to fusion** is writing monolithic hand-optimized kernels (e.g., FlashAttention for self-attention). This achieves maximal performance but is kernel-specific and not generalizable. Fusion automates the process for arbitrary operation chains.

## Description

Learn why operator fusion is the single most impactful ML compiler optimization. When operations like MatMul -> Add -> ReLU run as separate kernels, each writes intermediate results to GPU memory and the next kernel reads them back. Fusing them into one kernel eliminates these memory round-trips, saving massive memory bandwidth.

This practice builds custom fusion passes using torch.fx's graph transformation API. You'll implement pattern-matching fusion rules, measure the real memory bandwidth savings, and understand how production compilers (TorchInductor, TensorRT, XLA) apply these same techniques at scale.

### What you'll learn

1. **Why fusion matters** — memory bandwidth bottleneck, roofline model intuition, arithmetic intensity
2. **torch.fx graph capture** — `symbolic_trace`, `GraphModule`, `Node` types (placeholder, call_function, call_module, call_method, get_attr, output)
3. **Pattern matching** — identifying subgraphs that match fusion candidates by walking the graph in topological order
4. **Graph rewriting** — replacing matched subgraphs with fused operations using `replace_all_uses_with()` and `erase_node()`
5. **Benchmarking** — measuring actual speedup and memory savings from fusion with `torch.utils.benchmark`
6. **Common fusion patterns** — Conv+BatchNorm folding, MatMul+Bias+ReLU, element-wise operation chains

## Instructions

### Phase 1: torch.fx Graph Basics (~15 min)

Run `fusion/trace_basics.py` to see how torch.fx captures a model into a graph IR. Study the node types, how modules become `call_module` nodes, and how data flows through `args`.

```
uv run python -m fusion.trace_basics
```

### Phase 2: Manual Fusion Pass (~25 min)

Open `fusion/manual_fusion.py`. Implement the `TODO(human)` sections to write a graph pass that walks FX nodes and fuses Linear+ReLU pairs into a single `FusedLinearReLU` module.

```
uv run python -m fusion.manual_fusion
```

### Phase 3: Pattern-Based Rewriting (~20 min)

Open `fusion/pattern_rewriter.py`. Use `torch.fx.subgraph_rewriter.replace_pattern()` to define a pattern (matmul + add + relu) and its fused replacement. This is closer to how production compilers define fusion rules.

```
uv run python -m fusion.pattern_rewriter
```

### Phase 4: Benchmarking (~15 min)

Open `fusion/benchmark.py`. Implement the timing loop and memory measurement to compare fused vs unfused model throughput.

```
uv run python -m fusion.benchmark
```

### Phase 5: Advanced — Conv+BatchNorm Fusion (~15 min)

Open `fusion/conv_bn_fusion.py`. Implement the math to fold BatchNorm parameters into Conv2d weights — a real production optimization used in every inference framework.

```
uv run python -m fusion.conv_bn_fusion
```

## Motivation

- **Operator fusion is the #1 optimization** in every production ML compiler (TensorRT, TorchInductor, XLA, TVM, ONNX Runtime). Understanding it is essential for anyone working on ML systems.
- **Explains torch.compile's magic** — when `torch.compile` makes models 2-3x faster, fusion is the primary reason. This practice demystifies that.
- **Memory bandwidth is the real bottleneck** — modern GPUs have enormous compute (hundreds of TFLOPS) but limited memory bandwidth. Most DL workloads are memory-bound, not compute-bound. Fusion directly attacks this.
- **Builds on 025a** — directly extends graph IR understanding from Practice 025a into practical compiler transformations.
- **Industry relevance** — ML compiler engineering is one of the fastest-growing specializations, and fusion passes are the first thing every compiler engineer learns.

## References

- [torch.fx Documentation](https://pytorch.org/docs/stable/fx.html)
- [torch.fx.subgraph_rewriter](https://pytorch.org/docs/stable/fx.html#torch.fx.subgraph_rewriter)
- [Building a Compiler with torch.fx (PyTorch tutorial)](https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html)
- [Roofline Model (Wikipedia)](https://en.wikipedia.org/wiki/Roofline_model)
- [PyTorch 2.0: torch.compile deep dive](https://pytorch.org/get-started/pytorch-2.0/)

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `uv sync` | Install all dependencies from pyproject.toml |
| Phase 1 | `uv run python -m fusion.trace_basics` | Run torch.fx tracing demo — inspect graph IR and node types |
| Phase 2 | `uv run python -m fusion.manual_fusion` | Run manual Linear+ReLU fusion pass |
| Phase 3 | `uv run python -m fusion.pattern_rewriter` | Run pattern-based MatMul+Bias+ReLU fusion |
| Phase 4 | `uv run python -m fusion.benchmark` | Benchmark fused vs unfused model throughput |
| Phase 5 | `uv run python -m fusion.conv_bn_fusion` | Run Conv+BatchNorm fusion pass |
| All | `uv run python -m fusion.main` | Run all phases sequentially |

## State

`not-started`
