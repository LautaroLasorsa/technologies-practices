# Practice 025c: ML Compilers — XLA & HLO Inspection

## Technologies

- **JAX** — Google's numerical computing library with automatic differentiation and XLA compilation. JAX transforms pure Python+NumPy functions into optimized XLA computations via `jit`, `grad`, and `vmap`.
- **XLA (Accelerated Linear Algebra)** — Google's ML compiler that powers JAX, TensorFlow, and PyTorch/XLA. XLA takes high-level operations and compiles them into optimized machine code, applying fusion, layout optimization, and algebraic simplification.
- **HLO (High-Level Optimizer)** — XLA's intermediate representation. HLO is the graph IR that XLA optimization passes operate on — it's what you read when debugging XLA performance.

## Stack

Python 3.12+ (uv), CPU-only (GPU optional via WSL)

> **Platform note:** JAX CPU works natively on Windows: `pip install jax`. GPU support (`jax[cuda12]`) requires Linux or WSL. This practice focuses on IR inspection — CPU is sufficient. All exercises produce meaningful output on CPU; the compiler optimizations are the same (fusion, layout, simplification), only the backend-specific code generation differs.

## Description

Explore XLA through JAX's lens. Capture computation graphs using `jax.make_jaxpr` (JAX's IR), then inspect the lower-level HLO IR that XLA produces. Understand what optimizations XLA applies automatically: algebraic simplification, operator fusion, layout assignment, and constant folding. No custom compiler work — the goal is to **read and understand compiler IRs**.

This is the "reading compiler output" practice — essential for debugging performance issues in JAX/TPU workloads. When a JAX model runs slowly, the first thing you do is dump the HLO and see what XLA decided. This practice teaches you to do exactly that.

### What you'll learn

1. **JAX fundamentals** — `jax.numpy`, `jit`, `grad`, `vmap` — JAX's functional API and how it differs from PyTorch
2. **Jaxpr IR** — JAX's intermediate representation, what `make_jaxpr` captures, reading primitive operations
3. **HLO IR** — XLA's optimization representation, reading HLO text format, understanding HLO instructions
4. **XLA optimizations** — fusion, layout assignment, algebraic simplification, constant folding
5. **Comparing Jaxpr vs HLO** — what changes between JAX-level and XLA-level IR, where optimizations happen

## Instructions

### Phase 1: JAX Basics (~15 min)

Run `xla_inspect/basics.py` to see JAX's core APIs in action: `jax.numpy`, `jit` compilation, and `grad` for automatic differentiation. Study how JAX treats functions as first-class objects that can be transformed.

```
uv run python -m xla_inspect.basics
```

Then implement the `TODO(human)` sections: a 2-layer MLP as a pure function (no classes — JAX style), JIT-compile it, and compute its gradient.

### Phase 2: Jaxpr Inspection (~20 min)

Open `xla_inspect/jaxpr_demo.py`. Use `jax.make_jaxpr` to capture the MLP's computation graph as a Jaxpr — JAX's internal IR. Read the output and understand the primitives: `dot_general`, `add`, `max` (ReLU), and how data flows.

```
uv run python -m xla_inspect.jaxpr_demo
```

### Phase 3: HLO Dump & Reading (~25 min)

Open `xla_inspect/hlo_dump.py`. Trigger XLA compilation and dump the HLO text representation. Read the dumped IR and identify fusion operations, layout assignments, and constant folding.

```
uv run python -m xla_inspect.hlo_dump
```

To get full HLO dumps to a directory (for manual inspection):

```
set XLA_FLAGS=--xla_dump_to=xla_dump --xla_dump_hlo_as_text
uv run python -m xla_inspect.hlo_dump
```

### Phase 4: Optimization Analysis (~20 min)

Open `xla_inspect/optimization_compare.py`. Compare un-optimized vs optimized HLO to see what XLA's optimization passes changed. Count operations, identify fusions, and summarize the compiler's decisions.

```
uv run python -m xla_inspect.optimization_compare
```

### Phase 5: Grad and Vmap IR (~15 min)

Return to `xla_inspect/jaxpr_demo.py` and the `TODO(human)` sections for `grad` and `vmap`. See how these transformations change the Jaxpr: `grad` adds backward-pass primitives, `vmap` adds batching dimensions.

```
uv run python -m xla_inspect.jaxpr_demo
```

## Motivation

- **XLA powers Google's TPU ecosystem** and is increasingly used for GPU workloads. Understanding its IR is essential for anyone working with JAX or TensorFlow at scale.
- **Reading compiler IR is a fundamental skill** for ML performance engineering. When a model is slow, you dump the HLO and read it — this practice teaches you how.
- **Understanding what the compiler does automatically** helps you write compiler-friendly code. Knowing that XLA fuses element-wise ops means you don't need to manually optimize them.
- **JAX is growing rapidly** in research (DeepMind, Google Brain) and production (Google Cloud TPU workloads). Understanding its internals is increasingly valuable.
- **Builds on 025a and 025b** — after learning graph IR (025a) and fusion passes (025b), this practice shows how a real production compiler (XLA) implements those same concepts.

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Understanding Jaxprs](https://jax.readthedocs.io/en/latest/jaxpr.html)
- [XLA: Optimizing Compiler for ML](https://www.tensorflow.org/xla)
- [XLA HLO Operation Semantics](https://openxla.org/xla/operation_semantics)
- [JAX — The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [How JAX primitives work](https://jax.readthedocs.io/en/latest/jaxpr.html#understanding-jaxprs)

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `uv sync` | Install all dependencies (JAX CPU) from pyproject.toml |
| Phase 1 | `uv run python -m xla_inspect.basics` | Run JAX basics — jit, grad, jax.numpy on simple functions |
| Phase 2 | `uv run python -m xla_inspect.jaxpr_demo` | Capture and print Jaxpr IR for MLP and transformations |
| Phase 3 | `uv run python -m xla_inspect.hlo_dump` | Trigger XLA compilation and inspect HLO via JAX APIs |
| Phase 3 (full dump) | `set XLA_FLAGS=--xla_dump_to=xla_dump --xla_dump_hlo_as_text && uv run python -m xla_inspect.hlo_dump` | Dump full HLO text files to `xla_dump/` directory for manual inspection |
| Phase 4 | `uv run python -m xla_inspect.optimization_compare` | Compare un-optimized vs optimized HLO, count ops and fusions |
| Phase 5 | `uv run python -m xla_inspect.jaxpr_demo` | Re-run jaxpr_demo after implementing grad/vmap TODO sections |
| All | `uv run python -m xla_inspect.main` | Run all phases sequentially |

## State

`not-started`
