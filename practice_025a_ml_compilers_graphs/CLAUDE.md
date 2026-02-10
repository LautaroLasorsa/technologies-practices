# Practice 025a: ML Compilers — Computation Graphs & IR from Scratch

## Technologies

- **PyTorch** — Deep learning framework, used as reference and for torch.fx comparison
- **torch.fx** — PyTorch's graph capture and transformation framework for symbolic tracing and IR manipulation

## Stack

- Python 3.12+ (uv)

## Description

Build a minimal computation graph framework from scratch — nodes representing operations, edges representing data flow, forward evaluation, and backward (gradient) propagation. Then implement two classic compiler passes: constant folding (evaluate operations on known constants at compile time) and dead code elimination (remove nodes whose outputs are never used). Finally, compare your hand-built IR with `torch.fx.symbolic_trace` output on the same model.

This practice bridges the gap between "using PyTorch" and "understanding how PyTorch works under the hood." The concepts here are the foundation for ALL ML compiler work in 025b-025f.

### What you'll learn

1. **Computation graphs** — DAG representation of mathematical operations
2. **Forward evaluation** — topological-order traversal of the graph
3. **Backward propagation** — reverse-mode autodiff through the graph
4. **Constant folding** — evaluating known-at-compile-time operations early
5. **Dead code elimination** — removing unused computations
6. **torch.fx IR** — how PyTorch captures and represents graphs

## Instructions

### Phase 1: Computation Graph Data Structures (~15 min)

1. From this folder: `uv sync`
2. Open `graph/ir.py` — read the `Op` enum and `Node` dataclass (fully provided)
3. Read `Graph.add_input()` and `Graph.add_const()` as reference implementations
4. **User implements:** `Graph.add_op()` — creates a new operation node, wires it to its input nodes, and appends it to the graph
5. Test: `uv run python -m graph.main` — Phase 1 output should show the graph structure
6. Key question: Why is a computation graph a DAG (directed acyclic graph) and not a general graph? What would cycles mean?

### Phase 2: Forward Evaluation (~20 min)

1. Open `graph/evaluator.py` — read the module docstring explaining topological sort and evaluation
2. **User implements:** `topological_sort(graph)` — Kahn's algorithm or DFS-based topological ordering
3. **User implements:** `evaluate(graph, inputs)` — walk nodes in topo order, compute each node's value based on its op and inputs
4. Test: `uv run python -m graph.main` — Phase 2 output should show evaluated values
5. Key question: Why must we evaluate in topological order? What happens if we evaluate a MUL node before its inputs are computed?

### Phase 3: Backward Pass (~20 min)

1. Open `graph/backward.py` — read the gradient rules documented in the TODO block
2. **User implements:** `backward(graph, loss_node)` — reverse-mode automatic differentiation
3. Walk nodes in reverse topological order, propagating gradients from output back to inputs
4. Test: `uv run python -m graph.main` — Phase 3 output should show gradients for each input
5. Key question: Why reverse-mode and not forward-mode? (Hint: think about the number of outputs vs inputs in ML — one loss, millions of parameters)

### Phase 4: Graph Optimization Passes (~20 min)

1. Open `graph/passes.py` — read the descriptions of constant folding and DCE
2. **User implements:** `constant_fold(graph)` — find nodes where all inputs are CONST, evaluate them, replace with CONST nodes
3. **User implements:** `dead_code_elimination(graph, outputs)` — mark reachable nodes from outputs, remove unreachable
4. Test: `uv run python -m graph.main` — Phase 4 output should show before/after node counts
5. Key question: In what order should you apply these passes? Does constant folding create more DCE opportunities or vice versa?

### Phase 5: torch.fx Comparison (~15 min)

1. Open `graph/fx_compare.py` — read the provided MLP model and torch.fx tracing code
2. Run: `uv run python -m graph.fx_compare` — observe the FX graph output
3. **User implements:** Build the same MLP using your `graph/ir.py` framework and evaluate it
4. Compare your IR representation with torch.fx's tabular output
5. Key question: What information does torch.fx capture that your IR doesn't? (Hint: shapes, dtypes, module hierarchy)

## Motivation

- **ML compiler knowledge** is the #1 differentiator for senior ML infrastructure roles (NVIDIA, Google, Meta)
- **Understanding computation graphs** demystifies autograd, torch.compile, ONNX, and TensorRT
- **Foundation for the entire 025 series** (fusion, XLA, Triton, TVM, torch.compile)
- **Directly applicable** to optimizing ML inference pipelines at AutoScheduler.AI

## References

- [PyTorch torch.fx Documentation](https://pytorch.org/docs/stable/fx.html)
- [torch.fx: Practical Program Capture and Transformation](https://pytorch.org/docs/stable/fx.html#torch.fx)
- [CS231n: Computational Graphs](https://cs231n.github.io/)
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)

## Commands

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies (PyTorch) |

### Run All Phases

| Command | Description |
|---------|-------------|
| `uv run python -m graph.main` | Run all 5 phases sequentially: build graph, evaluate, backward, optimize, compare |

### Run Individual Modules

| Command | Description |
|---------|-------------|
| `uv run python -m graph.fx_compare` | Run only the torch.fx comparison (Phase 5) |

## State

`not-started`
