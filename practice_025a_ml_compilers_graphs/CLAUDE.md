# Practice 025a: ML Compilers — Computation Graphs & IR from Scratch

## Technologies

- **PyTorch** — Deep learning framework, used as reference and for torch.fx comparison
- **torch.fx** — PyTorch's graph capture and transformation framework for symbolic tracing and IR manipulation

## Stack

- Python 3.12+ (uv)

## Theoretical Context

### What are Computation Graphs?

A **computation graph** is a directed acyclic graph (DAG) representation of mathematical operations. Each node represents either an operation (addition, multiplication, ReLU activation) or a data value (input, constant). Edges represent data dependencies — if node B uses the output of node A, there's an edge A → B. This explicit DAG structure makes mathematical expressions manipulable as data: you can traverse, transform, and optimize them programmatically.

Computation graphs solve the **dual challenges** of modern deep learning: (1) automatic differentiation (computing gradients for backpropagation) and (2) optimization (transforming the computation to run faster). By representing models as graphs rather than imperative code, frameworks like PyTorch and TensorFlow can automatically compute derivatives via the chain rule, and compilers like XLA and TVM can apply algebraic simplifications, operator fusion, and device-specific code generation.

### How Computation Graphs Work Internally

The graph is built in **forward order** (inputs → outputs) but can be traversed in any valid **topological order** — any ordering where each node appears after all nodes it depends on. During the **forward pass**, nodes are evaluated in topological order: leaf nodes (inputs, constants) have their values set, then operation nodes compute their outputs based on their already-evaluated inputs. This produces the model's output.

During the **backward pass** (reverse-mode automatic differentiation), gradients flow in the **opposite direction** — from outputs back to inputs. Starting with the gradient of the loss (dL/dL = 1), we walk the graph in reverse topological order. For each node, we apply the **chain rule**: multiply the node's accumulated gradient by the local derivative of its operation (e.g., for z = x * y, dL/dx = dL/dz * y), then add this contribution to each input's gradient. The += (accumulate) is critical — a node used by multiple consumers receives gradients from all of them (multivariate chain rule).

Computation graphs use **Static Single Assignment (SSA) form**: each node has exactly one definition (no reassignment). This immutability enables aggressive optimization passes. Two fundamental passes are:

1. **Constant Folding**: If all inputs to an operation are constants, evaluate it at compile time and replace with a CONST node. This reduces runtime computation.
2. **Dead Code Elimination (DCE)**: Starting from output nodes, mark all nodes reachable via backward traversal as "live." Remove unmarked nodes — their results are never used.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **DAG (Directed Acyclic Graph)** | Graph structure with directed edges and no cycles. Topological ordering exists. |
| **Topological Sort** | Ordering nodes so each node appears after its dependencies. Enables forward evaluation. |
| **Forward Pass** | Evaluate nodes in topo order to compute outputs from inputs. |
| **Backward Pass** | Reverse-mode autodiff: propagate gradients output → inputs via chain rule. |
| **Reverse-Mode Autodiff** | Compute gradients of one output w.r.t. many inputs efficiently (O(edges)). |
| **SSA Form (Static Single Assignment)** | Each node defined once, never reassigned. Enables optimization passes. |
| **Constant Folding** | Evaluate operations on constants at compile time, replace with CONST nodes. |
| **Dead Code Elimination** | Remove nodes whose outputs are never used (not reachable from outputs). |
| **Intermediate Representation (IR)** | The graph format used by compilers. Our Node/Graph classes are a minimal IR. |

### Ecosystem Context

In the ML ecosystem, **dynamic graphs** (PyTorch eager mode, TensorFlow 1.x) build the graph implicitly during execution via operator overloading, while **static graphs** (TensorFlow graph mode, JAX, torch.compile) trace or compile the graph ahead of time. Static graphs enable aggressive optimization but sacrifice Python flexibility (data-dependent control flow is limited).

**torch.fx** bridges both worlds: it captures PyTorch code as a traceable static graph while preserving Python semantics where possible. This is the IR that torch.compile uses. Alternatives include **ONNX** (an interchange format for models, used by TensorRT and ONNX Runtime), **TorchScript** (PyTorch's older static graph format, now deprecated in favor of torch.compile), and **XLA's HLO** (JAX's IR, covered in Practice 025c).

The trade-off: dynamic graphs are easier to debug and support arbitrary Python, but static graphs can be optimized more aggressively. Modern frameworks converge on **lazy tracing** (torch.compile, JAX's jit) — execute eagerly by default, trace and compile on-demand for hot paths.

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
