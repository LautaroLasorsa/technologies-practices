"""Phase 2: TorchDynamo Graph Capture — how Python becomes an FX graph.

TorchDynamo is the component of torch.compile that sits closest to Python. Its job
is to take your regular Python code (model.forward) and capture the PyTorch operations
into an FX graph (a structured IR that backends can optimize and compile).

### How TorchDynamo works (conceptually)

1. **Bytecode rewriting.** When you call a compiled model, Dynamo intercepts the
   Python bytecode of the forward() method. It uses CPython's frame evaluation API
   to get access to the bytecode before it runs.

2. **Symbolic execution.** Dynamo "runs" the bytecode symbolically — instead of
   computing real tensor values, it tracks which PyTorch operations are called and
   builds an FX graph of those operations.

3. **Guards.** Dynamo records "guards" — conditions that must be true for the
   compiled code to be valid. For example: "input has shape [32, 784]", "input is
   on CPU", "input dtype is float32". If a guard fails on a future call, Dynamo
   recompiles (this is called a "cache miss" or "recompilation").

4. **Graph output.** The captured FX graph is handed to the backend (eager, aot_eager,
   or inductor) for optimization and code generation.

### torch._dynamo.explain()

This is the key debugging tool. It runs Dynamo on your model and reports:
- How many "graph regions" were captured (ideally 1 — the whole model)
- Whether there were any graph breaks (and why)
- The number of operations in each captured graph

Usage:
    explanation = torch._dynamo.explain(model)(input_tensor)
    print(explanation)

The explanation object has attributes like:
    - explanation.graph_count — number of graph regions
    - explanation.graph_break_count — number of graph breaks
    - explanation.break_reasons — list of reasons for each graph break

### FX Graph structure

An FX graph is a list of Node objects, each representing one operation:

    Node types:
    - placeholder: function inputs (like function parameters)
    - call_function: a call to a PyTorch function (torch.relu, torch.add, etc.)
    - call_method: a method call on a tensor (.view(), .reshape(), etc.)
    - call_module: a call to an nn.Module (self.fc1, self.relu, etc.)
    - get_attr: accessing a module attribute (self.weight, etc.)
    - output: the return value of the graph

Example for y = relu(linear(x)):
    %x         : placeholder  (input)
    %linear    : call_module  (self.linear)
    %relu      : call_function (torch.relu)
    %output    : output       (%relu,)
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────
# Models for graph capture exploration (fully provided)
# ─────────────────────────────────────────────────────────

class SimpleLinear(nn.Module):
    """A single linear layer — the simplest possible model for graph capture.

    This should produce a clean FX graph with very few nodes:
    placeholder -> call_module(linear) -> output
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ThreeLayerMLP(nn.Module):
    """An MLP with operations that show how Dynamo decomposes higher-level ops.

    When Dynamo captures this, observe how:
    - nn.Linear becomes separate weight multiplication + bias addition
    - nn.ReLU becomes a call to a relu function
    - nn.LayerNorm decomposes into mean, variance, normalization operations
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc3 = nn.Linear(dim, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm1(self.fc1(x)))
        x = self.relu(self.norm2(self.fc2(x)))
        x = self.fc3(x)
        return x


class ModelWithReshape(nn.Module):
    """A model that includes tensor shape operations.

    Shape operations (view, reshape, flatten) are interesting because they are
    "zero-cost" at runtime (no data is copied) but they DO appear in the FX graph
    as nodes. Dynamo must track shapes symbolically to correctly capture them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 8, 8)
        x = torch.relu(self.conv(x))
        # x shape: (batch, 16, 8, 8)
        x = x.flatten(start_dim=1)
        # x shape: (batch, 16*8*8)
        x = self.fc(x)
        return x


class ModelWithInplaceMath(nn.Module):
    """A model using torch math operations (not just nn.Modules).

    This shows that Dynamo captures both:
    - nn.Module calls (self.fc1)
    - torch function calls (torch.sigmoid, torch.add)
    - Tensor method calls (.mean())
    """

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.fc1(x)
        b = self.fc2(x)
        # Element-wise operations — each becomes an FX node
        combined = torch.sigmoid(a) * torch.tanh(b)
        return combined.mean(dim=-1)


# ─────────────────────────────────────────────────────────
# Graph printing helpers (fully provided)
# ─────────────────────────────────────────────────────────

def print_fx_graph_summary(model: nn.Module, x: torch.Tensor, model_name: str) -> None:
    """Trace a model and print a summary of the captured FX graph.

    This uses torch.fx.symbolic_trace which does simple Python-level tracing
    (no bytecode rewriting). It shows the "raw" FX graph structure.

    Note: symbolic_trace doesn't handle dynamic control flow. For models with
    if-statements or loops dependent on tensor values, use torch._dynamo.export
    or torch.compile instead.
    """
    try:
        traced = torch.fx.symbolic_trace(model)
        graph = traced.graph

        print(f"\n  --- FX Graph for {model_name} ---")
        print(f"  Nodes: {len(list(graph.nodes))}")
        print()

        # Count node types
        node_types: dict[str, int] = {}
        for node in graph.nodes:
            op = node.op
            node_types[op] = node_types.get(op, 0) + 1

        print("  Node type counts:")
        for op, count in sorted(node_types.items()):
            print(f"    {op:20s}: {count}")

        print()
        print("  Full graph:")
        traced.graph.print_tabular()

    except Exception as e:
        print(f"  [WARN] symbolic_trace failed for {model_name}: {e}")
        print(f"  (This model may have dynamic control flow — try Dynamo instead)")


def run_dynamo_explain(model: nn.Module, x: torch.Tensor, model_name: str) -> None:
    """Run torch._dynamo.explain() on a model and print the results.

    explain() is the primary debugging tool for understanding what Dynamo captures.
    It shows:
    - Number of graph regions (ideally 1)
    - Number of graph breaks (ideally 0)
    - Reasons for any graph breaks
    """
    print(f"\n  --- Dynamo explain() for {model_name} ---")

    torch._dynamo.reset()

    try:
        explanation = torch._dynamo.explain(model)(x)
        print(f"  Graph count:       {explanation.graph_count}")
        print(f"  Graph break count: {explanation.graph_break_count}")

        if explanation.graph_break_count > 0:
            print(f"  Break reasons:")
            for i, reason in enumerate(explanation.break_reasons):
                print(f"    [{i}] {reason}")
        else:
            print(f"  No graph breaks — model compiles as a single graph!")

    except Exception as e:
        print(f"  [ERROR] explain() failed: {e}")

    torch._dynamo.reset()


# ─────────────────────────────────────────────────────────
# TODO(human): Implement Dynamo exploration
# ─────────────────────────────────────────────────────────

def run_dynamo_exploration() -> None:
    """Explore TorchDynamo graph capture on multiple models.

    TODO(human): Implement this function.

    This is the core exercise of Phase 2. You will trace several models and
    analyze the captured FX graphs to understand how Dynamo represents operations.

    ### Steps to implement:

    1. **Trace each model with both tools.**
       For each of the 4 models defined above (SimpleLinear, ThreeLayerMLP,
       ModelWithReshape, ModelWithInplaceMath):

       a) Create the model instance and put it in eval mode.
       b) Create an appropriate input tensor:
          - SimpleLinear: `torch.randn(4, 64)`
          - ThreeLayerMLP: `torch.randn(4, 128)`
          - ModelWithReshape: `torch.randn(4, 1, 8, 8)`
          - ModelWithInplaceMath: `torch.randn(4, 64)`
       c) Call `print_fx_graph_summary(model, x, name)` to see the FX graph.
       d) Call `run_dynamo_explain(model, x, name)` to see Dynamo's analysis.

    2. **Count and categorize FX operations.**
       After tracing ThreeLayerMLP, look at the FX graph output and answer:
       - How many `call_module` nodes are there? (Each nn.Module call = 1 node)
       - How many `call_function` nodes are there? (torch.* function calls)
       - How many `placeholder` nodes? (Should be 1 — the input x)
       - How many `output` nodes? (Should be 1 — the return value)

       Print your counts. Compare with the graph summary output.

    3. **Compare symbolic_trace vs Dynamo.**
       Both `torch.fx.symbolic_trace` and `torch._dynamo.explain` capture FX graphs,
       but they work differently:
       - `symbolic_trace`: Python-level tracing. Cannot handle data-dependent control
         flow (if tensor.item() > 0). Traces at the nn.Module level.
       - `torch._dynamo`: Bytecode-level capture. Handles more Python constructs.
         Traces at the ATen operator level (lower-level decomposition).

       After running both on ThreeLayerMLP, observe:
       - symbolic_trace shows `call_module` for nn.Linear, nn.LayerNorm, nn.ReLU
       - Dynamo may decompose these into lower-level operations

       Print a brief comparison noting the differences you observe.

    4. **Observe the flatten/reshape handling.**
       For ModelWithReshape, look at how `x.flatten(start_dim=1)` appears in the
       FX graph. Is it a `call_method` node (tensor.flatten) or a `call_function`
       node (torch.flatten)?

       Print what you observe about shape operations in the graph.

    ### Key insights to look for:

    - **Every PyTorch operation becomes a node.** Even operations you might think are
      "free" (like reshape, view, contiguous) appear in the graph.

    - **Module boundaries are preserved** in the FX graph (as call_module nodes).
      This is important because later optimization passes can see module structure.

    - **Dynamo captures MORE than symbolic_trace** because it rewrites bytecode
      instead of relying on __torch_function__ protocol.

    - **explain() is your best friend** for debugging torch.compile issues.
      Always start with explain() when a model doesn't compile as expected.

    Hint: Use a list of (model_class, input_shape, name) tuples and loop over them
    to avoid repeating the same trace/explain/print pattern for each model.
    """
    # TODO(human): implement Dynamo exploration
    # Stub: print a placeholder message
    print("  [STUB] run_dynamo_exploration() not yet implemented.")
    print("  Implement the TODO above to explore Dynamo graph capture.")


# ─────────────────────────────────────────────────────────
# Phase runner
# ─────────────────────────────────────────────────────────

def run_phase() -> None:
    """Run Phase 2: TorchDynamo Graph Capture."""
    print("\n" + "#" * 60)
    print("  Phase 2: TorchDynamo Graph Capture")
    print("#" * 60)
    print()

    run_dynamo_exploration()


if __name__ == "__main__":
    run_phase()
