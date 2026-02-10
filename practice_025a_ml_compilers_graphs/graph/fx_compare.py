"""torch.fx comparison — trace a PyTorch model and compare with our IR.

torch.fx is PyTorch's built-in graph capture framework. It uses "symbolic
tracing" to record the operations a model performs, producing a Graph IR
that can be inspected, transformed, and optimized.

This module:
1. Defines a simple 2-layer MLP as an nn.Module
2. Traces it with torch.fx.symbolic_trace
3. Prints the captured graph in tabular form
4. (TODO) Builds the same model using our graph/ir.py and compares

### How torch.fx symbolic tracing works

When you call `torch.fx.symbolic_trace(model)`, PyTorch:
1. Creates "proxy" objects that record operations instead of executing them
2. Runs the model's forward() with these proxies
3. Captures every operation as a node in an FX Graph

The result is an IR (intermediate representation) very similar to what we
built by hand — nodes with operations, inputs, and outputs. But torch.fx
also captures:
- Module hierarchy (which nn.Linear produced this operation)
- Tensor shapes and dtypes (metadata)
- Python control flow (with limitations — no data-dependent branching)

This is the SAME IR that torch.compile uses as its starting point for
optimization. Understanding it is key to 025f.
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """A minimal 2-layer MLP for tracing demonstration.

    Architecture: input(4) → Linear(4,8) → ReLU → Linear(8,2) → output(2)

    This is intentionally simple — the point is to see how torch.fx
    captures these operations as a graph, not to solve a real task.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def trace_and_print_fx_graph() -> None:
    """Trace the SimpleMLP with torch.fx and print the graph.

    This shows you the IR that torch.fx produces — a flat list of nodes
    with operations, targets (which module/function), and arguments.
    """
    model = SimpleMLP()
    print("\n" + "=" * 60)
    print("  torch.fx Symbolic Trace of SimpleMLP")
    print("=" * 60)

    # Symbolic tracing: captures the forward() as a graph
    traced = torch.fx.symbolic_trace(model)

    # Print the graph in tabular form — this shows:
    # - opcode: placeholder (input), call_module (nn.Module), output
    # - target: which module or function was called
    # - args: references to other nodes (the edges in the graph)
    traced.graph.print_tabular()

    print("\n--- Generated code (what torch.fx would execute) ---")
    print(traced.code)

    # Also show the raw graph nodes for comparison with our IR
    print("--- Raw FX nodes ---")
    for node in traced.graph.nodes:
        print(f"  {node.name:20s} | op={node.op:15s} | target={node.target}")

    print()

    # Evaluate with a sample input to verify correctness
    sample_input = torch.randn(1, 4)
    original_output = model(sample_input)
    traced_output = traced(sample_input)
    print(f"Original model output: {original_output.detach().numpy()}")
    print(f"Traced model output:   {traced_output.detach().numpy()}")
    print(f"Outputs match: {torch.allclose(original_output, traced_output)}")
    print()


def build_mlp_with_our_ir() -> None:
    """Build the same MLP using our graph/ir.py framework.

    TODO(human): Build an equivalent computation graph using our IR.

    The SimpleMLP does: output = Linear2(ReLU(Linear1(x)))
    Where Linear(x) = x @ weight.T + bias (matrix multiply + bias add)

    Since our IR uses scalar operations (not tensors), we'll simplify:
    Pick ONE element from the input and trace the scalar computation path.

    For a single scalar input x going through a simplified 2-layer network:
        h = relu(w1 * x + b1)     # "layer 1" for one neuron
        y = w2 * h + b2           # "layer 2" for one neuron

    Steps:
    1. Import Graph, Op from graph.ir
    2. Import evaluate from graph.evaluator
    3. Create a Graph instance
    4. Add INPUT node for "x"
    5. Add CONST nodes for weights and biases: "w1"=0.5, "b1"=-0.1, "w2"=0.3, "b2"=0.2
    6. Build the computation:
       - mul1 = add_op(MUL, "mul1", [x, w1])        # w1 * x
       - add1 = add_op(ADD, "add1", [mul1, b1])      # w1*x + b1
       - relu1 = add_op(RELU, "relu1", [add1])       # relu(w1*x + b1)
       - mul2 = add_op(MUL, "mul2", [relu1, w2])     # w2 * relu(...)
       - out = add_op(ADD, "out", [mul2, b2])         # w2*relu(...) + b2
    7. Evaluate with x=1.0 using evaluate(graph, {"x": 1.0})
    8. Print the graph and results

    Then compare the STRUCTURE (not values) with the torch.fx output above:
    - Both have: input → multiply → add → relu → multiply → add → output
    - torch.fx uses call_module nodes (referencing nn.Linear, nn.ReLU)
    - Our IR uses explicit MUL, ADD, RELU operations
    - torch.fx captures the full tensor computation; ours is scalar

    This comparison shows that the FUNDAMENTAL STRUCTURE is the same —
    the difference is the level of abstraction. ML compilers work by
    lowering the high-level graph (like torch.fx) into lower-level
    operations (like our IR), optimizing, and then generating code.
    """
    # TODO(human): Build the MLP computation graph using our IR and evaluate it
    print("\n" + "=" * 60)
    print("  Our IR: Scalar MLP Equivalent")
    print("=" * 60)
    print("  (Not yet implemented — complete the TODO in fx_compare.py)")
    print()


def main() -> None:
    """Run the torch.fx comparison demo."""
    print("\n" + "#" * 60)
    print("  Phase 5: torch.fx Comparison")
    print("#" * 60)

    trace_and_print_fx_graph()
    build_mlp_with_our_ir()


if __name__ == "__main__":
    main()
