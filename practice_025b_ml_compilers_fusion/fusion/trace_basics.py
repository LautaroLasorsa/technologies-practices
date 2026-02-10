"""
Phase 1: torch.fx Graph Basics — FULLY IMPLEMENTED reference.

This module demonstrates how torch.fx captures a PyTorch model into a graph IR
(Intermediate Representation) that you can inspect, analyze, and transform.

Key concepts:
  - symbolic_trace() converts an nn.Module into a GraphModule
  - A GraphModule is STILL a valid nn.Module (you can call it like normal)
  - The .graph attribute holds the IR as a list of Node objects
  - Each Node has: op, target, args, kwargs, name, users

Node types (node.op values):
  ┌──────────────────┬────────────────────────────────────────────────────┐
  │ placeholder       │ Function input (one per forward() parameter)      │
  │ get_attr          │ Reads a parameter/buffer from the module          │
  │ call_function     │ Calls a free function (torch.relu, operator.add)  │
  │ call_module       │ Calls a submodule's forward() (self.linear1)      │
  │ call_method       │ Calls a method on a tensor (x.reshape)            │
  │ output            │ The return value of the traced function            │
  └──────────────────┴────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.fx


# ---------------------------------------------------------------------------
# A simple model to trace
# ---------------------------------------------------------------------------
class SimpleModel(nn.Module):
    """Two linear layers with a ReLU in between.

    forward: x -> linear1 -> relu -> linear2 -> output
    """

    def __init__(self, in_features: int = 64, hidden: int = 128, out_features: int = 10):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


# ---------------------------------------------------------------------------
# Tracing and inspection
# ---------------------------------------------------------------------------
def trace_and_inspect(model: nn.Module) -> torch.fx.GraphModule:
    """Symbolically trace a model and print its graph IR."""

    # symbolic_trace runs the forward() with proxy objects instead of real
    # tensors. It records every operation the proxies flow through, building
    # a graph. The result is a GraphModule: an nn.Module whose forward() is
    # *generated from the graph* rather than from your Python source code.
    graph_module: torch.fx.GraphModule = torch.fx.symbolic_trace(model)

    # ---- 1. Print the generated Python code ----
    # GraphModule can show you the Python code it would generate from the graph.
    # This is useful to verify that tracing captured what you expect.
    print("=" * 70)
    print("GENERATED FORWARD CODE")
    print("=" * 70)
    print(graph_module.code)

    # ---- 2. Print the tabular IR ----
    # print_tabular() shows each node with its op, name, target, and args.
    # This is the canonical way to inspect an FX graph.
    print("=" * 70)
    print("GRAPH IR (tabular)")
    print("=" * 70)
    graph_module.graph.print_tabular()

    # ---- 3. Walk nodes manually ----
    # The graph is a doubly-linked list of Node objects in topological order.
    # Topological order means: if node A's output is used by node B,
    # then A appears before B in the iteration.
    print()
    print("=" * 70)
    print("NODE-BY-NODE WALKTHROUGH")
    print("=" * 70)

    for node in graph_module.graph.nodes:
        print(f"\n  Node name : {node.name}")
        print(f"  Op type   : {node.op}")
        print(f"  Target    : {node.target}")
        print(f"  Args      : {node.args}")
        print(f"  Kwargs    : {node.kwargs}")
        # node.users is a dict of {node: None} for all nodes that consume
        # this node's output. The length tells you how many downstream
        # consumers there are.
        print(f"  # Users   : {len(node.users)}")

    return graph_module


def verify_equivalence(original: nn.Module, traced: torch.fx.GraphModule) -> None:
    """Verify that the traced model produces the same output as the original."""
    x = torch.randn(4, 64)

    with torch.no_grad():
        y_original = original(x)
        y_traced = traced(x)

    # They should be identical (not just close — tracing doesn't change math)
    assert torch.equal(y_original, y_traced), "Traced model output differs!"
    print("\n[OK] Traced model output matches original exactly.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_phase() -> None:
    """Run Phase 1: trace a simple model and inspect its graph."""
    print("\n" + "=" * 70)
    print("  PHASE 1: torch.fx Graph Basics")
    print("=" * 70)

    model = SimpleModel(in_features=64, hidden=128, out_features=10)

    print(f"\nOriginal model:\n{model}\n")

    traced = trace_and_inspect(model)
    verify_equivalence(model, traced)

    # ---- Key takeaways ----
    print("\n" + "-" * 70)
    print("KEY TAKEAWAYS:")
    print("-" * 70)
    print("""
  1. symbolic_trace() captures forward() as a graph of Nodes.
  2. Each Node records: what operation (op), on what (target), with what inputs (args).
  3. The graph is in topological order — dependencies always come first.
  4. A GraphModule IS an nn.Module — you can call it, save it, train with it.
  5. The graph is MUTABLE — you can add, remove, and rewire nodes.
     This mutability is what makes fusion possible (Phase 2).
""")


if __name__ == "__main__":
    run_phase()
