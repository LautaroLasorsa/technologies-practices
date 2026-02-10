"""
Phase 2: Manual Fusion Pass — Walk the FX graph and fuse Linear+ReLU.

GOAL: Write a compiler pass that scans an FX graph for the pattern:
    call_module(linear) -> call_module(relu) or call_function(torch.relu)
  and replaces it with a single FusedLinearReLU module.

WHY THIS MATTERS:
  When Linear and ReLU are separate operations:
    1. Linear runs a CUDA kernel: reads input from memory, computes W@x+b, writes output to memory
    2. ReLU runs ANOTHER kernel: reads that output back from memory, applies max(0,x), writes again
  That intermediate write+read is pure waste — the data doesn't need to touch global memory.

  A fused kernel does: read input -> compute W@x+b -> apply max(0,result) -> write final output.
  One read, one write, instead of two reads and two writes. On memory-bound workloads (which
  most DL inference is), this can be a 1.5-2x speedup for that pair of operations.

GRAPH REWRITING MECHANICS:
  torch.fx graphs are mutable linked lists of Node objects. To fuse two nodes:
    1. Find the pattern (Linear node followed by ReLU node that consumes it)
    2. Create a new fused module and register it on the GraphModule
    3. Insert a new node that calls the fused module
    4. Redirect all consumers of the ReLU node to use the fused node instead
    5. Remove the now-dead ReLU and Linear nodes

  CRITICAL: You must walk the graph in topological order (which graph.nodes already is)
  and collect matches BEFORE modifying the graph. Modifying nodes while iterating
  can corrupt the linked list or miss patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx
from torch.fx import GraphModule, Node


# ---------------------------------------------------------------------------
# Fused module — combines Linear + ReLU into one forward()
# ---------------------------------------------------------------------------
class FusedLinearReLU(nn.Module):
    """A module that performs Linear + ReLU in a single forward call.

    In a real compiler, this wouldn't just call them sequentially in Python —
    it would dispatch to a single fused CUDA kernel (e.g., cuBLAS + activation
    fusion, or a custom Triton kernel). For this practice, we simulate the
    fusion at the graph level; the actual kernel fusion would be the backend's job.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x))


# ---------------------------------------------------------------------------
# Model to fuse
# ---------------------------------------------------------------------------
class StackedLinearReLU(nn.Module):
    """A model with multiple Linear+ReLU pairs — perfect fusion target."""

    def __init__(self, in_features: int = 64, hidden: int = 128, out_features: int = 10):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden, out_features)
        # Note: no ReLU after linear3 — the pass should NOT fuse this one

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


# ---------------------------------------------------------------------------
# TODO(human): Implement the fusion pass
# ---------------------------------------------------------------------------
def find_linear_relu_pairs(graph_module: GraphModule) -> list[tuple[Node, Node]]:
    """Find all (linear_node, relu_node) pairs that can be fused.

    TODO(human): Implement this function.

    Walk the graph nodes in topological order and find pairs where:
      - A node is a call_module whose target resolves to an nn.Linear
      - The ONLY user of that node is a call_module whose target resolves to nn.ReLU
        (or a call_function with target torch.relu / torch.nn.functional.relu)

    DETAILED GUIDANCE:

    1. ITERATING NODES:
       The graph.nodes property gives you nodes in topological order. This means
       if node A feeds into node B, A comes before B. You can iterate with:
           for node in graph_module.graph.nodes:

    2. CHECKING NODE TYPE:
       Each node has a .op attribute that tells you what kind of operation it is:
         - "call_module" means it calls self.<something> (e.g., self.linear1)
         - "call_function" means it calls a free function (e.g., torch.relu)
       And a .target attribute:
         - For call_module: target is a string like "linear1" or "relu1"
         - For call_function: target is the actual function (e.g., torch.relu)

    3. RESOLVING MODULE TYPE:
       When node.op == "call_module", node.target is a string (the module path).
       To get the actual module object, use:
           module = graph_module.get_submodule(node.target)
       Then check:  isinstance(module, nn.Linear)

    4. CHECKING SINGLE USER:
       node.users is a dict of {downstream_node: None}. To check that a Linear
       node's output is consumed ONLY by a ReLU (and nothing else), verify:
           len(linear_node.users) == 1
       If the Linear output is used by multiple nodes (e.g., a skip connection),
       we CANNOT fuse because fusing would change the semantics.

    5. CHECKING THE CONSUMER IS RELU:
       Get the single user node:
           relu_candidate = next(iter(linear_node.users))
       Then check if it's a ReLU, which could be either:
         - call_module targeting an nn.ReLU instance, OR
         - call_function with target torch.relu or F.relu

    6. COLLECTING PAIRS:
       Collect all valid (linear_node, relu_node) tuples in a list and return it.
       Do NOT modify the graph during this scan — collect first, modify later.
       This is important because modifying the graph while iterating can corrupt
       the node linked list or cause you to skip/revisit nodes.

    Returns:
        List of (linear_node, relu_node) pairs eligible for fusion.
    """
    # STUB: returns empty list so the file runs without errors
    return []


def apply_linear_relu_fusion(graph_module: GraphModule) -> GraphModule:
    """Apply Linear+ReLU fusion to all eligible pairs in the graph.

    TODO(human): Implement this function.

    For each (linear_node, relu_node) pair found by find_linear_relu_pairs():
      1. Get the original nn.Linear module from the graph_module
      2. Create a FusedLinearReLU wrapping that Linear
      3. Register it as a new submodule on the graph_module
      4. Insert a new call_module node for the fused module
      5. Redirect all users of relu_node to use the new fused node
      6. Remove the now-dead relu_node and linear_node

    DETAILED GUIDANCE:

    1. GETTING THE LINEAR MODULE:
       Use the linear_node.target (a string like "linear1") to retrieve the
       actual nn.Linear module:
           linear_module = graph_module.get_submodule(linear_node.target)

    2. CREATING THE FUSED MODULE:
       Wrap it:
           fused_module = FusedLinearReLU(linear_module)

    3. REGISTERING ON GRAPH_MODULE:
       You need to add the fused module as an attribute on graph_module so that
       the new call_module node can reference it:
           fused_name = f"fused_{linear_node.target}_relu"
           setattr(graph_module, fused_name, fused_module)

    4. INSERTING THE NEW NODE:
       Use graph.inserting_before(relu_node) as a context manager to control
       where the new node appears in the graph:
           with graph_module.graph.inserting_before(relu_node):
               fused_node = graph_module.graph.call_module(
                   fused_name,           # target: the attribute name we just set
                   args=linear_node.args, # same input as the original linear
               )
       The new node takes the LINEAR's inputs (not the relu's), because the
       fused module does both operations.

    5. REDIRECTING USERS:
       Replace all uses of the relu_node's output with the fused_node's output:
           relu_node.replace_all_uses_with(fused_node)
       This rewires every downstream node that consumed relu_node to now consume
       fused_node instead. After this, relu_node has no users.

    6. REMOVING DEAD NODES:
       Now relu_node and linear_node are dead (no users). Remove them:
           graph_module.graph.erase_node(relu_node)   # must go first (it used linear_node)
           graph_module.graph.erase_node(linear_node)
       ORDER MATTERS: erase_node checks that the node has no users. Since
       relu_node used to consume linear_node, you must erase relu_node first.
       If you erase linear_node first, it will fail because relu_node still
       references it (even though we redirected relu_node's *output* users,
       relu_node itself still has linear_node in its args).

       Wait — actually, after replace_all_uses_with, relu_node has no users,
       so it's safe to erase. But linear_node still has relu_node as a user
       (relu_node.args includes linear_node). So: erase relu first, then linear.

    7. RECOMPILE:
       After modifying the graph, call:
           graph_module.graph.lint()    # validates the graph is well-formed
           graph_module.recompile()     # regenerates the forward() code from the graph
       lint() checks things like: no dangling references, valid node types, etc.
       recompile() is REQUIRED — without it, the GraphModule's forward() still
       uses the old generated code, not your modified graph.

    Returns:
        The modified GraphModule with fused nodes.
    """
    # STUB: returns the graph_module unmodified so the file runs without errors
    return graph_module


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_fusion(original: nn.Module, fused: GraphModule) -> None:
    """Verify that the fused model produces the same output as the original."""
    x = torch.randn(4, 64)
    with torch.no_grad():
        y_original = original(x)
        y_fused = fused(x)

    if torch.allclose(y_original, y_fused, atol=1e-6):
        print("[OK] Fused model output matches original.")
    else:
        max_diff = (y_original - y_fused).abs().max().item()
        print(f"[FAIL] Outputs differ! Max difference: {max_diff}")


def count_node_types(graph_module: GraphModule) -> dict[str, int]:
    """Count how many nodes of each op type are in the graph."""
    counts: dict[str, int] = {}
    for node in graph_module.graph.nodes:
        key = f"{node.op}:{node.target}" if node.op == "call_module" else node.op
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_phase() -> None:
    """Run Phase 2: manual Linear+ReLU fusion pass."""
    print("\n" + "=" * 70)
    print("  PHASE 2: Manual Linear+ReLU Fusion Pass")
    print("=" * 70)

    model = StackedLinearReLU()
    print(f"\nOriginal model:\n{model}\n")

    # Trace the model
    traced = torch.fx.symbolic_trace(model)

    print("Graph BEFORE fusion:")
    traced.graph.print_tabular()
    before_counts = count_node_types(traced)

    # Apply fusion
    fused = apply_linear_relu_fusion(traced)

    print("\nGraph AFTER fusion:")
    fused.graph.print_tabular()
    after_counts = count_node_types(fused)

    print(f"\nNode counts before: {before_counts}")
    print(f"Node counts after:  {after_counts}")

    # Verify correctness
    verify_fusion(model, fused)

    # Check if fusion actually happened
    pairs = find_linear_relu_pairs(torch.fx.symbolic_trace(model))
    if len(pairs) == 0:
        print("\n[HINT] find_linear_relu_pairs() returned 0 pairs.")
        print("       Implement the TODO(human) to find Linear->ReLU patterns!")
    else:
        print(f"\n[INFO] Found {len(pairs)} fusible Linear+ReLU pairs.")


if __name__ == "__main__":
    run_phase()
