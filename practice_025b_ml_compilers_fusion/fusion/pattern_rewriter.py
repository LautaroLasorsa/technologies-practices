"""
Phase 3: Pattern-Based Rewriting — Use torch.fx.subgraph_rewriter.replace_pattern().

GOAL: Instead of manually walking the graph and rewriting nodes (Phase 2), define
a "pattern" function and a "replacement" function, and let torch.fx match and
replace for you. This is how production compilers define fusion rules — as
declarative pattern/replacement pairs, not imperative graph walks.

HOW PATTERN-BASED REWRITING WORKS:

  The torch.fx.subgraph_rewriter module provides replace_pattern(gm, pattern, replacement),
  which does the following:

  1. TRACE THE PATTERN: It calls symbolic_trace on your `pattern` function to get
     a small FX graph representing the subgraph you want to find. For example, if your
     pattern function does `linear(x)` then `relu(result)`, the pattern graph has nodes:
       placeholder(x) -> call_function(linear, x) -> call_function(relu, ...) -> output

  2. TRACE THE REPLACEMENT: Similarly, it traces `replacement` to get the graph that
     should replace each match. The replacement function must have the same signature
     (same number of parameters) as the pattern function.

  3. SUBGRAPH ISOMORPHISM: It searches the target graph (gm) for subgraphs that are
     *isomorphic* to the pattern graph. "Isomorphic" means: same sequence of operations
     with the same data-flow structure. It does NOT match by module names — it matches
     by the *structure* of operations.

  4. REPLACE: For each match, it splices in the replacement subgraph, rewiring inputs
     and outputs to maintain the original data flow.

WHAT IS A PATTERN FUNCTION?

  A pattern function is a plain Python function whose body describes the subgraph you
  want to find. torch.fx traces it to get the pattern graph. Example:

      def pattern(x, weight, bias):
          linear_out = torch.nn.functional.linear(x, weight, bias)
          return torch.nn.functional.relu(linear_out)

  This says: "find any place in the graph where F.linear feeds into F.relu".

  IMPORTANT: Pattern functions work with call_function nodes (torch.nn.functional.*),
  NOT call_module nodes (self.linear1). This is because symbolic_trace on a plain
  function produces call_function nodes. If your model uses nn.Module submodules
  (which trace as call_module), you need to first "decompose" them into functional
  form, or use the functional API in your model.

WHAT IS A REPLACEMENT FUNCTION?

  A replacement function has the SAME signature as the pattern function and returns the
  same number of outputs, but implements the fused/optimized version. Example:

      def replacement(x, weight, bias):
          return fused_linear_relu(x, weight, bias)

  The replacement is traced the same way. After matching, every matched subgraph in the
  target is replaced by the replacement subgraph.

MATCHING SEMANTICS:

  - replace_pattern matches by OPERATION STRUCTURE, not by variable names or module names.
  - The pattern's placeholder parameters are "wildcards" — they match any input.
  - Operations must match exactly: if the pattern has F.linear followed by F.relu, it will
    NOT match F.linear followed by torch.sigmoid.
  - Multiple matches in the same graph are all replaced (non-overlapping).
  - The return value is a list of ReplacedPatterns objects describing what was replaced.

WHY THIS IS BETTER THAN MANUAL REWRITING:

  1. DECLARATIVE: You say WHAT to find and WHAT to replace it with, not HOW to walk the graph.
  2. COMPOSABLE: You can define a library of pattern/replacement pairs and apply them all.
  3. CORRECT: The framework handles node rewiring, dead code elimination, and graph integrity.
  4. MAINTAINABLE: Adding a new fusion rule = adding one pattern + one replacement function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx
from torch.fx import GraphModule


# ---------------------------------------------------------------------------
# Custom fused operation (simulates what a real backend kernel would do)
# ---------------------------------------------------------------------------
def fused_linear_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Simulate a fused Linear+ReLU kernel.

    In a real compiler backend, this would dispatch to a single CUDA kernel that
    computes matmul + bias + relu without writing intermediate results to global
    memory. Here we just call both ops in Python — the graph-level fusion is
    what we're practicing.
    """
    return F.relu(F.linear(x, weight, bias))


# ---------------------------------------------------------------------------
# Model that uses functional API (important for pattern matching!)
# ---------------------------------------------------------------------------
class FunctionalLinearReLUModel(nn.Module):
    """A model that uses F.linear + F.relu instead of nn.Linear + nn.ReLU modules.

    WHY FUNCTIONAL?
    Pattern-based rewriting with replace_pattern works on call_function nodes.
    When you trace a model that uses nn.Linear (a module), torch.fx records it
    as a call_module node. But when you trace a plain function that uses F.linear,
    it records call_function nodes. For replace_pattern to match, both the model
    and the pattern must use the same call style.

    In practice, you would either:
      (a) Use a functional model (like this one), or
      (b) Write a pre-pass that converts call_module nodes to call_function nodes
          (this is what torch.compile's decomposition passes do).

    For this practice, we use approach (a) to keep things simple.
    """

    def __init__(self, in_features: int = 64, hidden: int = 128, out_features: int = 10):
        super().__init__()
        # Store weights as parameters (not as nn.Linear modules)
        self.weight1 = nn.Parameter(torch.randn(hidden, in_features))
        self.bias1 = nn.Parameter(torch.randn(hidden))
        self.weight2 = nn.Parameter(torch.randn(hidden, hidden))
        self.bias2 = nn.Parameter(torch.randn(hidden))
        self.weight3 = nn.Parameter(torch.randn(out_features, hidden))
        self.bias3 = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1: linear + relu (should be matched and fused)
        x = F.linear(x, self.weight1, self.bias1)
        x = F.relu(x)
        # Layer 2: linear + relu (should be matched and fused)
        x = F.linear(x, self.weight2, self.bias2)
        x = F.relu(x)
        # Layer 3: linear only (no relu — should NOT be fused)
        x = F.linear(x, self.weight3, self.bias3)
        return x


# ---------------------------------------------------------------------------
# TODO(human): Define the pattern and replacement functions, then apply rewriting
# ---------------------------------------------------------------------------
def linear_relu_pattern(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Define the PATTERN to search for: F.linear followed by F.relu.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches declarative pattern matching — defining what to find
    # as a function that gets traced into a pattern graph. This is how production
    # compilers (Inductor, XLA) define fusion rules at scale.

    TODO(human): Implement this function.

    This function will be symbolically traced by replace_pattern to produce the
    pattern graph. Your implementation should:
      1. Call F.linear(x, weight, bias) to produce the linear output
      2. Call F.relu on the linear output
      3. Return the relu output

    DETAILED GUIDANCE:

    When torch.fx traces this function, it does NOT execute real tensor math.
    Instead, it passes Proxy objects as x, weight, and bias. Each operation you
    perform on those proxies (F.linear, F.relu) is recorded as a Node in the
    pattern graph. The resulting pattern graph looks like:

        placeholder: x
        placeholder: weight
        placeholder: bias
        call_function: torch.nn.functional.linear(x, weight, bias)
        call_function: torch.nn.functional.relu(linear_out)
        output: relu_out

    This pattern graph is then matched against the model's graph using subgraph
    isomorphism. Every place in the model's graph where the same sequence of
    operations appears (F.linear feeding into F.relu) is identified as a match.

    The placeholders (x, weight, bias) act as WILDCARDS — they match any input
    node in the model's graph. So this pattern will match F.linear + F.relu
    regardless of what tensors are used as inputs.

    IMPORTANT NOTES:
    - The function signature defines the wildcards. You need exactly as many
      parameters as there are distinct inputs to the pattern subgraph.
    - F.linear takes (input, weight, bias), so we need 3 parameters.
    - The replacement function MUST have the exact same signature.
    - You must use the same functions (F.linear, F.relu) as the model uses,
      otherwise the operation types won't match.

    Returns:
        The output tensor after linear + relu.
    """
    # STUB: returns F.linear without relu so the file runs (but won't match the full pattern)
    return F.linear(x, weight, bias)


def linear_relu_replacement(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Define the REPLACEMENT: a fused linear+relu operation.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches defining optimized replacements for matched patterns.
    # The replacement graph becomes the fused operation in the final IR. Backend
    # code generators use these fused nodes to emit optimized kernels.

    TODO(human): Implement this function.

    This function will be symbolically traced by replace_pattern to produce the
    replacement graph. Every matched occurrence of the pattern in the model's
    graph will be replaced by this subgraph.

    Your implementation should:
      1. Call fused_linear_relu(x, weight, bias)
      2. Return the result

    DETAILED GUIDANCE:

    The replacement function must:
    - Have the SAME SIGNATURE as the pattern function (same parameter names and count).
      This is how replace_pattern knows how to wire the replacement's inputs to the
      original inputs in the model's graph.
    - Return the SAME NUMBER of outputs as the pattern function.
    - Use whatever operations you want for the replacement — here we use our custom
      fused_linear_relu function.

    When traced, this produces a replacement graph like:

        placeholder: x
        placeholder: weight
        placeholder: bias
        call_function: fused_linear_relu(x, weight, bias)
        output: fused_out

    Notice: 3 placeholders + 1 operation, vs the pattern's 3 placeholders + 2 operations.
    The pattern has F.linear + F.relu (2 nodes), but the replacement has just
    fused_linear_relu (1 node). This is the fusion!

    After replacement, the model's graph will have fused_linear_relu nodes where
    there used to be F.linear -> F.relu chains.

    WHY A CUSTOM FUNCTION?
    We use fused_linear_relu (defined at the top of this file) as the replacement
    operation. In production, this would point to a real fused CUDA kernel. For
    this practice, it just calls F.relu(F.linear(...)) — but at the graph level,
    it's a single node, which is what matters for compiler optimization. A backend
    code generator would see this single node and emit a fused kernel.

    Returns:
        The output tensor from the fused operation.
    """
    # STUB: returns F.linear without fusion so the file runs (replacement = pattern = no change)
    return F.linear(x, weight, bias)


def apply_pattern_rewrite(graph_module: GraphModule) -> tuple[GraphModule, int]:
    """Apply pattern-based rewriting to fuse Linear+ReLU in the graph.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches using torch.fx.subgraph_rewriter — the declarative API
    # for pattern matching and replacement. This approach scales to hundreds of
    # fusion rules without manual graph traversal code.

    TODO(human): Implement this function.

    Use torch.fx.subgraph_rewriter.replace_pattern() to find all occurrences
    of the linear_relu_pattern in graph_module and replace them with
    linear_relu_replacement.

    DETAILED GUIDANCE:

    1. IMPORT THE REWRITER:
       The function you need is:
           from torch.fx import subgraph_rewriter
       Or access it as torch.fx.subgraph_rewriter.replace_pattern.

    2. CALLING replace_pattern:
       The API is:
           matches = torch.fx.subgraph_rewriter.replace_pattern(
               gm=graph_module,           # the graph to modify (modified IN PLACE)
               pattern=linear_relu_pattern,  # the pattern function
               replacement=linear_relu_replacement,  # the replacement function
           )
       This returns a list of ReplacedPatterns objects, one per match found and replaced.

    3. WHAT replace_pattern DOES INTERNALLY:
       a) Traces `pattern` with symbolic_trace to get the pattern graph
       b) Traces `replacement` with symbolic_trace to get the replacement graph
       c) Searches `gm.graph` for subgraphs isomorphic to the pattern graph
       d) For each match: splices in the replacement graph, rewires inputs/outputs
       e) Returns list of matches (so you can count how many fusions happened)

    4. AFTER REPLACEMENT:
       The graph_module is modified in-place. You should call:
           graph_module.graph.lint()    # validate graph integrity
           graph_module.recompile()     # regenerate forward() from the modified graph
       Just like in manual fusion (Phase 2).

    5. RETURN VALUE:
       Return a tuple of (graph_module, number_of_matches).
       The number_of_matches is len(matches) from replace_pattern's return value.

    6. DEBUGGING TIPS:
       - If replace_pattern finds 0 matches, print the model's graph and the pattern
         graph side by side. Check that both use call_function (not call_module).
       - replace_pattern will NOT match call_module nodes against call_function patterns.
         This is why our model uses F.linear/F.relu instead of nn.Linear/nn.ReLU.
       - You can print the pattern graph with:
           pattern_gm = torch.fx.symbolic_trace(linear_relu_pattern)
           pattern_gm.graph.print_tabular()

    Returns:
        Tuple of (modified graph_module, number of replacements made).
    """
    # STUB: returns the graph_module unmodified with 0 matches
    return graph_module, 0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_rewrite(original: nn.Module, rewritten: GraphModule) -> None:
    """Verify that the rewritten model produces the same output as the original."""
    x = torch.randn(4, 64)
    with torch.no_grad():
        y_original = original(x)
        y_rewritten = rewritten(x)

    if torch.allclose(y_original, y_rewritten, atol=1e-6):
        print("[OK] Rewritten model output matches original.")
    else:
        max_diff = (y_original - y_rewritten).abs().max().item()
        print(f"[FAIL] Outputs differ! Max difference: {max_diff}")


def print_graph_comparison(before: GraphModule, after: GraphModule) -> None:
    """Print the graph before and after rewriting for comparison."""
    print("Graph BEFORE pattern rewrite:")
    before.graph.print_tabular()

    print("\nGraph AFTER pattern rewrite:")
    after.graph.print_tabular()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_phase() -> None:
    """Run Phase 3: pattern-based Linear+ReLU fusion."""
    print("\n" + "=" * 70)
    print("  PHASE 3: Pattern-Based Rewriting (replace_pattern)")
    print("=" * 70)

    model = FunctionalLinearReLUModel()
    print(f"\nOriginal model (functional style):\n{model}\n")

    # Trace the model
    traced = torch.fx.symbolic_trace(model)

    # Save a copy of the graph for comparison (print_tabular before modification)
    print("Graph BEFORE pattern rewrite:")
    traced.graph.print_tabular()

    # Count nodes before
    node_count_before = len(list(traced.graph.nodes))

    # Apply pattern-based rewriting
    rewritten, num_matches = apply_pattern_rewrite(traced)

    print(f"\nGraph AFTER pattern rewrite:")
    rewritten.graph.print_tabular()

    # Count nodes after
    node_count_after = len(list(rewritten.graph.nodes))

    print(f"\nNodes before: {node_count_before}")
    print(f"Nodes after:  {node_count_after}")
    print(f"Patterns matched and replaced: {num_matches}")

    if num_matches == 0:
        print("\n[HINT] apply_pattern_rewrite() returned 0 matches.")
        print("       Implement the TODO(human) functions:")
        print("         1. linear_relu_pattern()  — define the F.linear + F.relu pattern")
        print("         2. linear_relu_replacement() — define the fused replacement")
        print("         3. apply_pattern_rewrite() — call replace_pattern()")
    else:
        print(f"\n[INFO] Successfully fused {num_matches} Linear+ReLU pair(s).")

    # Verify correctness
    verify_rewrite(model, rewritten)

    # Show the generated code after rewriting
    print("\n" + "=" * 70)
    print("GENERATED FORWARD CODE (after rewrite)")
    print("=" * 70)
    print(rewritten.code)

    # Key takeaways
    print("-" * 70)
    print("KEY TAKEAWAYS:")
    print("-" * 70)
    print("""
  1. replace_pattern is DECLARATIVE — you define what to find and what to replace.
  2. Pattern functions are traced to produce a "template" subgraph for matching.
  3. Replacement functions must have the SAME signature as the pattern.
  4. This approach scales: define 100 pattern/replacement pairs, apply them all.
  5. Limitation: patterns match call_function nodes; call_module needs decomposition.
  6. Production compilers (TorchInductor, TVM) use this same approach at scale.
""")


if __name__ == "__main__":
    run_phase()
