"""Phase 3: Graph Break Analysis — detect, understand, and fix graph breaks.

A "graph break" occurs when TorchDynamo encounters Python code it cannot capture
into the FX graph. When this happens, Dynamo:

1. Ends the current graph region at the break point
2. Falls back to Python interpreter for the unsupported code
3. Starts a NEW graph region after the break

This means instead of one big optimized graph, you get multiple smaller graphs with
Python interpreter execution between them. Each graph-to-Python-to-graph transition
has overhead, and the backend cannot optimize ACROSS graph breaks (no cross-region
fusion).

### Common causes of graph breaks

1. **print() and logging statements.** print() is pure Python I/O — Dynamo cannot
   represent it in the FX graph. Every print() in forward() causes a graph break.

   Fix: Remove print() calls, or guard them:
       if not torch.compiler.is_compiling():
           print(...)

2. **Data-dependent control flow.** If your code branches on the VALUE of a tensor
   (not just its shape), Dynamo cannot capture it because the graph must be valid
   for all possible tensor values:

       if x.sum() > 0:    # Graph break! Decision depends on tensor data
           return x
       else:
           return -x

   Fix: Use torch.where() or torch.cond() instead:
       return torch.where(x.sum() > 0, x, -x)

3. **Calling .item() or .tolist().** These extract Python scalars from tensors,
   breaking out of the tensor computation graph:

       threshold = x.max().item()    # Graph break!
       if threshold > 0.5:
           ...

   Fix: Keep computations in tensor land. Compare tensors directly:
       mask = x.max() > 0.5

4. **Unsupported Python constructs.** Some Python features cannot be traced:
   - Constructing new tensors from Python lists of tensor elements
   - Using Python random module
   - Complex string formatting with tensor values
   - try/except blocks that catch tensor-related exceptions

5. **Third-party library calls.** Calls to non-PyTorch libraries (numpy, scipy,
   custom C extensions) cause graph breaks because Dynamo only understands PyTorch
   operations.

   Fix: Use PyTorch equivalents (torch.linalg instead of numpy.linalg, etc.)

### Detecting graph breaks

Use torch._dynamo.explain() to find graph breaks:

    explanation = torch._dynamo.explain(model)(input)
    print(explanation.graph_break_count)
    print(explanation.break_reasons)

Or set the environment variable TORCH_LOGS="graph_breaks" to see breaks during
compilation.

### Why graph breaks matter for performance

Each graph break:
- Prevents cross-region operator fusion (the biggest optimization)
- Adds Python interpreter overhead between regions
- May trigger recompilation if guards change
- Makes the compilation cache less effective

A model with 0 graph breaks compiles to a single optimized kernel (or small set of
fused kernels). A model with 10 graph breaks compiles to 11 separate regions with
Python between each — potentially SLOWER than uncompiled eager mode due to overhead.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────
# Models WITH graph breaks (provided — don't modify these)
# ─────────────────────────────────────────────────────────

class BrokenModel_Print(nn.Module):
    """This model has a graph break caused by print().

    The print() statement inside forward() is pure Python I/O. TorchDynamo
    cannot represent it in the FX graph, so it must break the graph around it.

    Expected: 2 graph regions (before print, after print), 1 graph break.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        print(f"Intermediate shape: {x.shape}")  # <-- GRAPH BREAK
        x = self.fc2(x)
        return x


class BrokenModel_DataDependent(nn.Module):
    """This model has a graph break caused by data-dependent control flow.

    The `if x.sum() > 0` check depends on the actual tensor VALUES — Dynamo
    cannot know at compile time which branch to take, so it must break.

    Expected: graph breaks at the if-statement.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 64)
        self.scale_pos = nn.Linear(64, 32)
        self.scale_neg = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc(x))
        if x.sum().item() > 0:  # <-- GRAPH BREAK: .item() + Python if
            return self.scale_pos(x)
        else:
            return self.scale_neg(x)


class BrokenModel_ItemCall(nn.Module):
    """This model has a graph break caused by .item().

    Calling .item() extracts a Python scalar from a tensor, which forces Dynamo
    to exit the tensor computation graph.

    Expected: graph break at .item() call.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        norm_val = x.norm().item()  # <-- GRAPH BREAK: .item()
        x = x / (norm_val + 1e-6)  # Normalize using Python float
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class BrokenModel_ListComprehension(nn.Module):
    """This model has a graph break caused by building a tensor from a Python list.

    Constructing a new tensor from individual tensor elements via a Python list
    comprehension is not capturable by Dynamo.

    Expected: graph break at the list comprehension / tensor construction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(64, 64) for _ in range(4)])
        self.output = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply each linear and collect results in a Python list
        results = [linear(x) for linear in self.linears]  # This part may be OK
        # Stack into a tensor — this can cause breaks depending on Dynamo version
        stacked = torch.stack(results, dim=0)  # <-- Potential graph break
        x = stacked.mean(dim=0)
        x = self.output(x)
        return x


# ─────────────────────────────────────────────────────────
# Verification helpers (fully provided)
# ─────────────────────────────────────────────────────────

def check_graph_breaks(model: nn.Module, x: torch.Tensor, model_name: str) -> int:
    """Run torch._dynamo.explain() and report graph break count.

    Returns the number of graph breaks found.
    """
    torch._dynamo.reset()

    try:
        explanation = torch._dynamo.explain(model)(x)
        graph_count = explanation.graph_count
        break_count = explanation.graph_break_count

        print(f"\n  [{model_name}]")
        print(f"    Graph regions:  {graph_count}")
        print(f"    Graph breaks:   {break_count}")

        if break_count > 0:
            print(f"    Break reasons:")
            for i, reason in enumerate(explanation.break_reasons):
                # Truncate long reasons for readability
                reason_str = str(reason)
                if len(reason_str) > 120:
                    reason_str = reason_str[:120] + "..."
                print(f"      [{i}] {reason_str}")
        else:
            print(f"    Clean compile — no graph breaks!")

        return break_count

    except Exception as e:
        print(f"\n  [{model_name}] ERROR: {e}")
        return -1

    finally:
        torch._dynamo.reset()


def verify_fix(
    original_model: nn.Module,
    fixed_model: nn.Module,
    x: torch.Tensor,
    model_name: str,
) -> bool:
    """Verify that a fixed model produces the same output AND has no graph breaks.

    A valid fix must:
    1. Produce numerically close output to the original (within tolerance)
    2. Have 0 graph breaks when analyzed with explain()
    """
    print(f"\n  --- Verifying fix for {model_name} ---")

    # Check output equivalence
    with torch.no_grad():
        try:
            original_out = original_model(x)
        except Exception:
            # Some broken models may error; that's fine, just check the fix compiles
            original_out = None

        fixed_out = fixed_model(x)

    if original_out is not None:
        max_diff = (original_out - fixed_out).abs().max().item()
        output_ok = max_diff < 1e-4
        print(f"    Output match: {'PASS' if output_ok else 'FAIL'} (max_diff={max_diff:.6f})")
    else:
        output_ok = True
        print(f"    Output match: SKIP (original model errors)")

    # Check graph breaks
    break_count = check_graph_breaks(fixed_model, x, f"{model_name} (fixed)")
    breaks_ok = break_count == 0

    success = output_ok and breaks_ok
    print(f"    Overall: {'PASS' if success else 'FAIL'}")
    return success


# ─────────────────────────────────────────────────────────
# TODO(human): Implement fixed versions of the broken models
# ─────────────────────────────────────────────────────────

def run_analyze_breaks() -> None:
    """Analyze graph breaks in all broken models.

    This first part is straightforward: run explain() on each broken model
    to see what breaks and why. The helpers above do the heavy lifting.
    """
    print("\n  --- Analyzing graph breaks in broken models ---")

    x = torch.randn(4, 64)

    models: list[tuple[nn.Module, str]] = [
        (BrokenModel_Print().eval(), "BrokenModel_Print"),
        (BrokenModel_DataDependent().eval(), "BrokenModel_DataDependent"),
        (BrokenModel_ItemCall().eval(), "BrokenModel_ItemCall"),
        (BrokenModel_ListComprehension().eval(), "BrokenModel_ListComprehension"),
    ]

    total_breaks = 0
    for model, name in models:
        breaks = check_graph_breaks(model, x, name)
        if breaks > 0:
            total_breaks += breaks

    print(f"\n  Total graph breaks across all models: {total_breaks}")
    if total_breaks > 0:
        print("  Your task: fix each model to eliminate all graph breaks!")
    else:
        print("  No breaks found! (Some models may not break on newer PyTorch versions.)")


def run_fix_graph_breaks() -> None:
    """Fix graph breaks in the broken models.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches identifying and fixing graph breaks — the #1 issue when
    # adopting torch.compile. Understanding common break causes (print, .item(), control
    # flow) and their fixes is essential for production torch.compile usage.

    TODO(human): Implement fixed versions of each broken model.

    For EACH broken model, you need to create a fixed version that:
    - Produces the same (or equivalent) output
    - Has ZERO graph breaks when analyzed with torch._dynamo.explain()

    ### Fix 1: BrokenModel_Print

    The fix is simple: remove the print() statement, or guard it so it only
    runs outside of compilation:

        if not torch.compiler.is_compiling():
            print(...)

    `torch.compiler.is_compiling()` returns True when the code is being traced
    by TorchDynamo, and False during normal eager execution. This lets you keep
    debug prints that only fire in non-compiled mode.

    Create a `FixedModel_Print` class that is identical to BrokenModel_Print
    but without the graph-breaking print().

    ### Fix 2: BrokenModel_DataDependent

    Replace the data-dependent if/else with `torch.where()`:

        # Instead of:
        if x.sum().item() > 0:
            return self.scale_pos(x)
        else:
            return self.scale_neg(x)

        # Use:
        pos_out = self.scale_pos(x)
        neg_out = self.scale_neg(x)
        condition = x.sum() > 0  # Keep as tensor, don't call .item()!
        return torch.where(condition, pos_out, neg_out)

    Note: This computes BOTH branches always (less efficient if branches are
    expensive), but it eliminates the graph break and allows full optimization.

    Create a `FixedModel_DataDependent` class with the torch.where() approach.

    ### Fix 3: BrokenModel_ItemCall

    Replace the .item() extraction with pure tensor operations:

        # Instead of:
        norm_val = x.norm().item()
        x = x / (norm_val + 1e-6)

        # Use:
        norm_val = x.norm()        # Keep as tensor!
        x = x / (norm_val + 1e-6)  # Tensor division, no Python float

    The key insight: there's no need to extract a Python float. Tensor division
    by a scalar tensor works the same way and stays within the graph.

    Create a `FixedModel_ItemCall` class with pure tensor operations.

    ### Fix 4: BrokenModel_ListComprehension

    The list comprehension over nn.ModuleList may or may not break depending on
    PyTorch version. If it does break, the fix is to use torch.stack() directly
    on the results without going through a Python list:

        # Option A: Use a loop that accumulates into a tensor
        result = self.linears[0](x).unsqueeze(0)
        for linear in self.linears[1:]:
            result = torch.cat([result, linear(x).unsqueeze(0)], dim=0)
        x = result.mean(dim=0)

        # Option B (simpler, often works): The list comprehension + torch.stack
        # may actually work fine in recent PyTorch versions (2.2+). In that case,
        # just ensure there are no other issues.

    Create a `FixedModel_ListComprehension` class if the original breaks, or
    note that it compiles cleanly if it does.

    ### Putting it together

    After creating all fixed model classes, for each pair (broken, fixed):

    1. Run `check_graph_breaks()` on the broken model to show the breaks
    2. Run `verify_fix(broken, fixed, x, name)` to confirm the fix works

    Print a summary at the end showing which fixes succeeded.

    ### Key insight

    The pattern is always the same: replace Python-level operations (print, if,
    .item(), Python lists) with PyTorch-level equivalents that Dynamo can capture.
    Think of it as: "if it's not a torch operation, Dynamo can't see it."
    """
    # TODO(human): implement fixed models and verification
    # Stub: print a placeholder message
    print("\n  [STUB] run_fix_graph_breaks() not yet implemented.")
    print("  Implement the TODO above to fix graph breaks in the broken models.")
    print("  Create FixedModel_Print, FixedModel_DataDependent, etc.")


# ─────────────────────────────────────────────────────────
# Phase runner
# ─────────────────────────────────────────────────────────

def run_phase() -> None:
    """Run Phase 3: Graph Break Analysis."""
    print("\n" + "#" * 60)
    print("  Phase 3: Graph Break Analysis")
    print("#" * 60)
    print()

    # First: analyze the breaks
    run_analyze_breaks()

    # Then: fix them
    run_fix_graph_breaks()


if __name__ == "__main__":
    run_phase()
