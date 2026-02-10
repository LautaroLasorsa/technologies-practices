"""Main entry point — runs all 5 phases of the computation graph practice.

Phase 1: Build computation graph data structures
Phase 2: Forward evaluation
Phase 3: Backward pass (reverse-mode autodiff)
Phase 4: Graph optimization passes (constant folding + DCE)
Phase 5: torch.fx comparison
"""

from graph.backward import backward
from graph.evaluator import evaluate
from graph.fx_compare import main as fx_main
from graph.ir import Graph, Op
from graph.passes import constant_fold, dead_code_elimination


# ─────────────────────────────────────────────────────────
# Phase 1: Build a computation graph
# ─────────────────────────────────────────────────────────

def run_phase1() -> Graph:
    """Build the graph: y = relu(x1 * x2 + 3.0)

    This is a simple expression that exercises:
    - Two inputs (x1, x2)
    - One constant (3.0)
    - Three operations (MUL, ADD, RELU)

    The resulting DAG:
        x1 ──┐
              ├── mul ──┐
        x2 ──┘         ├── add ── relu (output)
              bias(3.0)─┘
    """
    print("\n" + "#" * 60)
    print("  Phase 1: Computation Graph Data Structures")
    print("#" * 60)

    g = Graph()

    # Leaf nodes: inputs and constants
    x1 = g.add_input("x1")
    x2 = g.add_input("x2")
    bias = g.add_const("bias", 3.0)

    # Operation nodes: build the expression bottom-up
    mul = g.add_op(Op.MUL, "mul", [x1, x2])
    add = g.add_op(Op.ADD, "add", [mul, bias])
    relu = g.add_op(Op.RELU, "relu", [add])

    g.print_graph("y = relu(x1 * x2 + 3.0)")

    print("  Graph built successfully!")
    print(f"  Nodes: {[n.name for n in g.nodes]}")
    print(f"  Output node: {relu}")
    print()

    return g


# ─────────────────────────────────────────────────────────
# Phase 2: Forward evaluation
# ─────────────────────────────────────────────────────────

def run_phase2(g: Graph) -> dict[str, float]:
    """Evaluate the graph with x1=2.0, x2=5.0.

    Expected computation:
        x1 = 2.0
        x2 = 5.0
        bias = 3.0
        mul = 2.0 * 5.0 = 10.0
        add = 10.0 + 3.0 = 13.0
        relu = max(0, 13.0) = 13.0
    """
    print("\n" + "#" * 60)
    print("  Phase 2: Forward Evaluation")
    print("#" * 60)

    inputs = {"x1": 2.0, "x2": 5.0}
    print(f"\n  Inputs: {inputs}")

    results = evaluate(g, inputs)

    print("\n  Results:")
    for name, value in results.items():
        print(f"    {name:15s} = {value}")

    # Show the graph with values filled in
    g.print_graph("After forward evaluation")

    # Verify expected results
    expected_relu = max(0.0, 2.0 * 5.0 + 3.0)
    actual_relu = results.get("relu")
    if actual_relu is not None:
        print(f"  Expected relu = {expected_relu}, Got = {actual_relu}")
        print(f"  Correct: {abs(actual_relu - expected_relu) < 1e-9}")
    else:
        print("  (relu not yet computed — implement evaluate() in evaluator.py)")
    print()

    return results


# ─────────────────────────────────────────────────────────
# Phase 3: Backward pass
# ─────────────────────────────────────────────────────────

def run_phase3(g: Graph) -> dict[str, float]:
    """Compute gradients of relu output w.r.t. all nodes.

    For y = relu(x1 * x2 + 3.0) with x1=2.0, x2=5.0:

    Forward values:
        mul = 10.0, add = 13.0, relu = 13.0

    Backward (dL/d... where L = relu output):
        dL/d(relu) = 1.0  (seed)
        dL/d(add) = 1.0   (relu passes through because add > 0)
        dL/d(mul) = 1.0   (ADD passes gradient to both inputs)
        dL/d(bias) = 1.0  (ADD passes gradient to both inputs)
        dL/d(x1) = 1.0 * x2 = 5.0  (MUL: grad * other_input)
        dL/d(x2) = 1.0 * x1 = 2.0  (MUL: grad * other_input)
    """
    print("\n" + "#" * 60)
    print("  Phase 3: Backward Pass (Reverse-Mode Autodiff)")
    print("#" * 60)

    # Find the output (relu) node
    relu_node = g.find_node("relu")
    if relu_node is None:
        print("  ERROR: 'relu' node not found. Did Phase 1 run correctly?")
        return {}

    grads = backward(g, relu_node)

    print("\n  Gradients (dL/d_node where L = relu output):")
    for name, grad in grads.items():
        print(f"    dL/d({name:10s}) = {grad}")

    # Verify expected gradients
    expected = {"x1": 5.0, "x2": 2.0, "bias": 1.0}
    print("\n  Expected gradients for inputs:")
    all_correct = True
    for name, expected_grad in expected.items():
        actual = grads.get(name, 0.0)
        correct = abs(actual - expected_grad) < 1e-9
        status = "OK" if correct else "WRONG"
        print(f"    dL/d({name}) = {actual} (expected {expected_grad}) [{status}]")
        if not correct:
            all_correct = False

    if all_correct:
        print("\n  All gradients correct!")
    else:
        print("\n  (Some gradients wrong — implement backward() in backward.py)")
    print()

    return grads


# ─────────────────────────────────────────────────────────
# Phase 4: Graph optimization passes
# ─────────────────────────────────────────────────────────

def run_phase4_constant_folding() -> None:
    """Demonstrate constant folding on a graph with known constants.

    Graph: y = x + (2.0 * 3.0)
    The (2.0 * 3.0) part is entirely constant → can be folded to 6.0.

    Before: x, const_2, const_3, mul, add  (5 nodes)
    After:  x, fold_mul(=6.0), add         (3 nodes, mul is folded)
    """
    print("\n  --- Constant Folding ---")

    g = Graph()
    x = g.add_input("x")
    c2 = g.add_const("const_2", 2.0)
    c3 = g.add_const("const_3", 3.0)
    mul = g.add_op(Op.MUL, "mul", [c2, c3])
    add = g.add_op(Op.ADD, "add", [x, mul])

    g.print_graph("Before constant folding: y = x + (2.0 * 3.0)")

    folded = constant_fold(g)
    folded.print_graph("After constant folding")

    print(f"  Nodes before: {len(g.nodes)}")
    print(f"  Nodes after:  {len(folded.nodes)}")

    # Check if folding actually happened
    has_fold = any("fold" in n.name for n in folded.nodes)
    if has_fold:
        print("  Constant folding detected! (found 'fold_' nodes)")
    else:
        print("  (No folding detected — implement constant_fold() in passes.py)")

    # Evaluate both graphs to verify they produce the same result
    original_result = evaluate(g, {"x": 10.0})
    folded_result = evaluate(folded, {"x": 10.0})
    print(f"\n  Original result (x=10): {original_result.get('add', 'N/A')}")
    print(f"  Folded result (x=10):   {folded_result.get('add', 'N/A')}")
    print()


def run_phase4_dce() -> None:
    """Demonstrate dead code elimination.

    Graph has two branches, but only one feeds into the output:
        x1 ──→ ADD ──→ output  (LIVE: feeds into output)
        x2 ──→ MUL ──→ dead    (DEAD: nobody uses this)
        x3 ──┘

    After DCE: x2, x3, MUL, dead are removed.
    """
    print("\n  --- Dead Code Elimination ---")

    g = Graph()
    x1 = g.add_input("x1")
    x2 = g.add_input("x2")
    x3 = g.add_input("x3")
    c = g.add_const("one", 1.0)

    # Live path: output = x1 + 1.0
    add = g.add_op(Op.ADD, "add_live", [x1, c])

    # Dead path: dead = x2 * x3 (result never used by output)
    mul = g.add_op(Op.MUL, "mul_dead", [x2, x3])

    g.print_graph("Before DCE: live path (x1+1) and dead path (x2*x3)")

    # The output is only the ADD node — the MUL path is dead
    cleaned = dead_code_elimination(g, outputs=[add])
    cleaned.print_graph("After DCE (only keeping 'add_live' output)")

    print(f"  Nodes before: {len(g.nodes)}")
    print(f"  Nodes after:  {len(cleaned.nodes)}")

    dead_removed = len(cleaned.nodes) < len(g.nodes)
    if dead_removed:
        print("  Dead code eliminated! Reduced graph size.")
    else:
        print("  (No elimination detected — implement dead_code_elimination() in passes.py)")
    print()


def run_phase4() -> None:
    """Run both optimization passes."""
    print("\n" + "#" * 60)
    print("  Phase 4: Graph Optimization Passes")
    print("#" * 60)

    run_phase4_constant_folding()
    run_phase4_dce()


# ─────────────────────────────────────────────────────────
# Main: run all phases
# ─────────────────────────────────────────────────────────

def main() -> None:
    """Run all phases sequentially."""
    print("\n" + "=" * 60)
    print("  ML Compilers 025a: Computation Graphs & IR from Scratch")
    print("=" * 60)

    # Phase 1: Build the graph
    graph = run_phase1()

    # Phase 2: Forward evaluation
    run_phase2(graph)

    # Phase 3: Backward pass
    run_phase3(graph)

    # Phase 4: Optimization passes
    run_phase4()

    # Phase 5: torch.fx comparison
    fx_main()

    print("\n" + "=" * 60)
    print("  All phases complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
