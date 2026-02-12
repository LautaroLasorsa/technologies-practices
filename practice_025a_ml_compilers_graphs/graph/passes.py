"""Graph optimization passes — constant folding and dead code elimination.

These are two of the most fundamental compiler optimizations, used in EVERY
production ML compiler (XLA, TVM, TensorRT, torch.compile, ONNX Runtime).

### What is a "pass"?

A "pass" is a transformation that takes a graph, analyzes it, and produces
an optimized graph. Passes are composable — you can chain them:

    graph → constant_fold → dead_code_elimination → fused_graph

This is the same pattern used in LLVM (the compiler behind clang/rustc),
GCC, and every ML compiler.

### Pass 1: Constant Folding

If ALL inputs to an operation are constants (known at compile time), then
the operation itself can be evaluated at compile time and replaced with a
single CONST node.

Example:
    Before: const_2 (=2.0) → MUL ← const_3 (=3.0)
    After:  const_fold_0 (=6.0)

Why it matters in ML:
- Shape computations are often constant (e.g., batch_size * seq_len)
- Bias additions with known biases can be folded
- Reduces the graph size → less work at runtime

### Pass 2: Dead Code Elimination (DCE)

Nodes whose outputs are never used (directly or transitively) by any output
node are "dead code" and can be safely removed.

Example:
    x1 ──→ ADD ──→ output
    x2 ──→ MUL ──→ (nobody uses this!)

    After DCE: x2 and MUL are removed.

Why it matters in ML:
- After constant folding, some nodes may become unreachable
- Debug/logging nodes left in the graph waste compute
- Pruned model branches (e.g., unused attention heads) should be removed
"""

from graph.evaluator import evaluate, topological_sort
from graph.ir import Graph, Node, Op


def constant_fold(graph: Graph) -> Graph:
    """Evaluate operations on known constants at compile time.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches compiler optimization passes — transforming IR to equivalent
    # but faster form. Constant folding is used in EVERY compiler (LLVM, GCC, XLA, TVM).
    # It reduces runtime computation by doing work at compile time.

    TODO(human): Implement constant folding.

    This pass scans the graph for nodes where ALL inputs are CONST nodes.
    For each such node, it evaluates the operation and replaces the node
    with a new CONST node holding the computed value.

    Parameters:
        graph: The input computation graph.

    Returns:
        A NEW Graph with constant expressions replaced by CONST nodes.
        The original graph is not modified.

    Implementation steps:

    1. **Create a new Graph** for the output. You'll rebuild the graph,
       replacing foldable nodes with constants.

    2. **Create a node mapping** (dict[Node, Node]) to track which old nodes
       map to which new nodes. As you process old nodes and create new ones,
       store old_node → new_node in this mapping.

    3. **Walk the original graph in topological order.**
       For each node in topological_sort(graph):

       a. **If the node is INPUT:**
          Add it to the new graph with new_graph.add_input(node.name).
          Store the mapping: old_node → new_input_node.

       b. **If the node is CONST:**
          Add it to the new graph with new_graph.add_const(node.name, node.value).
          Store the mapping.

       c. **If the node is an operation (ADD, MUL, RELU, etc.):**
          Look up the NEW versions of this node's inputs using the mapping.
          (Each old input was already processed → its new version exists.)

          Check: are ALL of the new input nodes CONST nodes?
          (i.e., all of them have op == Op.CONST)

          - **YES → fold it!**
            Evaluate the operation using the constant values.
            Create a CONST node in the new graph with the computed value.

            To evaluate, apply the same logic as forward evaluation:
            - ADD: inputs[0].value + inputs[1].value
            - MUL: inputs[0].value * inputs[1].value
            - SUB: inputs[0].value - inputs[1].value
            - DIV: inputs[0].value / inputs[1].value
            - NEG: -inputs[0].value
            - RELU: max(0.0, inputs[0].value)

            Use a name like f"fold_{node.name}" to indicate it was folded.
            Store the mapping: old_node → new_const_node.

          - **NO → keep the operation as-is.**
            Create the operation node in the new graph using add_op(),
            passing the NEW versions of the inputs (from the mapping).
            Store the mapping.

    4. **Return the new graph.**

    Example walkthrough:
        Original: const_2(=2.0) → MUL ← const_3(=3.0) → ADD ← x1

        Processing const_2: CONST → add to new graph, map it
        Processing const_3: CONST → add to new graph, map it
        Processing x1:      INPUT → add to new graph, map it
        Processing MUL:     inputs are [const_2, const_3] → both CONST!
                            → evaluate: 2.0 * 3.0 = 6.0
                            → add CONST "fold_mul" with value 6.0
        Processing ADD:     inputs are [MUL→fold_mul(CONST), x1(INPUT)]
                            → NOT all CONST (x1 is INPUT)
                            → keep ADD, with inputs [fold_mul, x1]

    Result: fold_mul(=6.0) → ADD ← x1
    (One less operation at runtime!)

    Note: A more aggressive approach would iterate until no more folding is
    possible (fixed-point iteration). For this practice, a single pass is fine.
    """
    # TODO(human): implement constant folding
    # Stub: return a copy of the graph (no optimization)
    new_graph = Graph()
    node_map: dict[Node, Node] = {}
    for node in graph.nodes:
        if node.op == Op.INPUT:
            new_node = new_graph.add_input(node.name)
        elif node.op == Op.CONST:
            new_node = new_graph.add_const(node.name, node.value)
        else:
            new_inputs = [node_map[inp] for inp in node.inputs]
            new_node = new_graph.add_op(node.op, node.name, new_inputs)
        node_map[node] = new_node
    return new_graph


def dead_code_elimination(graph: Graph, outputs: list[Node]) -> Graph:
    """Remove nodes whose outputs are never used by any output node.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches reachability analysis — identifying live code via graph
    # traversal. DCE is fundamental to garbage collection, linker optimization, and
    # ML compiler pruning (removing unused model branches after quantization).

    TODO(human): Implement dead code elimination.

    DCE works backward from the output nodes: any node that transitively
    feeds into an output is "live" (needed). Everything else is "dead"
    and can be removed.

    Parameters:
        graph:   The input computation graph.
        outputs: The list of nodes whose values we actually need.
                 These are the "roots" of the liveness analysis.
                 Typically the loss node, or the model's output nodes.

    Returns:
        A NEW Graph containing only the live nodes.
        The original graph is not modified.

    Implementation steps:

    1. **Mark live nodes using BFS/DFS backward from outputs.**

       Create a set `live: set[Node]` and a worklist (queue or stack).
       Add all output nodes to both the `live` set and the worklist.

       While the worklist is not empty:
       - Pop a node from the worklist.
       - For each input node in node.inputs:
         - If the input is NOT already in `live`:
           - Add it to `live`.
           - Add it to the worklist (so we process its inputs too).

       After this BFS/DFS, `live` contains every node that transitively
       feeds into at least one output. Everything else is dead.

       This is the same "reachability analysis" used in garbage collectors
       and compiler liveness analysis.

    2. **Rebuild the graph with only live nodes.**

       Create a new Graph and a node mapping (like in constant_fold).
       Walk the ORIGINAL graph in topological order.
       For each node:
       - If the node is NOT in `live` → skip it (dead code!).
       - If the node IS in `live` → add it to the new graph:
         - INPUT: new_graph.add_input(node.name)
         - CONST: new_graph.add_const(node.name, node.value)
         - Operation: new_graph.add_op(node.op, node.name, [mapped inputs])
       Store the mapping for future reference.

    3. **Return the new graph** (smaller, with dead nodes removed).

    Example walkthrough:
        Original graph:
            x1 ──→ ADD ──→ output (live!)
            x2 ──→ MUL ──→ dead_node (nobody uses this)
            x3 ──→ SUB ──→ dead_node

        outputs = [output]  (only the ADD result matters)

        Liveness BFS from output:
            live = {output}
            output.inputs = [x1, ...] → live = {output, ADD, x1, ...}
            (x2, MUL, x3, SUB, dead_node never reached)

        Result: only x1, ADD, output remain. x2, x3, MUL, SUB removed.

    Note: The output nodes passed here must be nodes from the ORIGINAL graph.
    The mapping handles translating them to the new graph's nodes.
    """
    # TODO(human): implement dead code elimination
    # Stub: return a copy of the graph (no optimization)
    new_graph = Graph()
    node_map: dict[Node, Node] = {}
    for node in graph.nodes:
        if node.op == Op.INPUT:
            new_node = new_graph.add_input(node.name)
        elif node.op == Op.CONST:
            new_node = new_graph.add_const(node.name, node.value)
        else:
            new_inputs = [node_map[inp] for inp in node.inputs]
            new_node = new_graph.add_op(node.op, node.name, new_inputs)
        node_map[node] = new_node
    return new_graph
