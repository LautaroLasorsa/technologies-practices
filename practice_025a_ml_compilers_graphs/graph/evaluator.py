"""Forward evaluation of computation graphs.

This module implements two key operations:

1. **Topological sort** — ordering nodes so that every node comes after its inputs.
   This is essential because you can't compute `x + y` before you know `x` and `y`.

2. **Forward evaluation** — walking the graph in topological order and computing
   each node's output value based on its operation and input values.

Together, these implement the "forward pass" — the same thing PyTorch does when
you call `model(input)`. The difference is that PyTorch builds the graph implicitly
(via operator overloading and autograd), while here we build it explicitly.

### Why topological sort?

A computation graph is a DAG (directed acyclic graph). In a DAG, there's always
at least one valid topological ordering — an ordering where every node appears
after all nodes it depends on.

Example graph: y = relu(x1 * x2 + 3.0)
    x1 ──┐
          ├── mul ──┐
    x2 ──┘         ├── add ── relu ── (output)
          const_3 ──┘

Valid topo order: [x1, x2, const_3, mul, add, relu]
Also valid:       [const_3, x2, x1, mul, add, relu]
Invalid:          [mul, x1, x2, ...]  (mul before its inputs!)

### Two classic algorithms for topological sort:

1. **Kahn's algorithm** (BFS-based):
   - Count in-degrees (number of inputs) for each node
   - Start with nodes that have in-degree 0 (INPUT, CONST)
   - Process them, decrement in-degree of their consumers
   - Repeat until all nodes are processed

2. **DFS-based**:
   - Run DFS from each unvisited node
   - Post-order (add node to result AFTER visiting all its inputs)
   - Reverse the result

Either works. Kahn's is more intuitive for graph problems.
DFS-based is more common in compiler implementations.
"""

from graph.ir import Graph, Node, Op


def topological_sort(graph: Graph) -> list[Node]:
    """Return nodes in topological order (inputs before their consumers).

    TODO(human): Implement topological sort using Kahn's algorithm or DFS.

    This is the foundational algorithm for graph evaluation. Without correct
    topological ordering, forward evaluation would try to use values that
    haven't been computed yet.

    ### Option A: Kahn's Algorithm (BFS-based, recommended)

    Kahn's algorithm is a BFS-style approach that processes nodes level by level:

    1. **Build an adjacency structure.** For each node, you need to know:
       - How many inputs it has (in-degree). Leaf nodes (INPUT, CONST) have in-degree 0.
       - Which nodes consume its output (so you can decrement their in-degree later).

       To build the "consumers" mapping, iterate over all nodes in graph.nodes.
       For each node, look at node.inputs — for each input node, record that
       the current node is a "consumer" of that input.

       Example: if node "add" has inputs ["mul", "const_3"], then:
         consumers["mul"] includes "add"
         consumers["const_3"] includes "add"

    2. **Initialize the queue** with all nodes that have in-degree 0.
       These are the "ready" nodes — they don't depend on anything.
       In our graph, these are always INPUT and CONST nodes.
       Use collections.deque for O(1) popleft.

    3. **Process the queue** (BFS loop):
       - Pop a node from the queue → add it to the result list.
       - For each consumer of this node:
         - Decrement the consumer's in-degree by 1.
         - If the consumer's in-degree reaches 0, add it to the queue.
           (All its inputs have been processed → it's ready.)

    4. **Return the result list.** It will be in topological order.

    If the result has fewer nodes than graph.nodes, there's a cycle (shouldn't
    happen in a well-formed computation graph, but worth asserting).

    ### Option B: DFS-based (alternative)

    1. Maintain a `visited` set and a `result` list.
    2. For each node in graph.nodes, if not visited:
       - Recursively visit all input nodes first (DFS into node.inputs).
       - After all inputs are visited, append this node to result.
    3. This gives a valid topological order (no reversal needed because we
       visit inputs BEFORE the node itself).

    ### Complexity

    Both algorithms are O(V + E) where V = number of nodes, E = number of edges.
    For our small graphs this doesn't matter, but it's worth knowing.

    Parameters:
        graph: The computation graph to sort.

    Returns:
        A list of Node objects in topological order.
        Leaf nodes (INPUT, CONST) appear first, output nodes appear last.
    """
    # TODO(human): implement topological sort
    # Stub: return nodes in insertion order (happens to be valid if graph was
    # built correctly, but you should implement a real topo sort)
    return list(graph.nodes)


def evaluate(graph: Graph, inputs: dict[str, float]) -> dict[str, float]:
    """Evaluate the graph in forward (topological) order.

    TODO(human): Implement forward evaluation.

    This is the forward pass — the same thing that happens when you call
    model(x) in PyTorch. Walk nodes in topological order, compute each
    node's value, and store it in node.value.

    Parameters:
        graph:  The computation graph to evaluate.
        inputs: A dictionary mapping INPUT node names to their runtime values.
                Example: {"x1": 2.0, "x2": 5.0}

    Returns:
        A dictionary mapping ALL node names to their computed values.
        Example: {"x1": 2.0, "x2": 5.0, "const_3": 3.0, "mul": 10.0, "add": 13.0, "relu": 13.0}

    Implementation steps:

    1. **Get topological order** by calling topological_sort(graph).

    2. **Walk nodes in topo order.** For each node, compute its value based on its op:

       - Op.INPUT:
         Look up node.name in the `inputs` dict. Store the value in node.value.
         If the name isn't in `inputs`, raise a ValueError — the caller forgot
         to provide a required input.

       - Op.CONST:
         node.value is already set (it was provided at graph-build time).
         No computation needed — just skip or confirm it's there.

       - Op.ADD:
         node.value = node.inputs[0].value + node.inputs[1].value
         Both inputs are guaranteed to have values because we're walking in
         topo order (inputs are always processed before their consumers).

       - Op.SUB:
         node.value = node.inputs[0].value - node.inputs[1].value

       - Op.MUL:
         node.value = node.inputs[0].value * node.inputs[1].value

       - Op.DIV:
         node.value = node.inputs[0].value / node.inputs[1].value
         (No need to handle division by zero for this practice.)

       - Op.NEG:
         node.value = -node.inputs[0].value

       - Op.RELU:
         node.value = max(0.0, node.inputs[0].value)
         ReLU is the most common activation function in deep learning.
         It simply clamps negative values to zero.

    3. **Build and return the results dict.**
       After all nodes are evaluated, create a dict mapping each node's name
       to its computed value: {node.name: node.value for node in topo_order}.

    Hint: Use a match/case statement on node.op for clean dispatching, or
    an if/elif chain — either works fine.
    """
    # TODO(human): implement forward evaluation
    # Stub: return only the provided inputs (nothing computed)
    return dict(inputs)
