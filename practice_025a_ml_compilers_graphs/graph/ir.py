"""Computation graph IR (Intermediate Representation).

This module defines the core data structures for a minimal computation graph:

- **Op**: An enum of operations (ADD, MUL, RELU, etc.) that nodes can perform.
- **Node**: A single node in the graph — it has an operation type, a name,
  a list of input nodes (edges in the DAG), and optional storage for
  a constant value and a computed gradient.
- **Graph**: A container that holds an ordered list of nodes and provides
  methods to add inputs, constants, and operations.

Think of it like an AST for math expressions, except:
- It's a DAG (nodes can be shared — e.g., x used in both x+y and x*z)
- It's flat (no nested expressions — every intermediate result is a named node)
- It's designed for forward evaluation AND backward differentiation

This flat, explicit representation is what ML compilers like XLA, TVM, and
torch.fx all use internally. The "graph" is the IR that optimization passes
operate on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class Op(Enum):
    """Operations that a computation graph node can represent.

    These map to the fundamental operations in neural network computation:
    - Arithmetic: ADD, SUB, MUL, DIV
    - Activation: RELU, NEG
    - Linear algebra: MATMUL (not used in this practice, but included for completeness)
    - Special: CONST (a known value), INPUT (a placeholder fed at runtime)
    """

    # --- Arithmetic operations ---
    ADD = auto()    # a + b  (binary)
    SUB = auto()    # a - b  (binary)
    MUL = auto()    # a * b  (binary)
    DIV = auto()    # a / b  (binary)

    # --- Unary operations ---
    NEG = auto()    # -a     (unary)
    RELU = auto()   # max(0, a)  (unary)

    # --- Linear algebra ---
    MATMUL = auto()  # matrix multiply (binary, included for completeness)

    # --- Leaf nodes (no computation, just hold or receive values) ---
    CONST = auto()   # a compile-time known constant (e.g., 3.0, bias vector)
    INPUT = auto()   # a runtime input placeholder (e.g., "x1", "x2")


@dataclass
class Node:
    """A single node in the computation graph.

    Each node represents either:
    - A leaf: an INPUT (fed at runtime) or a CONST (known at graph-build time)
    - An operation: takes 1+ input nodes and produces a result

    Attributes:
        op:     The operation this node performs (from the Op enum).
        name:   A unique human-readable name (e.g., "x1", "const_3.0", "add_0").
        inputs: List of nodes whose outputs feed into this node.
                Empty for INPUT and CONST nodes.
        value:  For CONST nodes, the compile-time known value (float).
                For other nodes, this is None at build time but gets filled
                during forward evaluation.
        grad:   The gradient of the loss with respect to this node's output.
                Filled during the backward pass. Initialized to 0.0.
    """

    op: Op
    name: str
    inputs: list[Node] = field(default_factory=list)
    value: float | None = None
    grad: float = 0.0

    def __repr__(self) -> str:
        input_names = [n.name for n in self.inputs]
        parts = [f"Node({self.op.name}, '{self.name}'"]
        if input_names:
            parts.append(f", inputs={input_names}")
        if self.value is not None:
            parts.append(f", value={self.value}")
        parts.append(")")
        return "".join(parts)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


class Graph:
    """A computation graph — an ordered collection of Nodes forming a DAG.

    The graph maintains nodes in insertion order. This is important because:
    - Nodes added later may depend on nodes added earlier (never the reverse)
    - Insertion order is a valid topological order IF you always add nodes
      after their inputs (which add_input, add_const, add_op enforce)

    Usage pattern:
        g = Graph()
        x1 = g.add_input("x1")           # leaf: runtime input
        x2 = g.add_input("x2")           # leaf: runtime input
        c  = g.add_const("bias", 3.0)    # leaf: known constant
        mul = g.add_op(Op.MUL, "mul", [x1, x2])  # x1 * x2
        add = g.add_op(Op.ADD, "add", [mul, c])   # (x1*x2) + 3.0
        out = g.add_op(Op.RELU, "out", [add])     # relu((x1*x2) + 3.0)
    """

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self._name_counter: dict[str, int] = {}

    def _unique_name(self, prefix: str) -> str:
        """Generate a unique name with the given prefix.

        If "add" is requested and already exists, returns "add_1", "add_2", etc.
        """
        count = self._name_counter.get(prefix, 0)
        self._name_counter[prefix] = count + 1
        if count == 0:
            return prefix
        return f"{prefix}_{count}"

    def add_input(self, name: str) -> Node:
        """Add an INPUT node — a placeholder for a runtime value.

        INPUT nodes are leaf nodes with no inputs and no compile-time value.
        Their value is provided when you call evaluate(graph, {"x1": 2.0, ...}).

        This is fully implemented as a reference for add_op().
        """
        node = Node(op=Op.INPUT, name=name)
        self.nodes.append(node)
        return node

    def add_const(self, name: str, value: float) -> Node:
        """Add a CONST node — a compile-time known constant.

        CONST nodes are leaf nodes with no inputs but a known value.
        They don't need runtime input — the value is baked into the graph.
        Constant folding (Phase 4) exploits these: if an operation's inputs
        are ALL constants, the operation itself can be evaluated at compile time.

        This is fully implemented as a reference for add_op().
        """
        node = Node(op=Op.CONST, name=name, value=value)
        self.nodes.append(node)
        return node

    def add_op(self, op: Op, name: str, inputs: list[Node]) -> Node:
        """Add an operation node to the graph.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches DAG construction — how edges are created by wiring
        # nodes to their inputs. Every ML compiler (XLA, TVM, torch.compile) has
        # this operation at its core: building a graph by adding nodes in topo order.

        TODO(human): Implement this method. It creates a new Node for the given
        operation, connects it to its input nodes, and appends it to the graph.

        This is the most important method in the Graph class — it's how you
        build the computation DAG. Every non-leaf node is created through this.

        Parameters:
            op:     The operation (e.g., Op.ADD, Op.MUL, Op.RELU).
                    Must NOT be Op.INPUT or Op.CONST — use add_input/add_const for those.
            name:   A human-readable name for this node (e.g., "mul", "relu_out").
                    Will be made unique via _unique_name() if collisions exist.
            inputs: List of Node objects whose outputs feed into this operation.
                    For binary ops (ADD, MUL, SUB, DIV): exactly 2 inputs.
                    For unary ops (NEG, RELU): exactly 1 input.
                    The caller is responsible for passing the right number of inputs.

        Returns:
            The newly created Node, already appended to self.nodes.

        Implementation steps:
        1. Generate a unique name using self._unique_name(name).
           This prevents collisions when you add multiple "add" or "mul" nodes.
           Example: first call with "add" -> "add", second -> "add_1", third -> "add_2".

        2. Create a new Node with:
           - op=op (the operation enum value)
           - name=<the unique name from step 1>
           - inputs=inputs (the list of input nodes — this creates the DAG edges)
           - value=None (not known until forward evaluation)
           - grad=0.0 (not known until backward pass)

        3. Append the new node to self.nodes.
           IMPORTANT: The node MUST be appended AFTER its inputs are already in the
           graph. This maintains the invariant that insertion order is a valid
           topological order. The caller ensures this by building the graph
           bottom-up (inputs first, operations second).

        4. Return the new node so the caller can use it as input to later operations.

        Example usage (after implementation):
            g = Graph()
            x = g.add_input("x")
            y = g.add_input("y")
            z = g.add_op(Op.ADD, "sum", [x, y])  # z = x + y
            # z is now a Node(ADD, 'sum', inputs=['x', 'y'])
            # g.nodes == [x, y, z]
        """
        # TODO(human): implement — create the node, append it, return it
        # Stub: return a placeholder node (graph won't work correctly until implemented)
        placeholder = Node(op=op, name=self._unique_name(name), inputs=inputs)
        return placeholder

    def find_node(self, name: str) -> Node | None:
        """Find a node by name. Returns None if not found."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def print_graph(self, title: str = "Graph") -> None:
        """Pretty-print the graph structure."""
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")
        for node in self.nodes:
            input_str = ", ".join(n.name for n in node.inputs) if node.inputs else "-"
            value_str = f"  val={node.value:.4f}" if node.value is not None else ""
            grad_str = f"  grad={node.grad:.4f}" if node.grad != 0.0 else ""
            print(f"  {node.name:15s} | {node.op.name:8s} | inputs=[{input_str}]{value_str}{grad_str}")
        print(f"{'=' * 50}")
        print(f"  Total nodes: {len(self.nodes)}")
        print()
