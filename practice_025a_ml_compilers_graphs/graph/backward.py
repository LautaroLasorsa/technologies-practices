"""Backward pass — reverse-mode automatic differentiation.

This module implements the backward pass (backpropagation) through a
computation graph. This is the core algorithm behind PyTorch's autograd.

### What is reverse-mode autodiff?

Given a computation graph that computes a scalar loss L from inputs x1, x2, ...,
we want to compute dL/dx1, dL/dx2, etc. — the gradient of the loss with
respect to each input.

**Reverse-mode** means we start from the output (loss) and propagate gradients
backward through the graph. This is efficient when you have many inputs
(parameters) and one output (loss) — exactly the ML training case.

**Forward-mode** would propagate derivatives from inputs to outputs, which is
efficient when you have one input and many outputs (rare in ML).

### The chain rule on graphs

For each node, we know:
- The node's output gradient (dL/d_node), accumulated from all consumers
- The local derivative of the node's operation

We multiply these (chain rule) to get the gradient for each input.

Example: node z = x * y
- dL/dx = dL/dz * dz/dx = dL/dz * y
- dL/dy = dL/dz * dz/dy = dL/dz * x

### Gradient rules for each operation

| Op   | Formula           | dL/d(input_0)        | dL/d(input_1)        |
|------|-------------------|----------------------|----------------------|
| ADD  | z = a + b         | dL/dz * 1 = dL/dz   | dL/dz * 1 = dL/dz   |
| SUB  | z = a - b         | dL/dz * 1 = dL/dz   | dL/dz * (-1)         |
| MUL  | z = a * b         | dL/dz * b            | dL/dz * a            |
| DIV  | z = a / b         | dL/dz * (1/b)        | dL/dz * (-a/b^2)     |
| NEG  | z = -a            | dL/dz * (-1)         | n/a                  |
| RELU | z = max(0, a)     | dL/dz if a > 0 else 0| n/a                  |

Note: For RELU, the gradient is either passed through (when input > 0) or
killed (when input <= 0). This is the "dying ReLU" problem in deep learning.
"""

from graph.evaluator import topological_sort
from graph.ir import Graph, Node, Op


def backward(graph: Graph, loss_node: Node) -> dict[str, float]:
    """Compute gradients via reverse-mode autodiff.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches backpropagation — the algorithm that makes deep learning
    # training possible. This is PyTorch autograd's core. Understanding reverse-mode
    # autodiff is essential for ML systems work and autodiff-based optimization (JAX).

    TODO(human): Implement the backward pass.

    This is backpropagation — the algorithm that makes deep learning training
    possible. It computes dL/d(node) for every node in the graph, starting
    from the loss node and working backward.

    Parameters:
        graph:     The computation graph (must already be forward-evaluated,
                   i.e., every node must have a .value set).
        loss_node: The output node representing the loss. Its gradient is 1.0
                   (dL/dL = 1 by definition).

    Returns:
        A dictionary mapping node names to their gradients.
        Example: {"x1": 5.0, "x2": 2.0, "bias": 1.0, ...}

    Implementation steps:

    1. **Initialize all gradients to 0.0.**
       Set node.grad = 0.0 for every node in graph.nodes.
       Then set loss_node.grad = 1.0 — the seed gradient.
       (dL/dL = 1.0 by the identity rule of calculus.)

    2. **Get reverse topological order.**
       Call topological_sort(graph) and reverse it.
       We need to process nodes from output → inputs so that when we process
       a node, we already know its accumulated gradient from all consumers.

       Why reversed? In the forward pass, we go inputs → outputs.
       In the backward pass, gradients flow outputs → inputs.

    3. **Walk nodes in reverse topo order.** For each node:

       Skip leaf nodes (INPUT, CONST) — they don't propagate gradients further.
       (Their .grad values are the final answer — the gradient of the loss
       with respect to that input/constant.)

       For operation nodes, apply the chain rule based on the op type:

       - **Op.ADD** (z = a + b):
         The derivative of addition is 1 for both inputs.
         input_0.grad += node.grad * 1.0  →  input_0.grad += node.grad
         input_1.grad += node.grad * 1.0  →  input_1.grad += node.grad
         (Gradients "pass through" addition unchanged.)

       - **Op.SUB** (z = a - b):
         input_0.grad += node.grad * 1.0   →  input_0.grad += node.grad
         input_1.grad += node.grad * (-1.0) →  input_1.grad -= node.grad
         (Subtraction flips the sign for the second operand.)

       - **Op.MUL** (z = a * b):
         input_0.grad += node.grad * input_1.value
         input_1.grad += node.grad * input_0.value
         This is the product rule: d(a*b)/da = b, d(a*b)/db = a.
         IMPORTANT: You need the *value* of the other input, which is why
         the forward pass must run before the backward pass.

       - **Op.DIV** (z = a / b):
         input_0.grad += node.grad * (1.0 / input_1.value)
         input_1.grad += node.grad * (-input_0.value / (input_1.value ** 2))
         This is the quotient rule.

       - **Op.NEG** (z = -a):
         input_0.grad += node.grad * (-1.0)

       - **Op.RELU** (z = max(0, a)):
         if input_0.value > 0:
             input_0.grad += node.grad
         else:
             input_0.grad += 0.0  (gradient is killed)
         This is the step function derivative. At exactly 0, the derivative
         is technically undefined — we use 0 by convention (matching PyTorch).

       IMPORTANT: Use += (accumulate), not = (assign). A node might have multiple
       consumers, and each consumer contributes to the node's total gradient.
       Example: if node x feeds into both (x + y) and (x * z), its gradient
       is the SUM of gradients from both paths. This is the multivariate
       chain rule in action.

    4. **Build and return the results dict.**
       Return {node.name: node.grad for node in graph.nodes}.

    Hint: Use match/case on node.op, and remember that node.inputs[i].value
    gives you the forward-pass value of input i (needed for MUL and DIV).
    """
    # TODO(human): implement backward pass
    # Stub: return zero gradients for all nodes
    return {node.name: 0.0 for node in graph.nodes}
