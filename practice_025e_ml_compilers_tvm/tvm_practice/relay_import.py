"""Phase 4: Relay Integration -- importing PyTorch models into TVM.

Relay is TVM's high-level graph IR. It sits above tensor expressions and
schedules, operating at the level of complete neural network graphs:

    PyTorch model
        │
        ▼ torch.jit.trace
    TorchScript IR
        │
        ▼ relay.frontend.from_pytorch
    Relay IR (graph of operators: dense, relu, batch_norm, etc.)
        │
        ▼ relay.build (applies graph passes: fusion, constant folding, layout opt)
    Compiled TVM module (optimized executable)
        │
        ▼ module.run()
    Output tensors

Relay's graph-level optimizations are complementary to the schedule-level
optimizations from Phase 2-3:
    - Schedule level: optimizes ONE operator (e.g., a single matmul)
    - Relay level: optimizes the GRAPH of operators (fusion, elimination, layout)

For example, Relay can fuse dense + bias_add + relu into a single operator,
then TVM schedules that fused operator as one kernel. This is the same
fusion concept from Practice 025b, but automated.
"""

from __future__ import annotations

import numpy as np

try:
    import tvm
    from tvm import relay

    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False
    print(
        "WARNING: TVM not found. This practice requires Docker.\n"
        "Run: docker compose run --rm tvm python -m tvm_practice.relay_import\n"
    )

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not found.")


# ---------------------------------------------------------------------------
# PyTorch model definition (fully implemented)
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """A 2-layer MLP: input -> linear -> relu -> linear -> output.

    This is a minimal model that exercises:
    - Dense (linear) layers: the most common operation in ML
    - ReLU activation: a non-linearity
    - Bias addition: built into nn.Linear

    When imported to Relay, this becomes a graph of:
        %0 = nn.dense(%input, %weight1) + %bias1
        %1 = nn.relu(%0)
        %2 = nn.dense(%1, %weight2) + %bias2
    """

    def __init__(self, in_features: int = 64, hidden: int = 128, out_features: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# TorchScript tracing (fully implemented)
# ---------------------------------------------------------------------------

def trace_pytorch_model(
    model: nn.Module, input_shape: tuple[int, ...]
) -> tuple[torch.jit.ScriptModule, torch.Tensor]:
    """Trace a PyTorch model using torch.jit.trace.

    TVM's Relay frontend requires a TorchScript model (traced or scripted).
    torch.jit.trace runs the model with a concrete input and records all
    operations. The result is a TorchScript IR that Relay can parse.

    Returns:
        (traced_model, example_input) -- both needed for relay.frontend.from_pytorch
    """
    model.eval()
    example_input = torch.randn(*input_shape)

    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    return traced, example_input


# ---------------------------------------------------------------------------
# Relay import and compilation -- TODO(human)
# ---------------------------------------------------------------------------

def import_and_compile(
    traced_model: torch.jit.ScriptModule,
    input_shape: tuple[int, ...],
    input_name: str = "input0",
) -> tuple:
    """Import a traced PyTorch model into TVM Relay, compile, and return the module.

    TODO(human): Implement the Relay import and compilation pipeline.

    This is the real-world TVM workflow: take a trained PyTorch model, import
    it into TVM, let TVM optimize it, and get a compiled module that runs
    without PyTorch.

    Steps:

        1. Define the input shape dict for Relay:
               shape_dict = {input_name: input_shape}
           - Relay needs to know the shape and name of each input tensor.
           - input_name must match what TorchScript uses. For traced models,
             the first input is typically "input0".
           - input_shape is the concrete shape, e.g., (1, 64) for batch=1, features=64.

        2. Import from PyTorch using relay.frontend.from_pytorch:
               mod, params = relay.frontend.from_pytorch(traced_model, [shape_dict])
           - mod: a Relay Module containing the computation graph (IRModule)
           - params: a dict of parameter name -> tvm.nd.array (the weights)
           - This parses the TorchScript IR and converts each operation to
             its Relay equivalent (e.g., aten::linear -> nn.dense + nn.bias_add)

        3. Print the Relay IR to see what was imported:
               print("Relay IR:")
               print(mod["main"])
           - The IR shows the graph of Relay operators.
           - Look for nn.dense, nn.relu, nn.bias_add -- these correspond to
             your PyTorch layers.
           - The IR also shows type information (tensor shapes and dtypes).

        4. Apply Relay optimization passes:
               with tvm.transform.PassContext(opt_level=3):
                   lib = relay.build(mod, target="llvm", params=params)
           - opt_level=3 enables all optimizations: operator fusion, constant
             folding, layout transformation, dead code elimination.
           - relay.build returns a compiled library that you can execute.
           - "llvm" target means compile for CPU.

        5. Create a runtime module for execution:
               dev = tvm.cpu(0)
               module = tvm.contrib.graph_executor.GraphModule(
                   lib["default"](dev)
               )
           - GraphModule wraps the compiled library and provides set_input/run/get_output.

        6. Return the module and device:
               return module, dev

    Why Relay matters:
        - Graph-level fusion: Relay can fuse dense+bias_add+relu into one kernel.
          This is impossible at the TE/schedule level because schedules only see
          one operator at a time.
        - Layout optimization: Relay can change tensor layouts (NCHW -> NHWC)
          to match the hardware's preferred format.
        - Constant folding: if any computation depends only on weights (not input),
          Relay evaluates it at compile time.
        - Dead code elimination: unused branches of the graph are removed.

    Args:
        traced_model: TorchScript model from torch.jit.trace
        input_shape: shape of the input tensor, e.g., (1, 64)
        input_name: name of the input in the Relay graph (usually "input0")

    Returns:
        Tuple of (tvm.contrib.graph_executor.GraphModule, tvm.runtime.Device)
    """
    if not TVM_AVAILABLE:
        print("  [SKIP] TVM not available")
        return None, None

    # TODO(human): implement the 6 steps above
    # Hint: the core is 5 lines of code -- the concepts are what matter

    # Stub: return None so the file runs
    print("  (Implement import_and_compile in relay_import.py)")
    return None, None


# ---------------------------------------------------------------------------
# Verification (fully implemented)
# ---------------------------------------------------------------------------

def verify_relay_output(
    pytorch_model: nn.Module,
    tvm_module,
    example_input: torch.Tensor,
    input_name: str = "input0",
) -> None:
    """Compare TVM output against PyTorch output.

    If the Relay import and compilation worked correctly, both should
    produce (nearly) identical results. Small floating-point differences
    are expected due to different computation order.
    """
    if tvm_module is None:
        print("  [SKIP] TVM module not available -- implement import_and_compile first")
        return

    # PyTorch reference output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(example_input).numpy()

    # TVM output
    tvm_module.set_input(input_name, tvm.nd.array(example_input.numpy()))
    tvm_module.run()
    tvm_output = tvm_module.get_output(0).numpy()

    # Compare
    print(f"\n  PyTorch output (first 5): {pytorch_output[0, :5]}")
    print(f"  TVM output (first 5):     {tvm_output[0, :5]}")

    try:
        np.testing.assert_allclose(tvm_output, pytorch_output, rtol=1e-4, atol=1e-4)
        print("  Verification: PASSED (TVM output matches PyTorch)")
    except AssertionError as e:
        print(f"  Verification: FAILED -- {e}")
        max_diff = np.max(np.abs(tvm_output - pytorch_output))
        print(f"  Max absolute difference: {max_diff:.6e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase() -> None:
    """Run Phase 4: Relay Integration."""
    print("\n" + "#" * 60)
    print("  PHASE 4: Relay Integration")
    print("#" * 60)

    if not TORCH_AVAILABLE:
        print("  [SKIP] PyTorch not available")
        return

    if not TVM_AVAILABLE:
        print("  [SKIP] TVM not available")
        return

    # Define model and input shape
    in_features, hidden, out_features = 64, 128, 10
    batch_size = 1
    input_shape = (batch_size, in_features)

    print(f"\n  Model: MLP({in_features} -> {hidden} -> {out_features})")
    print(f"  Input shape: {input_shape}")

    # Step 1: Create and trace the PyTorch model
    model = SimpleMLP(in_features, hidden, out_features)
    traced, example_input = trace_pytorch_model(model, input_shape)
    print(f"\n  TorchScript trace: OK")

    # Step 2: Import into Relay and compile
    print(f"\n--- Relay Import ---")
    tvm_module, dev = import_and_compile(traced, input_shape)

    # Step 3: Verify output matches PyTorch
    print(f"\n--- Verification ---")
    verify_relay_output(model, tvm_module, example_input)

    print("\n" + "-" * 60)
    print("KEY TAKEAWAYS:")
    print("-" * 60)
    print("""
  1. relay.frontend.from_pytorch converts TorchScript -> Relay IR.
  2. Relay IR represents the full graph: operators, shapes, dtypes.
  3. relay.build with opt_level=3 applies fusion, folding, layout optimization.
  4. The compiled module runs WITHOUT PyTorch -- pure TVM runtime.
  5. Graph-level fusion (Relay) + operator-level scheduling (TE) = full optimization.
""")


if __name__ == "__main__":
    run_phase()
