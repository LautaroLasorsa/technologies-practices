"""
Phase 5: Conv+BatchNorm Fusion — Fold BN Parameters into Conv2d Weights.

GOAL: Implement the most important inference optimization in deep learning —
folding BatchNorm parameters into the preceding Conv2d layer's weights and bias.
This eliminates the BatchNorm layer entirely at inference time.

WHY CONV+BN FUSION IS THE #1 INFERENCE OPTIMIZATION:

  Almost every CNN (ResNet, EfficientNet, MobileNet, YOLO, etc.) uses the pattern:
      Conv2d -> BatchNorm2d -> ReLU

  During TRAINING, BatchNorm computes running statistics (mean, variance) and
  applies a learned affine transform. During INFERENCE, those statistics are
  FIXED — BatchNorm becomes a simple affine operation:

      BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta

  where gamma (weight), beta (bias), mean (running_mean), and var (running_var)
  are all constants at inference time.

  Since both Conv2d and BatchNorm are affine (linear) operations, they can be
  ALGEBRAICALLY COMBINED into a single Conv2d with modified weights and bias.
  This is not an approximation — it is mathematically exact.

THE MATH — DERIVING FUSED PARAMETERS:

  Let Conv2d produce:   y = W_conv * x + b_conv     (matrix multiply + bias)
  Let BN transform:     z = gamma * (y - mean) / sqrt(var + eps) + beta

  Substituting y into BN:
      z = gamma * (W_conv * x + b_conv - mean) / sqrt(var + eps) + beta

  Distribute gamma / sqrt(var + eps):
      z = (gamma / sqrt(var + eps)) * W_conv * x
        + (gamma / sqrt(var + eps)) * (b_conv - mean)
        + beta

  Define the "scaling factor" per output channel:
      scale = gamma / sqrt(var + eps)

  Then:
      z = (scale * W_conv) * x + (scale * (b_conv - mean) + beta)

  So the fused Conv2d has:
      fused_weight = scale * W_conv
      fused_bias   = scale * (b_conv - mean) + beta

  where:
      scale = bn_weight / sqrt(bn_running_var + bn_eps)

  CHANNEL-WISE: The scale factor is a vector with one value per output channel.
  For a Conv2d with out_channels=C, scale has shape (C,). The weight tensor has
  shape (C, C_in, kH, kW), so we multiply each output channel's filter by its
  corresponding scale factor.

WHY THIS IS EXACT (NOT AN APPROXIMATION):

  Both Conv2d and BatchNorm (during inference) are affine transformations:
    f(x) = Ax + b
  The composition of two affine transforms is another affine transform:
    g(f(x)) = A'x + b'
  We're just computing A' and b' algebraically and storing them in a single Conv2d.
  There is NO approximation, NO loss of accuracy (up to floating-point precision).

WHAT CHANGES AT THE GRAPH LEVEL:

  Before:  input -> Conv2d -> BatchNorm2d -> output
  After:   input -> Conv2d(fused) -> output

  The BatchNorm node is REMOVED from the graph. The Conv2d node stays but with
  modified weight and bias tensors. This saves:
    - One kernel launch (the BN kernel is eliminated)
    - Memory bandwidth for reading/writing the intermediate tensor
    - The BN parameters themselves (gamma, beta, mean, var)

PRODUCTION USAGE:

  Every inference framework does this automatically:
    - TorchScript: torch.quantization.fuse_modules
    - TensorRT: built-in Conv+BN fusion pass
    - ONNX Runtime: built-in graph optimization
    - TVM: relay.transform.FoldBatchNorm
    - PyTorch FX: this is exactly what the official tutorial demonstrates
"""

import copy

import torch
import torch.nn as nn
import torch.fx
from torch.fx import GraphModule, Node


# ---------------------------------------------------------------------------
# Model with Conv+BN pattern
# ---------------------------------------------------------------------------
class ConvBNModel(nn.Module):
    """A simple CNN with Conv2d + BatchNorm2d + ReLU layers.

    This is the canonical pattern found in almost every CNN:
      Conv2d -> BatchNorm2d -> ReLU

    The model has two such blocks plus a final Conv2d without BN.
    """

    def __init__(self, in_channels: int = 3, mid_channels: int = 16, out_channels: int = 32):
        super().__init__()
        # Block 1: Conv + BN + ReLU (should be fusible)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU()

        # Block 2: Conv + BN + ReLU (should be fusible)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        # Block 3: Conv only (no BN — should NOT be fused)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


# ---------------------------------------------------------------------------
# TODO(human): Implement the Conv+BN weight fusion math
# ---------------------------------------------------------------------------
def fuse_conv_bn_weights(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the fused Conv2d weight and bias from Conv + BatchNorm parameters.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches algebraic fusion — the mathematical transformation that
    # folds BatchNorm into Conv weights. This is the #1 inference optimization in
    # CNNs, used by every framework (TensorRT, ONNX, TorchScript).

    TODO(human): Implement this function.

    Given a Conv2d followed by a BatchNorm2d, compute the weight and bias of a
    SINGLE Conv2d that produces the same output as applying Conv then BN.

    DETAILED GUIDANCE:

    1. EXTRACT PARAMETERS FROM THE CONV LAYER:
       - conv.weight: shape (out_channels, in_channels, kH, kW) — the convolution filters
       - conv.bias: shape (out_channels,) or None — the convolution bias
         IMPORTANT: Conv2d may have bias=None. If so, treat it as a zero vector:
             conv_bias = conv.bias if conv.bias is not None else torch.zeros(out_channels)
         where out_channels = conv.weight.shape[0]

    2. EXTRACT PARAMETERS FROM THE BN LAYER:
       BatchNorm2d has four parameter tensors, all of shape (num_features,) which
       equals out_channels of the preceding Conv:
         - bn.weight (gamma): the learned scale factor per channel
         - bn.bias (beta): the learned shift per channel
         - bn.running_mean (mean): the running average of batch means
         - bn.running_var (var): the running average of batch variances
         - bn.eps: a small constant for numerical stability (e.g., 1e-5)

       Access them as:
           gamma = bn.weight         # shape: (C,)
           beta = bn.bias            # shape: (C,)
           mean = bn.running_mean    # shape: (C,)
           var = bn.running_var      # shape: (C,)
           eps = bn.eps              # scalar float

       IMPORTANT: Use .detach() on these tensors to avoid autograd issues:
           gamma = bn.weight.detach()
           beta = bn.bias.detach()
           mean = bn.running_mean.detach()
           var = bn.running_var.detach()

    3. COMPUTE THE SCALE FACTOR:
       The key quantity is the per-channel scale factor:
           scale = gamma / sqrt(var + eps)

       In PyTorch:
           scale = gamma / torch.sqrt(var + eps)

       scale has shape (C,) — one value per output channel.

    4. COMPUTE FUSED WEIGHT:
       The fused weight is:
           fused_weight = scale * conv_weight

       But scale has shape (C,) and conv_weight has shape (C, C_in, kH, kW).
       We need to broadcast scale across the other dimensions. Reshape scale to
       (C, 1, 1, 1) so it broadcasts correctly:
           fused_weight = conv_weight * scale.reshape(-1, 1, 1, 1)

       This multiplies each output channel's entire filter (all C_in * kH * kW values)
       by that channel's scale factor.

       WHY THIS IS CORRECT:
       Conv2d computes one output channel by taking a weighted sum over the input.
       Multiplying the filter by `scale` is equivalent to multiplying the output
       by `scale`, because convolution is linear:
           scale * (W * x) = (scale * W) * x

    5. COMPUTE FUSED BIAS:
       The fused bias is:
           fused_bias = scale * (conv_bias - mean) + beta

       In PyTorch:
           fused_bias = scale * (conv_bias - mean) + beta

       All tensors here have shape (C,), so no reshaping is needed.

       WHY THIS IS CORRECT:
       Expanding the BN formula:
           BN(Conv(x)) = gamma * (W*x + b - mean) / sqrt(var + eps) + beta
                       = (gamma / sqrt(var + eps)) * W * x
                         + (gamma / sqrt(var + eps)) * (b - mean) + beta
                       = scale * W * x + scale * (b - mean) + beta
                       = fused_W * x + fused_b

    6. RETURN:
       Return (fused_weight, fused_bias) as a tuple of tensors.

    7. NUMERICAL VERIFICATION:
       After implementing, the run_phase() function will verify your result by
       comparing the fused Conv2d's output against Conv+BN's output on random
       input. They should match to within floating-point tolerance (~1e-6).

    8. EDGE CASES:
       - If conv.bias is None, create a zero bias: torch.zeros(conv.weight.shape[0])
       - The .detach() calls prevent gradient tracking (we're doing inference math)
       - All computation should be done with torch tensors (no numpy)

    Args:
        conv: The Conv2d layer (contains weight and optionally bias).
        bn: The BatchNorm2d layer (contains weight, bias, running_mean, running_var, eps).

    Returns:
        Tuple of (fused_weight, fused_bias) tensors for the replacement Conv2d.
    """
    # STUB: returns the original conv parameters unchanged so the file runs
    fused_weight = conv.weight.detach().clone()
    fused_bias = conv.bias.detach().clone() if conv.bias is not None else torch.zeros(conv.weight.shape[0])
    return fused_weight, fused_bias


# ---------------------------------------------------------------------------
# Graph-level Conv+BN fusion pass (fully implemented)
# ---------------------------------------------------------------------------
def find_conv_bn_pairs(graph_module: GraphModule) -> list[tuple[Node, Node]]:
    """Find all (conv_node, bn_node) pairs eligible for fusion.

    Walks the graph and finds call_module nodes where:
      - A node targets an nn.Conv2d
      - Its sole consumer is a node targeting an nn.BatchNorm2d
    """
    pairs: list[tuple[Node, Node]] = []

    for node in graph_module.graph.nodes:
        if node.op != "call_module":
            continue

        module = graph_module.get_submodule(node.target)
        if not isinstance(module, nn.Conv2d):
            continue

        # Check: Conv's only user is a BN
        if len(node.users) != 1:
            continue

        bn_candidate = next(iter(node.users))
        if bn_candidate.op != "call_module":
            continue

        bn_module = graph_module.get_submodule(bn_candidate.target)
        if not isinstance(bn_module, nn.BatchNorm2d):
            continue

        pairs.append((node, bn_candidate))

    return pairs


def apply_conv_bn_fusion(graph_module: GraphModule) -> GraphModule:
    """Fuse Conv+BN pairs in the graph by folding BN into Conv weights.

    For each matched pair:
      1. Compute fused weights using fuse_conv_bn_weights()
      2. Create a new Conv2d with the fused parameters
      3. Replace the Conv+BN pair with the single fused Conv in the graph
    """
    pairs = find_conv_bn_pairs(graph_module)
    print(f"  Found {len(pairs)} Conv+BN pair(s) to fuse.")

    for conv_node, bn_node in pairs:
        conv_module = graph_module.get_submodule(conv_node.target)
        bn_module = graph_module.get_submodule(bn_node.target)

        # Compute fused parameters (this is the TODO(human) part!)
        fused_weight, fused_bias = fuse_conv_bn_weights(conv_module, bn_module)

        # Create a new Conv2d with the fused parameters
        fused_conv = nn.Conv2d(
            in_channels=conv_module.in_channels,
            out_channels=conv_module.out_channels,
            kernel_size=conv_module.kernel_size,
            stride=conv_module.stride,
            padding=conv_module.padding,
            dilation=conv_module.dilation,
            groups=conv_module.groups,
            bias=True,  # Fused conv always has bias (from BN beta)
        )
        fused_conv.weight = nn.Parameter(fused_weight)
        fused_conv.bias = nn.Parameter(fused_bias)

        # Register the fused conv on the graph module
        fused_name = f"fused_{conv_node.target}"
        setattr(graph_module, fused_name, fused_conv)

        # Insert the new node and rewire the graph
        with graph_module.graph.inserting_before(bn_node):
            fused_node = graph_module.graph.call_module(
                fused_name,
                args=conv_node.args,
            )

        bn_node.replace_all_uses_with(fused_node)
        graph_module.graph.erase_node(bn_node)
        graph_module.graph.erase_node(conv_node)

    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_conv_bn_fusion(original: nn.Module, fused: GraphModule) -> None:
    """Verify that fused model matches original on random input."""
    # Use a typical image tensor: (batch=2, channels=3, height=8, width=8)
    x = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        y_original = original(x)
        y_fused = fused(x)

    if torch.allclose(y_original, y_fused, atol=1e-5):
        print("[OK] Fused Conv+BN model output matches original.")
        max_diff = (y_original - y_fused).abs().max().item()
        print(f"     Max absolute difference: {max_diff:.2e}")
    else:
        max_diff = (y_original - y_fused).abs().max().item()
        print(f"[FAIL] Outputs differ! Max difference: {max_diff:.2e}")
        print("       Check your fuse_conv_bn_weights() implementation.")


def verify_single_layer_fusion(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    """Verify fusion math on a single Conv+BN pair with explicit computation."""
    x = torch.randn(2, conv.in_channels, 8, 8)

    # Ground truth: Conv then BN
    with torch.no_grad():
        conv_out = conv(x)
        expected = bn(conv_out)

    # Fused: single Conv with fused parameters
    fused_weight, fused_bias = fuse_conv_bn_weights(conv, bn)
    fused_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, bias=True,
    )
    fused_conv.weight = nn.Parameter(fused_weight)
    fused_conv.bias = nn.Parameter(fused_bias)

    with torch.no_grad():
        actual = fused_conv(x)

    if torch.allclose(expected, actual, atol=1e-5):
        print("[OK] Single-layer fusion math is correct.")
    else:
        max_diff = (expected - actual).abs().max().item()
        print(f"[FAIL] Single-layer fusion incorrect! Max diff: {max_diff:.2e}")
        print("       The fused Conv2d does not match Conv+BN output.")
        print("       Review the math in fuse_conv_bn_weights().")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_phase() -> None:
    """Run Phase 5: Conv+BatchNorm fusion."""
    print("\n" + "=" * 70)
    print("  PHASE 5: Conv + BatchNorm Fusion")
    print("=" * 70)

    # Create the model and put in eval mode (critical for BN — uses running stats)
    model = ConvBNModel()
    model.eval()
    print(f"\nOriginal model:\n{model}\n")

    # Run a few forward passes to populate BN running statistics
    # (In practice, these come from training. Here we simulate with random data.)
    print("Populating BatchNorm running statistics with random data...")
    model.train()
    for _ in range(10):
        dummy = torch.randn(8, 3, 8, 8)
        _ = model(dummy)
    model.eval()
    print("  Done. BN running_mean and running_var are now non-trivial.\n")

    # Step 1: Verify single-layer fusion math
    print("=" * 70)
    print("STEP 1: Verify single-layer fusion math")
    print("=" * 70)
    verify_single_layer_fusion(model.conv1, model.bn1)
    verify_single_layer_fusion(model.conv2, model.bn2)

    # Step 2: Apply full graph-level fusion
    print("\n" + "=" * 70)
    print("STEP 2: Full graph-level Conv+BN fusion")
    print("=" * 70)

    # Deep copy because apply_conv_bn_fusion modifies the model
    model_copy = copy.deepcopy(model)
    traced = torch.fx.symbolic_trace(model_copy)

    print("\nGraph BEFORE fusion:")
    traced.graph.print_tabular()

    fused = apply_conv_bn_fusion(traced)

    print("\nGraph AFTER fusion:")
    fused.graph.print_tabular()

    # Step 3: Verify correctness
    print("\n" + "=" * 70)
    print("STEP 3: Verify full-model correctness")
    print("=" * 70)
    verify_conv_bn_fusion(model, fused)

    # Show generated code
    print("\n" + "=" * 70)
    print("GENERATED FORWARD CODE (after fusion)")
    print("=" * 70)
    print(fused.code)

    # Count eliminated operations
    original_nodes = len(list(torch.fx.symbolic_trace(model).graph.nodes))
    fused_nodes = len(list(fused.graph.nodes))
    print(f"Nodes eliminated: {original_nodes - fused_nodes} "
          f"(from {original_nodes} to {fused_nodes})")

    # Key takeaways
    print("\n" + "-" * 70)
    print("KEY TAKEAWAYS:")
    print("-" * 70)
    print("""
  1. Conv+BN fusion is EXACT (not approximate) because both are affine transforms.
  2. The key formula: scale = gamma / sqrt(var + eps), then scale the conv weight/bias.
  3. BN is eliminated entirely — no extra kernel, no extra memory traffic.
  4. This only works at INFERENCE time (model.eval()), when BN uses fixed statistics.
  5. Every production inference framework does this automatically.
  6. The same principle extends to: Linear+BN, Conv+BN+ReLU (fuse all three), etc.
""")


if __name__ == "__main__":
    run_phase()
