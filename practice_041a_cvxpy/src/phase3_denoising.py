"""Phase 3: Signal Denoising with Total Variation — SOCP with CVXPY.

Recovers a piecewise-constant signal from noisy observations using total
variation (TV) regularization. TV penalizes the sum of absolute first
differences, preserving sharp edges while smoothing noise.

DCP atoms used:
  - cp.sum_squares(x - y): convex — data fidelity term
  - cp.diff(x): affine — first differences [x[1]-x[0], x[2]-x[1], ...]
  - cp.norm1(cp.diff(x)): convex(affine) = convex — total variation penalty
"""

import cvxpy as cp
import numpy as np


# ============================================================================
# Signal generation
# ============================================================================

def generate_piecewise_signal(
    n: int = 200,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a piecewise-constant signal with additive Gaussian noise.

    The clean signal has 5 segments with different levels.
    This is the ideal test case for TV denoising: sharp transitions + flat regions.

    Returns:
        y: (n,) noisy signal
        x_true: (n,) clean piecewise-constant signal
    """
    rng = np.random.default_rng(seed)

    x_true = np.zeros(n)

    # Define segments: (start_frac, end_frac, value)
    segments = [
        (0.00, 0.20, 1.0),
        (0.20, 0.35, 3.0),
        (0.35, 0.55, 0.5),
        (0.55, 0.75, 2.5),
        (0.75, 1.00, 1.5),
    ]

    for start_frac, end_frac, value in segments:
        i_start = int(start_frac * n)
        i_end = int(end_frac * n)
        x_true[i_start:i_end] = value

    noise = noise_std * rng.standard_normal(n)
    y = x_true + noise

    return y, x_true


# ============================================================================
# TODO(human): Total variation denoising
# ============================================================================

def total_variation_denoise(y: np.ndarray, lambd: float) -> np.ndarray:
    """Denoise a 1D signal using total variation regularization.

    minimize ||x - y||_2^2 + lambda * TV(x)

    where TV(x) = sum(|x[i+1] - x[i]|) = ||diff(x)||_1

    Args:
        y: (n,) noisy signal
        lambd: regularization strength (>= 0)

    Returns:
        x_opt: (n,) denoised signal
    """
    # TODO(human): Total Variation Denoising with CVXPY
    # minimize ||x - y||_2^2 + lambda * TV(x)
    # where TV(x) = sum(|x[i+1] - x[i]|) = ||Dx||_1
    # D is the first-difference matrix.
    #
    # In CVXPY:
    #   x = cp.Variable(n)
    #   data_fidelity = cp.sum_squares(x - y)
    #   tv_penalty = cp.norm1(cp.diff(x))  # cp.diff computes differences
    #   prob = cp.Problem(cp.Minimize(data_fidelity + lambd * tv_penalty))
    #
    # DCP: sum_squares (convex) + norm1(affine) (convex) = convex. ✓
    #
    # TV denoising preserves edges (sharp transitions) while smoothing noise.
    # This is why it's used in image processing — it doesn't blur edges.
    # Small lambda → noisy but detailed. Large lambda → smooth but loses detail.
    raise NotImplementedError


# ============================================================================
# Display helpers
# ============================================================================

def print_signal_comparison(
    y: np.ndarray,
    x_true: np.ndarray,
    results: list[tuple[str, np.ndarray]],
) -> None:
    """Print text-based signal plot comparing noisy, true, and denoised signals."""
    n = len(y)

    for name, x_est in results:
        mse = np.mean((x_est - x_true) ** 2)
        tv = np.sum(np.abs(np.diff(x_est)))
        tv_true = np.sum(np.abs(np.diff(x_true)))
        print(f"\n  {name}:")
        print(f"    MSE to true: {mse:.4f}")
        print(f"    TV(estimate): {tv:.2f}  (TV(true) = {tv_true:.2f})")

    # Text-based plot: show signal at sampled positions
    print("\n" + "=" * 72)
    print("SIGNAL COMPARISON (sampled positions)")
    print("=" * 72)

    # Sample 40 evenly spaced points for display
    sample_idx = np.linspace(0, n - 1, 40, dtype=int)
    all_vals = np.concatenate([y[sample_idx], x_true[sample_idx]])
    for _, x_est in results:
        all_vals = np.concatenate([all_vals, x_est[sample_idx]])
    vmin, vmax = all_vals.min(), all_vals.max()
    plot_width = 50

    def val_to_col(v: float) -> int:
        if vmax - vmin < 1e-8:
            return plot_width // 2
        return int((v - vmin) / (vmax - vmin) * (plot_width - 1))

    print(f"\n  Legend: . = noisy, T = true, 1/2/3 = denoised (by lambda)")
    print(f"  {'Pos':>5s} |{'':^{plot_width}s}|")

    symbols = ["1", "2", "3", "4", "5"]
    for idx in sample_idx:
        row = [" "] * plot_width

        # Noisy
        c = val_to_col(y[idx])
        row[c] = "."

        # True
        c = val_to_col(x_true[idx])
        row[c] = "T"

        # Denoised signals
        for k, (_, x_est) in enumerate(results):
            c = val_to_col(x_est[idx])
            sym = symbols[k] if k < len(symbols) else "X"
            row[c] = sym

        print(f"  {idx:5d} |{''.join(row)}|")

    print(f"  {'':>5s} +{'-'*plot_width}+")
    print(f"  {'':>6s} {vmin:<8.2f}{' '*(plot_width-18)}{vmax:>8.2f}")


def print_summary_table(
    x_true: np.ndarray,
    y: np.ndarray,
    results: list[tuple[str, np.ndarray]],
) -> None:
    """Print summary metrics table."""
    print("\n" + "=" * 65)
    print("SUMMARY: Denoising Quality")
    print("=" * 65)

    noisy_mse = np.mean((y - x_true) ** 2)
    noisy_tv = np.sum(np.abs(np.diff(y)))
    true_tv = np.sum(np.abs(np.diff(x_true)))

    print(f"  {'Signal':<30s} {'MSE':>8s} {'TV':>10s} {'TV ratio':>10s}")
    print(f"  {'-'*60}")
    print(f"  {'Noisy input':<30s} {noisy_mse:>8.4f} {noisy_tv:>10.2f} {noisy_tv/true_tv:>10.2f}x")
    print(f"  {'True signal':<30s} {'0.0000':>8s} {true_tv:>10.2f} {'1.00':>10s}x")

    for name, x_est in results:
        mse = np.mean((x_est - x_true) ** 2)
        tv = np.sum(np.abs(np.diff(x_est)))
        ratio = tv / true_tv if true_tv > 1e-8 else float("inf")
        print(f"  {name:<30s} {mse:>8.4f} {tv:>10.2f} {ratio:>10.2f}x")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 65)
    print("Phase 3: Total Variation Signal Denoising")
    print("=" * 65)

    y, x_true = generate_piecewise_signal(n=200, noise_std=0.5)
    noisy_mse = np.mean((y - x_true) ** 2)
    print(f"\nSignal: {len(y)} samples, piecewise-constant with 5 segments")
    print(f"Noise std: 0.5 | Noisy MSE: {noisy_mse:.4f}")

    # Denoise at different lambda values
    results = []
    for lambd in [1.0, 5.0, 20.0]:
        print(f"\nSolving TV denoising with lambda = {lambd}...", end=" ")
        x_denoised = total_variation_denoise(y, lambd)
        mse = np.mean((x_denoised - x_true) ** 2)
        print(f"MSE = {mse:.4f}")
        results.append((f"TV lambda={lambd}", x_denoised))

    print_signal_comparison(y, x_true, results)
    print_summary_table(x_true, y, results)


if __name__ == "__main__":
    main()
