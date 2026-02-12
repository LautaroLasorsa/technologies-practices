"""Phase 2: Numerically-Stable Row-Wise Softmax Kernel.

This module implements row-wise softmax: for each row x of the input matrix,
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

Concept overview:
    Unlike vector add where each program processes a contiguous block of a 1D
    vector, softmax processes one ROW per program instance. Each row is loaded
    entirely into a single block (so BLOCK_SIZE >= number of columns).

    The key operations are **reductions** within a block:
    - tl.max: find the maximum value in the row (for numerical stability)
    - tl.sum: compute the normalization denominator

    Why numerical stability matters: exp(x) overflows to inf for x > ~88 (float32).
    If any element in the row is large, naive softmax = exp(x_i) / sum(exp(x_j))
    produces inf/inf = NaN. Subtracting the row maximum first ensures the largest
    exponent is exp(0) = 1, preventing overflow.

    This is the same kernel pattern that production frameworks use. PyTorch's own
    softmax is a CUDA kernel with this exact structure.

References:
    - https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# ---------------------------------------------------------------------------
# Triton kernel — TODO(human): implement the decorated function body
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def triton_softmax_kernel(
        input_ptr,        # Pointer to input matrix in GPU global memory (row-major)
        output_ptr,       # Pointer to output matrix in GPU global memory (row-major)
        n_cols,           # Number of columns in the matrix (row length)
        input_row_stride, # Stride (in elements) between consecutive rows of the input.
                          # For a contiguous (M, N) matrix, this is N.
        output_row_stride,# Stride between consecutive rows of the output.
        BLOCK_SIZE: tl.constexpr,  # Must be >= n_cols. Each program loads one entire row.
    ):
        """Numerically-stable softmax for a single row.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches reduction operations (tl.max, tl.sum) within Triton blocks.
        # Softmax's numerical stability pattern (subtract max before exp) is critical for
        # all normalization layers (LayerNorm, RMSNorm). Production frameworks use this pattern.

        TODO(human): Replace the stub body below with the actual kernel logic.

        Each program instance processes ONE ROW of the input matrix.
        The program_id tells you WHICH row to process.

        Step-by-step guide:

        1. GET YOUR ROW INDEX
           row_idx = tl.program_id(axis=0)

           - Each program instance handles one row.
           - If the matrix has M rows, the grid has M programs.
           - row_idx ranges from 0 to M-1.

        2. COMPUTE POINTERS TO THE START OF THIS ROW
           row_start_ptr = input_ptr + row_idx * input_row_stride

           - input_ptr is the base address of the matrix.
           - row_idx * input_row_stride skips past the previous rows.
           - For a contiguous (M, N) matrix with stride=N, row 3 starts at
             base + 3*N elements into memory.

        3. GENERATE COLUMN OFFSETS AND MASK
           col_offsets = tl.arange(0, BLOCK_SIZE)
           mask = col_offsets < n_cols

           - tl.arange(0, BLOCK_SIZE) = [0, 1, 2, ..., BLOCK_SIZE-1]
           - These represent column indices within the row.
           - mask prevents reading beyond the actual number of columns.
           - BLOCK_SIZE must be >= n_cols (we load the entire row in one block).
             If BLOCK_SIZE > n_cols, the extra positions are masked out.

        4. LOAD THE ENTIRE ROW FROM GLOBAL MEMORY
           row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

           - Loads BLOCK_SIZE elements starting from row_start_ptr.
           - For masked-out positions, uses -inf as the fill value.
           - Why -inf? Because exp(-inf) = 0, so masked positions contribute
             nothing to the softmax computation. This is cleaner than loading
             zeros and then masking later.

        5. NUMERICAL STABILITY: SUBTRACT THE ROW MAXIMUM
           row_max = tl.max(row, axis=0)
           row = row - row_max

           - tl.max performs a tree reduction across the block to find the max.
           - After subtraction, the largest element becomes 0, and all others
             are negative. This ensures exp() never overflows.
           - axis=0 means reduce along the single dimension of our 1D block.

           Why this works mathematically:
             softmax(x_i) = exp(x_i) / sum(exp(x_j))
                          = exp(x_i - c) / sum(exp(x_j - c))  for any constant c
           Setting c = max(x) ensures all exponents are <= 0.

        6. COMPUTE EXPONENTIALS
           numerator = tl.exp(row)

           - Element-wise exp() on the shifted row.
           - All values are in [0, 1] since row values are in (-inf, 0].
           - The maximum element has value exp(0) = 1.

        7. COMPUTE THE NORMALIZATION DENOMINATOR
           denominator = tl.sum(numerator, axis=0)

           - tl.sum performs a tree reduction to sum all exponentials.
           - This is the partition function Z = sum(exp(x_j - max(x))).

        8. NORMALIZE
           softmax_output = numerator / denominator

           - Element-wise division produces the final softmax probabilities.
           - Each element is in (0, 1] and the row sums to 1.0.

        9. STORE THE RESULT
           output_row_start_ptr = output_ptr + row_idx * output_row_stride
           tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)

           - Write the softmax result back to the output matrix.
           - Same masking to avoid writing beyond the actual columns.
        """
        # ---- STUB: remove this and implement the steps above ----
        row_idx = tl.program_id(axis=0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        # Stub: store zeros instead of actual softmax
        tl.store(output_row_start_ptr + col_offsets, tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=mask)

else:
    def triton_softmax_kernel(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Triton is not available. Run inside the Docker container with GPU access.")


# ---------------------------------------------------------------------------
# Launch function (fully implemented)
# ---------------------------------------------------------------------------

def run_phase() -> None:
    """Launch the softmax kernel, compare with PyTorch, print results."""
    if not TRITON_AVAILABLE:
        print("[Phase 2 — Softmax] Triton not available. Run inside Docker with GPU.")
        return

    print("=" * 60)
    print("Phase 2: Numerically-Stable Row-Wise Softmax")
    print("=" * 60)

    # --- Configuration ---
    n_rows: int = 256
    n_cols: int = 781  # Intentionally NOT a power of 2

    # BLOCK_SIZE must be >= n_cols and a power of 2 (Triton requirement for tl.arange).
    # Find the next power of 2 >= n_cols.
    BLOCK_SIZE: int = 1
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE *= 2
    # BLOCK_SIZE is now 1024 (next power of 2 after 781)

    # --- Create input on GPU ---
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)

    # --- Allocate output on GPU ---
    y_triton = torch.empty_like(x)

    # --- Grid: one program per row ---
    grid = (n_rows,)

    # --- Launch kernel ---
    triton_softmax_kernel[grid](
        x, y_triton,
        n_cols,
        x.stride(0),        # input_row_stride: number of elements between rows
        y_triton.stride(0),  # output_row_stride
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # --- Verify against PyTorch ---
    y_torch = torch.softmax(x, dim=-1)
    max_diff = (y_triton - y_torch).abs().max().item()

    # Check that rows sum to 1.0
    row_sums = y_triton.sum(dim=-1)
    max_sum_error = (row_sums - 1.0).abs().max().item()

    print(f"  Matrix shape:    ({n_rows}, {n_cols})")
    print(f"  Block size:      {BLOCK_SIZE}")
    print(f"  Max difference:  {max_diff:.2e}")
    print(f"  Max row sum err: {max_sum_error:.2e}")

    if max_diff < 1e-5:
        print("  PASS: Triton softmax matches PyTorch reference.")
    else:
        print("  FAIL: Output mismatch! Check your kernel implementation.")
        print("  Hint: Make sure you subtract the row max before exp().")

    print()


if __name__ == "__main__":
    run_phase()
