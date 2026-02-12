"""Phase 3: Fused MatMul + ReLU Kernel.

This module implements: C = ReLU(A @ B) where A is (M, K) and B is (K, N).

Concept overview:
    This is the most important pattern in ML compilers. Every linear layer in a
    neural network computes Y = activation(X @ W + bias). Without fusion, this
    requires:
      1. Compute X @ W → write intermediate to global memory
      2. Read intermediate from global memory → apply activation → write result

    With kernel fusion, the activation is applied **inside the matmul kernel**
    before writing to global memory, eliminating one full read+write of the
    entire output matrix. For large matrices, this memory bandwidth saving
    translates to significant speedup.

    The matmul itself uses **tiled computation**: instead of loading entire rows
    of A and columns of B (which wouldn't fit in shared memory), we process the
    K dimension in blocks. Each tile of the output C[m:m+BM, n:n+BN] is computed
    by accumulating partial dot products from tiles of A and B.

    The 2D grid has dimensions (ceil(M/BM), ceil(N/BN)). Each program instance
    computes one BM x BN tile of the output matrix.

References:
    - https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
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
    def triton_matmul_relu_kernel(
        # Pointers to matrices
        a_ptr,     # (M, K) input matrix A
        b_ptr,     # (K, N) input matrix B
        c_ptr,     # (M, N) output matrix C = ReLU(A @ B)
        # Matrix dimensions
        M,         # Number of rows in A and C
        N,         # Number of columns in B and C
        K,         # Number of columns in A / rows in B (shared dimension)
        # Strides (elements between consecutive rows)
        stride_am, # A's row stride: number of elements to skip to go from A[i,:] to A[i+1,:]
        stride_ak, # A's column stride (1 for row-major contiguous)
        stride_bk, # B's row stride
        stride_bn, # B's column stride (1 for row-major contiguous)
        stride_cm, # C's row stride
        stride_cn, # C's column stride (1 for row-major contiguous)
        # Tile sizes (compile-time constants)
        BLOCK_M: tl.constexpr,  # Tile height (rows of A/C per program)
        BLOCK_N: tl.constexpr,  # Tile width (columns of B/C per program)
        BLOCK_K: tl.constexpr,  # Tile depth (how many K elements to process per iteration)
    ):
        """Fused tiled matmul + ReLU: C = max(A @ B, 0).

        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches tiled matrix multiplication with kernel fusion — the most
        # important pattern in ML compilers. This is what TorchInductor generates for linear
        # layers. Understanding K-loop accumulation and fusion is essential for ML systems work.

        TODO(human): Replace the stub body below with the actual kernel logic.

        This kernel computes one BLOCK_M x BLOCK_N tile of the output matrix C.
        The 2D grid determines which tile: program_id(0) selects the row-tile,
        program_id(1) selects the column-tile.

        Step-by-step guide:

        1. GET YOUR TILE COORDINATES
           pid_m = tl.program_id(axis=0)  # Which row-tile (0 to ceil(M/BLOCK_M)-1)
           pid_n = tl.program_id(axis=1)  # Which col-tile (0 to ceil(N/BLOCK_N)-1)

           - The launch grid is 2D: (ceil(M/BLOCK_M), ceil(N/BLOCK_N)).
           - Program (pid_m, pid_n) computes the output tile:
               C[pid_m*BM : (pid_m+1)*BM, pid_n*BN : (pid_n+1)*BN]

           Visual example (M=8, N=8, BLOCK_M=4, BLOCK_N=4):
               Grid is (2, 2) → 4 program instances
               Program (0,0): C[0:4, 0:4]  |  Program (0,1): C[0:4, 4:8]
               Program (1,0): C[4:8, 0:4]  |  Program (1,1): C[4:8, 4:8]

        2. COMPUTE ROW AND COLUMN OFFSETS FOR THIS TILE
           offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM] row indices
           offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN] col indices

           - offs_m are the GLOBAL row indices this program is responsible for.
           - offs_n are the GLOBAL column indices.

        3. INITIALIZE THE ACCUMULATOR
           acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

           - The accumulator holds the partial result for this output tile.
           - We use float32 even if inputs are float16 (common for precision).
           - This will be accumulated over the K-dimension loop below.

        4. ITERATE OVER THE K DIMENSION IN BLOCKS
           for k_start in range(0, K, BLOCK_K):
               k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BK] current K indices

           - We can't load entire rows of A or columns of B at once (too large
             for registers/shared memory). Instead, we process K in tiles of BLOCK_K.
           - Each iteration loads a BLOCK_M x BLOCK_K tile from A and a
             BLOCK_K x BLOCK_N tile from B, then accumulates their product.

           Visualization (BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK):

               A (M x K)          B (K x N)         C (M x N)
               ┌─────────┐        ┌─────────┐       ┌─────────┐
               │         │        │  BK x BN │       │         │
               │ BM x BK │   @    │  (tile)  │  +=   │ BM x BN │
               │ (tile)  │        │         │       │ (accum) │
               └─────────┘        └─────────┘       └─────────┘
               k_start             k_start

        5. LOAD A TILE FROM MATRIX A
               a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
               a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
               a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

           - offs_m[:, None] is shape (BLOCK_M, 1) — broadcasting over columns.
           - k_offsets[None, :] is shape (1, BLOCK_K) — broadcasting over rows.
           - Together they form a (BLOCK_M, BLOCK_K) grid of pointers.
           - stride_am and stride_ak convert (row, col) indices to memory offsets.
           - The mask handles boundary conditions (last tile may be partial).
           - other=0.0: masked positions get zero, which is neutral for addition.

        6. LOAD A TILE FROM MATRIX B
               b_ptrs = b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn
               b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
               b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

           - Same pattern as A, but with (BLOCK_K, BLOCK_N) shape.
           - k_offsets[:, None] broadcasts over columns.
           - offs_n[None, :] broadcasts over rows.

        7. ACCUMULATE THE PARTIAL DOT PRODUCT
               acc += tl.dot(a_tile, b_tile)

           - tl.dot computes the matrix product of the two tiles.
           - a_tile is (BLOCK_M, BLOCK_K), b_tile is (BLOCK_K, BLOCK_N).
           - Result is (BLOCK_M, BLOCK_N), added to the accumulator.
           - After the K-loop completes, acc holds the full matmul result for
             this output tile.

           Why tiling works (data reuse):
           - Each element of a_tile is used BLOCK_N times (once per column of b_tile).
           - Each element of b_tile is used BLOCK_M times (once per row of a_tile).
           - Without tiling, each element would be loaded from global memory each
             time it's needed → BLOCK_M * BLOCK_N times slower.
           - The Triton compiler places tiles in shared memory or registers for reuse.

        8. APPLY FUSED RELU ACTIVATION
           acc = tl.maximum(acc, 0.0)

           - This is the "fusion" part: ReLU is applied to the matmul result
             BEFORE writing to global memory.
           - Without fusion, you'd write the matmul result, then read it back
             to apply ReLU, then write it again. That's 2 extra global memory
             transfers of the entire output matrix.
           - tl.maximum(acc, 0.0) computes max(x, 0) element-wise = ReLU.

        9. STORE THE OUTPUT TILE
           c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
           c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
           tl.store(c_ptrs, acc, mask=c_mask)

           - Same pointer arithmetic pattern as loading.
           - (BLOCK_M, BLOCK_N) grid of pointers into the output matrix.
           - Masking for boundary tiles (when M or N is not a multiple of tile size).
        """
        # ---- STUB: remove this and implement the steps above ----
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        # Stub: store zeros instead of the actual fused matmul+ReLU result
        tl.store(c_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32), mask=c_mask)

else:
    def triton_matmul_relu_kernel(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Triton is not available. Run inside the Docker container with GPU access.")


# ---------------------------------------------------------------------------
# Launch function (fully implemented)
# ---------------------------------------------------------------------------

def run_phase() -> None:
    """Launch the fused matmul+ReLU kernel, compare with PyTorch, print results."""
    if not TRITON_AVAILABLE:
        print("[Phase 3 — MatMul+ReLU] Triton not available. Run inside Docker with GPU.")
        return

    print("=" * 60)
    print("Phase 3: Fused MatMul + ReLU")
    print("=" * 60)

    # --- Configuration ---
    M, K, N = 512, 256, 384  # Intentionally NOT powers of 2 to exercise masking
    BLOCK_M: int = 64
    BLOCK_N: int = 64
    BLOCK_K: int = 32

    # --- Create input matrices on GPU ---
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    # --- Allocate output on GPU ---
    C_triton = torch.empty(M, N, device="cuda", dtype=torch.float32)

    # --- 2D Grid: one program per output tile ---
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,  # Number of row tiles
        (N + BLOCK_N - 1) // BLOCK_N,   # Number of column tiles
    )

    # --- Launch kernel ---
    triton_matmul_relu_kernel[grid](
        A, B, C_triton,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C_triton.stride(0), C_triton.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # --- Verify against PyTorch ---
    C_torch = torch.relu(A @ B)
    max_diff = (C_triton - C_torch).abs().max().item()

    # Check some statistics
    pct_zero = (C_triton == 0).float().mean().item() * 100

    print(f"  Matrix shapes:   A({M}, {K}) @ B({K}, {N}) -> C({M}, {N})")
    print(f"  Tile sizes:      BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
    print(f"  Grid:            ({grid[0]}, {grid[1]}) = {grid[0] * grid[1]} programs")
    print(f"  Max difference:  {max_diff:.2e}")
    print(f"  ReLU zeros:      {pct_zero:.1f}% of output is zero (expected ~50%)")

    if max_diff < 1e-3:  # Matmul accumulation has more floating-point error
        print("  PASS: Triton fused matmul+ReLU matches PyTorch reference.")
    else:
        print("  FAIL: Output mismatch! Check your kernel implementation.")
        print("  Hint: Make sure the K-loop accumulates correctly and ReLU is applied last.")

    print()


if __name__ == "__main__":
    run_phase()
