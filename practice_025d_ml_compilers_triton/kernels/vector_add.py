"""Phase 1: Vector Add Kernel — The "Hello World" of Triton GPU Programming.

This module implements element-wise vector addition: C = A + B.

Concept overview:
    In Triton, you don't program individual GPU threads like in CUDA. Instead,
    you program at the **block level**: each kernel instance (called a "program")
    processes a contiguous block of BLOCK_SIZE elements. The Triton compiler
    automatically maps your block-level code to the underlying GPU threads,
    handles shared memory, and optimizes memory access patterns (coalescing).

    Think of it like this: if you have a vector of 10,000 elements and
    BLOCK_SIZE=1024, Triton launches ceil(10000/1024) = 10 program instances.
    Program 0 handles elements [0..1023], program 1 handles [1024..2047], etc.
    The last program uses a **mask** to avoid accessing elements beyond index 9999.

References:
    - https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
"""

from __future__ import annotations

import torch

# Guard Triton imports so the file can be read/parsed on Windows (no GPU).
# The actual kernel execution requires Linux + NVIDIA GPU (run via Docker).
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
    def triton_vector_add_kernel(
        a_ptr,        # Pointer to first input vector A in GPU global memory
        b_ptr,        # Pointer to second input vector B in GPU global memory
        c_ptr,        # Pointer to output vector C in GPU global memory
        n_elements,   # Total number of elements in the vectors (Python int → Triton scalar)
        BLOCK_SIZE: tl.constexpr,  # Number of elements each program instance processes.
                                   # tl.constexpr means this is a compile-time constant —
                                   # Triton generates specialized PTX code for this value.
    ):
        """Element-wise vector addition: C[i] = A[i] + B[i].

        TODO(human): Replace the stub body below with the actual kernel logic.

        This kernel is called once per "program instance". Each instance handles
        a contiguous block of BLOCK_SIZE elements from the input vectors.

        Step-by-step guide:

        1. GET YOUR PROGRAM ID
           pid = tl.program_id(axis=0)

           - tl.program_id(axis=0) returns the index of THIS program instance
             along axis 0 of the launch grid.
           - If the grid has 10 programs, pid ranges from 0 to 9.
           - This is analogous to CUDA's blockIdx.x, but at a higher abstraction
             level — in CUDA you'd also need threadIdx.x; in Triton the compiler
             handles the thread-level mapping for you.

        2. COMPUTE THE OFFSETS THIS BLOCK HANDLES
           offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

           - tl.arange(0, BLOCK_SIZE) generates a vector [0, 1, 2, ..., BLOCK_SIZE-1].
           - Multiplying pid * BLOCK_SIZE gives the starting index for this block.
           - Adding tl.arange shifts that to [start, start+1, ..., start+BLOCK_SIZE-1].
           - These are the GLOBAL indices into the input arrays that this program
             instance is responsible for.

           Example with BLOCK_SIZE=4:
             Program 0: offsets = [0, 1, 2, 3]
             Program 1: offsets = [4, 5, 6, 7]
             Program 2: offsets = [8, 9, 10, 11]

        3. CREATE A MASK FOR BOUNDARY CONDITIONS
           mask = offsets < n_elements

           - When n_elements is NOT a multiple of BLOCK_SIZE, the last program
             instance would read/write beyond the end of the array without a mask.
           - Example: n_elements=10, BLOCK_SIZE=4 → Program 2 has offsets [8,9,10,11],
             but only indices 8 and 9 are valid. mask = [True, True, False, False].
           - The mask is passed to tl.load and tl.store to prevent out-of-bounds
             memory access (which would cause a GPU segfault or corrupt data).

        4. LOAD INPUT DATA FROM GLOBAL MEMORY
           a = tl.load(a_ptr + offsets, mask=mask)
           b = tl.load(b_ptr + offsets, mask=mask)

           - tl.load reads a block of data from GPU global memory.
           - a_ptr + offsets uses pointer arithmetic: a_ptr is the base address,
             offsets are the element indices. Triton handles the byte-level addressing.
           - mask=mask tells Triton to only load elements where mask is True.
             Masked-out elements get a default value of 0.
           - Under the hood, Triton generates coalesced memory transactions —
             consecutive threads read consecutive addresses, maximizing memory bandwidth.

        5. COMPUTE THE RESULT
           c = a + b

           - This is a vectorized operation on the entire block at once.
           - In CUDA you'd write this for a single element per thread; in Triton
             you express it for the whole block and the compiler handles the rest.

        6. STORE THE RESULT TO GLOBAL MEMORY
           tl.store(c_ptr + offsets, c, mask=mask)

           - tl.store writes the computed block back to GPU global memory.
           - Same pointer arithmetic and masking as tl.load.
           - The mask ensures we don't write beyond the array bounds.

        After implementing, the launch function below will verify your result
        against PyTorch's native addition. They should match exactly (for integers)
        or within floating-point tolerance (for floats).
        """
        # ---- STUB: remove this and implement the steps above ----
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        # Stub: store zeros instead of the actual sum
        tl.store(c_ptr + offsets, tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=mask)

else:
    # Placeholder when Triton is not installed (Windows / no GPU).
    def triton_vector_add_kernel(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Triton is not available. Run inside the Docker container with GPU access.")


# ---------------------------------------------------------------------------
# Launch function (fully implemented) — sets up tensors and calls the kernel
# ---------------------------------------------------------------------------

def run_phase() -> None:
    """Launch the vector add kernel, compare with PyTorch, print results."""
    if not TRITON_AVAILABLE:
        print("[Phase 1 — Vector Add] Triton not available. Run inside Docker with GPU.")
        return

    print("=" * 60)
    print("Phase 1: Vector Add Kernel")
    print("=" * 60)

    # --- Configuration ---
    n_elements: int = 98_432  # Intentionally NOT a power of 2 to exercise masking
    BLOCK_SIZE: int = 1024    # Each program instance handles 1024 elements

    # --- Create input tensors on GPU ---
    a = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    b = torch.randn(n_elements, device="cuda", dtype=torch.float32)

    # --- Allocate output tensor on GPU (uninitialized) ---
    c_triton = torch.empty_like(a)

    # --- Compute the grid size ---
    # How many program instances do we need?
    # Each handles BLOCK_SIZE elements, so we need ceil(n / BLOCK_SIZE).
    # Triton grids are specified as a tuple of ints (1D, 2D, or 3D).
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # --- Launch the kernel ---
    # This call is asynchronous on the GPU. The Python thread returns immediately
    # while the GPU executes. torch.cuda.synchronize() or any operation that reads
    # the output tensor will wait for completion.
    triton_vector_add_kernel[grid](
        a, b, c_triton,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # --- Verify against PyTorch ---
    c_torch = a + b
    max_diff = (c_triton - c_torch).abs().max().item()

    print(f"  Vector size:     {n_elements:,}")
    print(f"  Block size:      {BLOCK_SIZE}")
    print(f"  Grid size:       {grid[0]} programs")
    print(f"  Max difference:  {max_diff:.2e}")

    if max_diff < 1e-5:
        print("  PASS: Triton output matches PyTorch reference.")
    else:
        print("  FAIL: Output mismatch! Check your kernel implementation.")
        print("  Hint: Make sure you are computing c = a + b, not storing zeros.")

    print()


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_phase()
