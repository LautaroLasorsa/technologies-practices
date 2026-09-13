"""Package init.

Disables PyTensor's C-compilation backend *before* PyMC (or anything else in
this package) can trigger it. On some Windows setups, PyTensor's default C
linker calls out to a system C++ compiler even for trivial graph-shape ops
at model-*construction* time (not just at sampling time) — if that compiler
toolchain is broken or fights with real-time antivirus scanning of freshly
written `.pyd` files, model building itself fails with a linker error,
before a single MCMC step has run. Setting `cxx = ""` forces PyTensor onto
its pure-Python fallback for graph-level bookkeeping, which is fast enough
that it is not the bottleneck here. The actual expensive step — NUTS
sampling — never depends on this: every phase's `pm.sample(...,
nuts_sampler="nutpie")` call goes through `nutpie`'s own prebuilt Rust
backend instead, which needs no C compiler either. Together, the whole
practice runs with zero dependency on a working system C/C++ toolchain.
"""
import pytensor

pytensor.config.cxx = ""
