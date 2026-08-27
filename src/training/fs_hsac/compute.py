"""Device and intra-op thread pin for FS-HSAC.

Launch bats set ``OPTIMAL_DEMO_DEVICE=cpu`` and ``CUDA_VISIBLE_DEVICES=-1``
before Python starts. This helper then caps OpenMP/MKL/torch threads so three
seasonal jobs on one 32-thread CPU do not oversubscribe the FMU.
"""

from __future__ import annotations

import os


def configure_torch_compute() -> dict[str, str | int]:
    """Pin device and thread counts. Safe to call once per process."""
    import torch

    raw = (os.environ.get("OPTIMAL_DEMO_DEVICE") or "").strip().lower()
    hidden = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if raw == "cpu" or hidden in ("-1", "none"):
        device = "cpu"
    elif raw == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    threads = int(os.environ.get("OPTIMAL_DEMO_TORCH_THREADS") or os.environ.get("OMP_NUM_THREADS") or "0")
    if threads <= 0:
        threads = 6 if device == "cpu" else max(1, min(4, os.cpu_count() or 4))
    threads = max(1, threads)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(threads))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return {"device": device, "threads": threads}
