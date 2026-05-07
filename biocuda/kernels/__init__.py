"""
biocuda.kernels
===============

Per-GPU CUDA kernel registry with automatic arch selection.

Each of the 10 supported NVIDIA GPUs has its own kernel module that is
specialized for that target's compute capability (sm_XX), shared memory
size, and tensor-core generation. The :func:`select_kernel_module` function
returns the right module based on the matched GPU profile key.

Supported targets (10 GPUs):
    T4, V100, A100, A10, L4, L40, RTX3090, RTX4090, H100_SXM5, H100_PCIE

Each per-GPU module exposes:
    SM_ARCH      : str  -- e.g. "sm_75"
    CC_MAJOR     : int
    CC_MINOR     : int
    HAS_DPX      : bool
    HAS_TMA      : bool
    TC_GEN       : str  -- "volta" / "turing" / "ampere" / "ada" / "hopper"
    MMA_SHAPE    : tuple -- (M, N, K) of native PTX mma.sync instruction
    MMA_DTYPE    : str  -- "f16" / "bf16" / "tf32"
    HAMMING_KERNEL_SRC : str  -- CUDA source for popc-based Hamming kernel
    SW_KERNEL_SRC      : str  -- Smith-Waterman wavefront kernel
    TC_CALIBRATION_SRC : str  -- chained mma.sync PTX kernel for tau_TC
    BUILD_OPTS         : tuple -- nvrtc options including -arch=sm_XX

The shared infrastructure (compile, run, fall back) lives in
:mod:`biocuda.kernels.dispatch`.
"""
from __future__ import annotations

from . import sm75_t4
from . import sm70_v100
from . import sm80_a100
from . import sm86_a10
from . import sm89_l4
from . import sm89_l40
from . import sm86_rtx3090
from . import sm89_rtx4090
from . import sm90_h100_sxm5
from . import sm90_h100_pcie

# Mapping from canonical GPU spec key -> kernel module
KERNEL_MODULES = {
    "T4":         sm75_t4,
    "V100":       sm70_v100,
    "A100":       sm80_a100,
    "A10":        sm86_a10,
    "L4":         sm89_l4,
    "L40":        sm89_l40,
    "RTX3090":    sm86_rtx3090,
    "RTX4090":    sm89_rtx4090,
    "H100_SXM5":  sm90_h100_sxm5,
    "H100_PCIE":  sm90_h100_pcie,
}


def select_kernel_module(gpu_key: str):
    """Return the per-GPU kernel module for ``gpu_key`` (10 supported)."""
    if gpu_key not in KERNEL_MODULES:
        raise KeyError(
            f"unsupported gpu_key {gpu_key!r}; "
            f"supported keys: {sorted(KERNEL_MODULES)}"
        )
    return KERNEL_MODULES[gpu_key]


def list_supported_keys():
    return sorted(KERNEL_MODULES)
