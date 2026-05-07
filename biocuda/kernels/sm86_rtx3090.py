"""Kernels specialized for GeForce RTX 3090 (Ampere, sm_86)."""
from ._common import (
    HAMMING_BASE_SRC, SW_BASE_SRC,
    render_tc_kernel, MMA_BLOCK_AMPERE,
)

GPU_KEY     = "RTX3090"
SM_ARCH     = "sm_86"
CC_MAJOR    = 8
CC_MINOR    = 6
HAS_DPX     = False
HAS_TMA     = False
TC_GEN      = "ampere"
MMA_SHAPE   = (16, 8, 16)
MMA_DTYPE   = "f16"

HAMMING_KERNEL_SRC  = HAMMING_BASE_SRC
SW_KERNEL_SRC       = SW_BASE_SRC
TC_CALIBRATION_SRC  = render_tc_kernel(MMA_BLOCK_AMPERE)

BUILD_OPTS = ("-arch=sm_86", "--use_fast_math", "-lineinfo")
