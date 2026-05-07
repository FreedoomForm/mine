"""Kernels specialized for Tesla T4 (Turing, sm_75)."""
from ._common import (
    HAMMING_BASE_SRC, SW_BASE_SRC,
    render_tc_kernel, MMA_BLOCK_TURING,
)

GPU_KEY     = "T4"
SM_ARCH     = "sm_75"
CC_MAJOR    = 7
CC_MINOR    = 5
HAS_DPX     = False
HAS_TMA     = False
TC_GEN      = "turing"
MMA_SHAPE   = (16, 8, 8)
MMA_DTYPE   = "f16"

HAMMING_KERNEL_SRC  = HAMMING_BASE_SRC
SW_KERNEL_SRC       = SW_BASE_SRC
TC_CALIBRATION_SRC  = render_tc_kernel(MMA_BLOCK_TURING)

BUILD_OPTS = ("-arch=sm_75", "--use_fast_math", "-lineinfo")
