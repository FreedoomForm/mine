"""Kernels specialized for Tesla V100 (Volta, sm_70)."""
from ._common import (
    HAMMING_BASE_SRC, SW_BASE_SRC,
    render_tc_kernel, MMA_BLOCK_VOLTA,
)

GPU_KEY     = "V100"
SM_ARCH     = "sm_70"
CC_MAJOR    = 7
CC_MINOR    = 0
HAS_DPX     = False
HAS_TMA     = False
TC_GEN      = "volta"
MMA_SHAPE   = (8, 8, 4)
MMA_DTYPE   = "f16"

HAMMING_KERNEL_SRC  = HAMMING_BASE_SRC
SW_KERNEL_SRC       = SW_BASE_SRC
TC_CALIBRATION_SRC  = render_tc_kernel(MMA_BLOCK_VOLTA)

BUILD_OPTS = ("-arch=sm_70", "--use_fast_math", "-lineinfo")
