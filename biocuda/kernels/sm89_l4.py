"""Kernels specialized for NVIDIA L4 (Ada Lovelace, sm_89)."""
from ._common import (
    HAMMING_BASE_SRC, SW_BASE_SRC,
    render_tc_kernel, MMA_BLOCK_ADA,
)

GPU_KEY     = "L4"
SM_ARCH     = "sm_89"
CC_MAJOR    = 8
CC_MINOR    = 9
HAS_DPX     = False
HAS_TMA     = False
TC_GEN      = "ada"
MMA_SHAPE   = (16, 8, 16)
MMA_DTYPE   = "f16"

HAMMING_KERNEL_SRC  = HAMMING_BASE_SRC
SW_KERNEL_SRC       = SW_BASE_SRC
TC_CALIBRATION_SRC  = render_tc_kernel(MMA_BLOCK_ADA)

BUILD_OPTS = ("-arch=sm_89", "--use_fast_math", "-lineinfo")
