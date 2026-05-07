"""Kernels specialized for H100 PCIe (Hopper, sm_90a).

Same ISA as the SXM5 part; the difference is power/clock/bandwidth
(captured in the spec database, not in the kernel source).
"""
from .sm90_h100_sxm5 import (
    HAMMING_KERNEL_SRC, SW_KERNEL_SRC, TC_CALIBRATION_SRC, BUILD_OPTS,
    MMA_SHAPE, MMA_DTYPE, TC_GEN, HAS_DPX, HAS_TMA,
)

GPU_KEY  = "H100_PCIE"
SM_ARCH  = "sm_90a"
CC_MAJOR = 9
CC_MINOR = 0
