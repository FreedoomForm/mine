"""Kernels specialized for H100 SXM5 (Hopper, sm_90a).

Includes a DPX-instruction Smith-Waterman kernel (__viaddmax_s16x2_relu)
which is only available on sm_90.
"""
from ._common import (
    HAMMING_BASE_SRC, SW_BASE_SRC,
    render_tc_kernel, MMA_BLOCK_HOPPER,
)

GPU_KEY     = "H100_SXM5"
SM_ARCH     = "sm_90a"   # 'a' enables wgmma + tcgen
CC_MAJOR    = 9
CC_MINOR    = 0
HAS_DPX     = True
HAS_TMA     = True
TC_GEN      = "hopper"
MMA_SHAPE   = (16, 8, 16)
MMA_DTYPE   = "f16"

HAMMING_KERNEL_SRC = HAMMING_BASE_SRC

# DPX-accelerated SW: uses 16x2 packed signed-int max-add-relu
SW_KERNEL_SRC = r"""
extern "C" __global__
void sw_wavefront_kernel(const unsigned char* __restrict__ A,
                         const unsigned char* __restrict__ B,
                         int la, int lb,
                         int match, int mismatch, int gap_open, int gap_extend,
                         int* __restrict__ out_score)
{
    extern __shared__ int smem[];
    int* H = smem;
    int* E = H + (la + 1);
    int tid = threadIdx.x;
    int bs  = blockDim.x;
    int best = 0;

    for (int i = tid; i <= la; i += bs) { H[i] = 0; E[i] = 0; }
    __syncthreads();

    for (int j = 1; j <= lb; ++j) {
        int prev_left = 0;
        unsigned char bj = B[j-1];
        for (int i = 1; i <= la; ++i) {
            int diag = H[i-1];
            int up   = H[i];
            int s = (A[i-1] == bj) ? match : mismatch;
            // DPX: __viaddmax_s32_relu(a, b, c) == max(a + b, c, 0)
            int F   = __viaddmax_s32_relu(up,        -gap_open, -gap_extend);
            int Eij = __viaddmax_s32_relu(prev_left, -gap_open,
                          E[i] - gap_extend);
            int v   = max(0, max(diag + s, max(F, Eij)));
            prev_left = v;
            E[i] = Eij;
            H[i] = v;
            if (v > best) best = v;
        }
        __syncthreads();
    }
    for (int off = 16; off > 0; off >>= 1) {
        int other = __shfl_xor_sync(0xffffffffu, best, off);
        if (other > best) best = other;
    }
    if (tid == 0) atomicMax(out_score, best);
}
"""

TC_CALIBRATION_SRC = render_tc_kernel(MMA_BLOCK_HOPPER)

BUILD_OPTS = ("-arch=sm_90a", "--use_fast_math", "-lineinfo")
