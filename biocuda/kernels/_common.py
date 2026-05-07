"""
Shared CUDA source fragments used by every per-GPU kernel module.

Per-GPU modules wrap these with their own ``-arch=sm_XX`` and any
arch-specific tweaks (DPX intrinsics on sm_90, vectorized loads on sm_80+,
etc.).
"""
from __future__ import annotations

# ----------------------------------------------------------------------
#  Hamming distance with popc -- works on all sm_70+ targets.
# ----------------------------------------------------------------------
HAMMING_BASE_SRC = r"""
extern "C" __global__
void hamming_popc_kernel(const unsigned int* __restrict__ a,
                         const unsigned int* __restrict__ b,
                         unsigned long long* __restrict__ partial,
                         int n_words)
{
    unsigned int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride= blockDim.x * gridDim.x;
    unsigned long long acc = 0ULL;
    for (unsigned int i = tid; i < (unsigned)n_words; i += stride) {
        unsigned int xa = a[i];
        unsigned int xb = b[i];
        acc += (unsigned long long) __popc(xa ^ xb);
    }
    // warp reduction
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(partial, acc);
    }
}
"""

# ----------------------------------------------------------------------
#  Smith-Waterman wavefront kernel (anti-diagonal). Uses shared memory.
#  Works on sm_70+ (uses __shfl_sync). For sm_90 we have a DPX variant
#  defined in the H100 modules.
# ----------------------------------------------------------------------
SW_BASE_SRC = r"""
extern "C" __global__
void sw_wavefront_kernel(const unsigned char* __restrict__ A,
                         const unsigned char* __restrict__ B,
                         int la, int lb,
                         int match, int mismatch, int gap_open, int gap_extend,
                         int* __restrict__ out_score)
{
    extern __shared__ int smem[];
    int* H = smem;                  // (la+1)
    int* E = H + (la + 1);          // (la+1)
    int tid = threadIdx.x;
    int bs  = blockDim.x;
    int best = 0;

    for (int i = tid; i <= la; i += bs) { H[i] = 0; E[i] = 0; }
    __syncthreads();

    for (int j = 1; j <= lb; ++j) {
        int prev_diag = 0;
        int prev_left = 0;
        unsigned char bj = B[j-1];
        for (int i = 1; i <= la; ++i) {
            int diag = H[i-1];
            int up   = H[i];
            int s = (A[i-1] == bj) ? match : mismatch;
            int F  = max(up - gap_open, 0) - gap_extend;
            int Eij= max(prev_left - gap_open, E[i] - gap_extend);
            int v  = max(0, max(diag + s, max(F, Eij)));
            prev_diag = up;
            prev_left = v;
            E[i] = Eij;
            H[i] = v;
            if (v > best) best = v;
        }
        __syncthreads();
    }

    // warp reduction of best
    for (int off = 16; off > 0; off >>= 1) {
        int other = __shfl_xor_sync(0xffffffffu, best, off);
        if (other > best) best = other;
    }
    if (tid == 0) atomicMax(out_score, best);
}
"""

# ----------------------------------------------------------------------
#  τ_TC calibration kernel — chained mma.sync.aligned via PTX inline asm.
#  Each GPU's module picks the right mnemonic for its tensor core gen.
#  The kernel runs ITERS chained instructions on a single warp,
#  measures the cycle delta with %clock64, and the host divides by
#  ITERS * #ops to get τ_TC (cycles per MMA).
#
#  We expose four flavors via a #define switch:
#     MMA_VARIANT == 0  -> sm_70/Volta:   m8n8k4 f16
#     MMA_VARIANT == 1  -> sm_75/Turing:  m16n8k8 f16
#     MMA_VARIANT == 2  -> sm_80/Ampere+: m16n8k16 f16
#     MMA_VARIANT == 3  -> sm_89/Ada:     m16n8k16 f16
#     MMA_VARIANT == 4  -> sm_90/Hopper:  m16n8k16 f16 (wgmma is async, we
#                                         keep the warp-level mma here for a
#                                         cycle-accurate τ_TC measurement)
# ----------------------------------------------------------------------
TC_CALIBRATION_SRC_TEMPLATE = r"""
#include <cuda_fp16.h>

extern "C" __global__
void tc_calibrate_kernel(unsigned long long* __restrict__ cycles_out,
                         int iters)
{
    // each warp does its own chain
    int lane = threadIdx.x & 31;

    // 4 fp16x2 input registers for A, 2 for B, 4 fp32 accumulators
    unsigned int a0=0x3c003c00u, a1=0x3c003c00u, a2=0x3c003c00u, a3=0x3c003c00u;
    unsigned int b0=0x3c003c00u, b1=0x3c003c00u;
    float c0=0.f, c1=0.f, c2=0.f, c3=0.f;

    // warm-up
    for (int i = 0; i < 4; ++i) {
{MMA_BLOCK}
    }

    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
{MMA_BLOCK}
    }
    unsigned long long t1 = clock64();

    // prevent dead-code elimination
    if (lane == 0 && (c0+c1+c2+c3) == 1234567.0f) {
        cycles_out[1] = (unsigned long long)(c0+c1+c2+c3);
    }
    if (lane == 0 && blockIdx.x == 0) {
        cycles_out[0] = t1 - t0;
    }
}
"""

# Per-architecture PTX mma blocks
MMA_BLOCK_VOLTA = r"""        asm volatile (
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
            " {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(b0));"""

MMA_BLOCK_TURING = r"""        asm volatile (
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            " {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(b0));"""

MMA_BLOCK_AMPERE = r"""        asm volatile (
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));"""

# Ada uses the same m16n8k16 mma.sync as Ampere
MMA_BLOCK_ADA   = MMA_BLOCK_AMPERE
# Hopper still supports warp-level m16n8k16; the async wgmma is a
# separate fast path. For τ_TC cycle measurement we want a synchronous
# instruction, so we keep mma.sync.
MMA_BLOCK_HOPPER = MMA_BLOCK_AMPERE


def render_tc_kernel(mma_block: str) -> str:
    return TC_CALIBRATION_SRC_TEMPLATE.replace("{MMA_BLOCK}", mma_block)
