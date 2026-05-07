"""
biocuda.kernels.dispatch
========================

Compile-and-run helpers that pick the right per-GPU kernel module given
the detected GPU and run real measurements:

  * compile_kernel(gpu_key, src, name, extra_opts=())
  * run_hamming(gpu_key, a, b)
  * run_sw(gpu_key, A, B, match, mismatch, gap_open, gap_extend, block=128)
  * calibrate_tau_tc(gpu_key, iters=4096, blocks=4, warps_per_block=4)

All functions degrade gracefully if CuPy/NVRTC are unavailable: they raise
a clear ``KernelUnavailable`` exception that the notebook layer can catch
and translate into a ``"available": false`` field, so the rest of the
falsification suite still runs.
"""
from __future__ import annotations
import os
from typing import Optional, Tuple

from . import select_kernel_module

try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:  # pragma: no cover -- runtime guard
    _cp = None
    _HAS_CUPY = False


class KernelUnavailable(RuntimeError):
    """Raised when the GPU/CuPy stack cannot run the requested kernel."""


def _require_cupy():
    if not _HAS_CUPY:
        raise KernelUnavailable("cupy is not importable in this environment")
    if _cp.cuda.runtime.getDeviceCount() == 0:
        raise KernelUnavailable("no CUDA device visible to cupy")


def compile_kernel(gpu_key: str, src: str, name: str,
                   extra_opts: Tuple[str, ...] = ()):
    """Compile ``src`` with the per-GPU build options for ``gpu_key``.

    Returns a :class:`cupy.RawKernel`.
    """
    _require_cupy()
    mod = select_kernel_module(gpu_key)
    opts = tuple(mod.BUILD_OPTS) + tuple(extra_opts)
    try:
        return _cp.RawKernel(src, name, options=opts, backend="nvrtc")
    except _cp.cuda.compiler.CompileException as exc:
        raise KernelUnavailable(
            f"NVRTC failed to compile {name} for {gpu_key}/{mod.SM_ARCH}: "
            f"{exc}"
        ) from exc


# ----------------------------------------------------------------------
#  Hamming
# ----------------------------------------------------------------------
def run_hamming(gpu_key: str, a_words, b_words) -> int:
    _require_cupy()
    mod = select_kernel_module(gpu_key)
    kern = compile_kernel(gpu_key, mod.HAMMING_KERNEL_SRC,
                          "hamming_popc_kernel")
    a = _cp.asarray(a_words, dtype=_cp.uint32)
    b = _cp.asarray(b_words, dtype=_cp.uint32)
    n = int(a.size)
    out = _cp.zeros(1, dtype=_cp.uint64)
    threads = 256
    blocks  = max(1, min(1024, (n + threads - 1) // threads))
    kern((blocks,), (threads,), (a, b, out, _cp.int32(n)))
    _cp.cuda.runtime.deviceSynchronize()
    return int(out.get()[0])


# ----------------------------------------------------------------------
#  Smith-Waterman
# ----------------------------------------------------------------------
def run_sw(gpu_key: str, A: bytes, B: bytes,
           match: int = 2, mismatch: int = -1,
           gap_open: int = 2, gap_extend: int = 1,
           block: int = 128) -> int:
    _require_cupy()
    mod = select_kernel_module(gpu_key)
    kern = compile_kernel(gpu_key, mod.SW_KERNEL_SRC, "sw_wavefront_kernel")
    A_d = _cp.asarray(bytearray(A), dtype=_cp.uint8)
    B_d = _cp.asarray(bytearray(B), dtype=_cp.uint8)
    out = _cp.zeros(1, dtype=_cp.int32)
    smem_bytes = (len(A) + 1) * 2 * 4
    kern((1,), (block,),
         (A_d, B_d, _cp.int32(len(A)), _cp.int32(len(B)),
          _cp.int32(match), _cp.int32(mismatch),
          _cp.int32(gap_open), _cp.int32(gap_extend),
          out),
         shared_mem=smem_bytes)
    _cp.cuda.runtime.deviceSynchronize()
    return int(out.get()[0])


# ----------------------------------------------------------------------
#  τ_TC calibration via chained mma.sync (PTX inline asm)
# ----------------------------------------------------------------------
def calibrate_tau_tc(gpu_key: str,
                     iters: int = 4096,
                     blocks: int = 4,
                     warps_per_block: int = 4,
                     repeats: int = 5) -> dict:
    """Run the chained-mma.sync calibration kernel and report cycles/MMA.

    Returns a dict with:
        sm_arch, mma_shape, mma_dtype, tc_gen,
        iters, cycles_total, tau_tc_cycles_per_mma,
        tflops_estimated, sm_clock_ghz_assumed,
        notes
    """
    _require_cupy()
    mod = select_kernel_module(gpu_key)
    kern = compile_kernel(gpu_key, mod.TC_CALIBRATION_SRC,
                          "tc_calibrate_kernel")
    out = _cp.zeros(2, dtype=_cp.uint64)
    threads = 32 * warps_per_block

    # warm-up
    kern((blocks,), (threads,), (out, _cp.int32(iters)))
    _cp.cuda.runtime.deviceSynchronize()

    cycle_samples = []
    for _ in range(repeats):
        out[0] = 0
        kern((blocks,), (threads,), (out, _cp.int32(iters)))
        _cp.cuda.runtime.deviceSynchronize()
        cycle_samples.append(int(out.get()[0]))

    cycle_samples.sort()
    median_cycles = cycle_samples[len(cycle_samples) // 2]
    tau_tc = median_cycles / max(1, iters)

    # FLOP/instruction for an MMA: 2 * M * N * K
    M, N, K = mod.MMA_SHAPE
    flops_per_mma = 2 * M * N * K

    # Approximate SM clock from device props (Hz)
    props = _cp.cuda.runtime.getDeviceProperties(0)
    sm_clock_hz = float(props["clockRate"]) * 1e3   # kHz -> Hz
    sm_count    = int(props["multiProcessorCount"])

    # Per-warp throughput in mma/sec, then aggregate to per-SM (4 warps),
    # then to whole GPU. This is a rough peak estimator from τ_TC.
    if tau_tc > 0:
        mma_per_sec_per_warp = sm_clock_hz / tau_tc
        # Each SM has 4 tensor-core schedulers (Volta+); we use 4 warps.
        mma_per_sec_total = mma_per_sec_per_warp * 4 * sm_count
        tflops_estimated = mma_per_sec_total * flops_per_mma / 1e12
    else:
        tflops_estimated = float("nan")

    return {
        "gpu_key": gpu_key,
        "sm_arch": mod.SM_ARCH,
        "tc_gen":  mod.TC_GEN,
        "mma_shape": mod.MMA_SHAPE,
        "mma_dtype": mod.MMA_DTYPE,
        "iters": int(iters),
        "cycles_samples": cycle_samples,
        "cycles_median": int(median_cycles),
        "tau_tc_cycles_per_mma": float(tau_tc),
        "flops_per_mma": int(flops_per_mma),
        "sm_clock_hz": float(sm_clock_hz),
        "sm_count": int(sm_count),
        "tflops_estimated_peak": float(tflops_estimated),
        "notes": (
            "tau_TC is the cycle cost of one chained mma.sync per warp "
            "scheduler (instruction issue latency). "
            "tflops_estimated_peak assumes 4 warp-scheduler / SM utilization."
        ),
    }
