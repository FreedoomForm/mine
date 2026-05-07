# Changelog

## v39.3.0 (current)

### Added
- **10 per-GPU CUDA kernel modules** (`biocuda/kernels/sm70_v100.py`, `sm75_t4.py`,
  `sm80_a100.py`, `sm86_a10.py`, `sm86_rtx3090.py`, `sm89_l4.py`, `sm89_l40.py`,
  `sm89_rtx4090.py`, `sm90_h100_sxm5.py`, `sm90_h100_pcie.py`) each shipping its
  own `-arch=sm_XX` build option set, MMA shape, and tensor-core generation.
- **Automatic kernel selection** via `biocuda.kernels.select_kernel_module(gpu_key)`
  driven by the auto-detect logic.
- **τ_TC calibration via chained `mma.sync` (PTX inline asm)** in
  `biocuda/kernels/dispatch.py::calibrate_tau_tc`. The kernel emits the right
  MMA mnemonic per tensor-core gen:
  - Volta `m8n8k4 f32.f16.f16.f32`
  - Turing `m16n8k8 f32.f16.f16.f32`
  - Ampere/Ada/Hopper `m16n8k16 f32.f16.f16.f32`
  Compiled with the matching `-arch=sm_XX` (`sm_70`/`75`/`80`/`86`/`89`/`90a`).
  Cycle delta is bracketed with `clock64()` and divided by chain length to
  get τ_TC (cycles/MMA); host derives estimated peak TFLOPS from device
  clock and compares to vendor figures.
- **DPX-accelerated Smith-Waterman kernel for Hopper** (sm_90a) using
  `__viaddmax_s32_relu`.
- **Tier M falsification suite** (`biocuda/tier_m.py`) with 5 model-based
  tests that *do not require MPS or Nsight Compute*:
  - M1 Kendall τ(Ψ_HW, T) > roofline τ + 0.1
  - M2 Hill response R² > 0.95
  - M3 EXP3 regret ≤ √(2·T·K·ln K) + ε
  - M4 occupancy model error < 10 % (stream-pair concurrency probe)
  - M5 τ_TC measured vs vendor TFLOPS within tolerance
- **Vendor TFLOPS table** (`biocuda/vendor_perf.py`) for all 10 GPUs.
- **Notebook v39.3** (`BioCUDA_v39.ipynb`, 12 cells) self-contained, runs
  end-to-end on Kaggle/Colab/CPU and prints final summary JSON.

### Verified
- All 60 T0+N+C falsification tests pass.
- All 5 Tier M tests pass on the simulator and at runtime.
- All 7 unit tests pass.
- Notebook executes cleanly on CPU-only environment (no cupy required).
- All 10 per-GPU kernel modules import and produce the correct `-arch=sm_XX`.

## v39.2.0
- Implemented all 35 G-formulas (G1..G35).
- T0 + N + C falsification suite (60 tests).
- Initial Kaggle/Colab notebook.

## v39.1.0
- Initial real GPU-optimization package: spec database, detection, planner,
  EXP3 autotuner, CLI.
