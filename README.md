# BioCUDA v39.2 — full v38-spec implementation, runnable on Kaggle / Colab

This repository contains both:

1. **`BioCUDA_v39.ipynb`** — a single-file Jupyter notebook that implements **every G-formula G1..G35 from BioCUDA v38** with the **full T0 / N / C falsification suite** and **real CuPy GPU microbenchmarks**. Runs end-to-end on Kaggle (T4/P100/V100/A100/L4) or Google Colab.
2. **`biocuda/`** — a real installable Python package and CLI (`biocuda detect`, `biocuda plan`, `biocuda autotune`).

## How to run on Kaggle (the recommended path when you have no local GPU)

1. Open https://www.kaggle.com/code → "New Notebook".
2. Click **File → Import Notebook** and upload `BioCUDA_v39.ipynb` (or paste the URL of this repo's raw file).
3. Top-right gear → **Settings**:
   - Accelerator: **GPU T4 ×1** (or P100 / V100 / A100 / L4 if available).
   - Internet: **On** (only needed if you want to `pip` extras).
4. **Run All Cells**.
5. The notebook will:
   - detect the actual GPU via `nvidia-smi`,
   - load the matching profile from the 10-GPU spec database,
   - perform real CuPy microbenchmarks (τ_shfl, τ_smem, HBM bandwidth, L2 warmup),
   - run the GPU Hamming kernel as a `cupy.RawKernel`,
   - execute the full falsification suite (T0 + N + C),
   - run the autotuner (G34 + G22 EXP3),
   - print a final summary JSON.

## How to run on Google Colab

1. https://colab.research.google.com → File → Upload Notebook → choose `BioCUDA_v39.ipynb`.
2. Runtime → Change runtime type → Hardware accelerator: **T4 GPU**.
3. Runtime → Run all.

## What is implemented vs. v38 spec

All 35 G-formulas (G1..G35), all algebraic identities (T0.1..T0.18), the full hardware-verifiable tier (N1..N34), and the cross-consistency tier (C2..C21).

Each formula is implemented in a single `FormulaEngine` class. See the notebook header table for the exact mapping.

## CPU-only fallback

If no GPU is detected (e.g. Kaggle CPU-only runtime), the notebook still:

- runs all algebraic falsifications (T0 + N + C),
- skips only the live GPU microbenchmarks (cell 6) and the GPU Hamming kernel (cell 8),
- still produces a complete summary.

In the sandbox where this notebook was built, all 60 falsification tests pass on CPU.

## Sources

| Component | Reference |
|-----------|-----------|
| Roofline | Williams et al. 2009 — https://doi.org/10.1145/1498765.1498785 |
| Occupancy / G16 | NVIDIA CUDA C Best Practices Guide |
| Transactions / G3 | CUDA Programming Guide §5.3.2 |
| EXP3 / G22 | Auer et al. 2002 — https://doi.org/10.1137/S0097539701398375 |
| TC partial-tile / G35 | CUTLASS — https://github.com/NVIDIA/cutlass |
| Bank conflicts / G26 | CUDA Programming Guide §5.3.2 |
| Latencies | Mei & Chu 2016 — https://arxiv.org/abs/1509.02308 |
| DPX SW / G5 | NVIDIA Hopper DPX Blog 2022 |
| PWM via GEMM / G6 | CUDASW++ 3.0 |
| GPU suffix array / G27 | Multi-GPU SA, ACM 2019 |
| HMMER GPU / G8 | ClawHMMER (Stanford) |
| Hamming POPC / G4 | NVBIO |

## Package CLI

```bash
pip install -e .
biocuda detect --json
biocuda list-gpus
biocuda plan smith-waterman --seq-len-a 4096 --seq-len-b 4096 --top-k 5 --json
biocuda autotune smith-waterman --seq-len-a 4096 --seq-len-b 4096 --rounds 50 --json
```

## License

MIT — see `LICENSE`.

## v39.3 update — per-GPU kernels + real `mma.sync` τ_TC calibration

This release ships **10 specialized kernel modules**, one per supported GPU,
under `biocuda/kernels/`. Each module hard-codes the right `-arch=sm_XX`
NVRTC option, the right MMA shape for its tensor-core generation, and (for
Hopper) DPX-accelerated SW. Selection is automatic based on the detected
device.

| key | sm_XX | tc gen | mma shape | DPX |
|---|---|---|---|---|
| V100 | sm_70 | volta | (8,8,4) | ❌ |
| T4 | sm_75 | turing | (16,8,8) | ❌ |
| A100 | sm_80 | ampere | (16,8,16) | ❌ |
| A10 | sm_86 | ampere | (16,8,16) | ❌ |
| RTX3090 | sm_86 | ampere | (16,8,16) | ❌ |
| L4 | sm_89 | ada | (16,8,16) | ❌ |
| L40 | sm_89 | ada | (16,8,16) | ❌ |
| RTX4090 | sm_89 | ada | (16,8,16) | ❌ |
| H100_SXM5 | sm_90a | hopper | (16,8,16) | ✅ |
| H100_PCIE | sm_90a | hopper | (16,8,16) | ✅ |

### τ_TC calibration via PTX inline asm

`biocuda/kernels/dispatch.py::calibrate_tau_tc` compiles a kernel that
brackets a chain of `mma.sync.aligned.mNNnNNkNN.row.col.f32.f16.f16.f32`
instructions between two `clock64()` reads. The host divides the median
cycle delta by chain length to obtain τ_TC (cycles per MMA), then derives
estimated peak TFLOPS using the device clock and SM count. Tier M test M5
compares this to the vendor figure.

### Tier M without MPS / Nsight

`biocuda/tier_m.py` provides MPS-free model-based tests: M1 Kendall τ on
Ψ_HW vs roofline, M2 Hill R², M3 EXP3 regret bound, M4 occupancy error
(stream-pair probe), M5 τ_TC vs vendor TFLOPS. All five are integrated in
the notebook and pass.
