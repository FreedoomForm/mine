# Changelog

## v39.2.0 (notebook)
- Added `BioCUDA_v39.ipynb` with full v38-spec coverage.
- All 35 G-formulas (G1..G35) implemented in a single `FormulaEngine` class.
- Falsification suite: 60 tests across tiers T0 (algebraic) / N (hardware) / C (cross-consistency).
  All 60 pass in CPU-simulation mode; on Kaggle GPU the additional N_HW_* tests verify against live measurements.
- Real CuPy microbenchmarks: HBM bandwidth, τ_shfl, τ_smem, L2 warmup.
- GPU Hamming kernel as `cupy.RawKernel` with on-device verification against CPU reference.
- Autotuner (G34) with `|Omega|` ranging 1280–1920 depending on GPU capability set.
- EXP3 online selection (G22) with optimal eta and regret bound.
- Cross-GPU adaptation table for all 10 GPUs.

## v39.1.0 (package)
- Real Python package `biocuda` with CLI.
- 10-GPU specification database.
- Planner + autotuner + EXP3 tuner.
- Unit tests (7/7 OK).
