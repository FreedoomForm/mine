# BioCUDA v39.1

BioCUDA v39.1 is a real Python package and CLI for **GPU-aware optimization of bioinformatics kernels**.
It is not a notebook demo: it ships a reusable package, a command-line interface, workload models,
GPU detection, occupancy-based pruning, architecture-aware planning, and an EXP3 autotuner.

## What is implemented

- Runtime GPU detection from `nvidia-smi`, with compute-capability matching and fallback logic.
- Full 10-GPU database: T4, V100, A100, A10, L4, L40, RTX 3090, RTX 4090, H100 SXM5, H100 PCIe.
- Cost-model formulas for occupancy, transaction count, wavefront latency, bank conflicts,
  staging thresholds, tensor-core partial-tile efficiency, and Hamming POPC prediction.
- Workload planners for:
  - Smith-Waterman
  - PWM scoring
  - profile-HMM forward
  - Hamming distance
  - suffix array construction
- Real autotuning pipeline:
  1. search-space generation
  2. infeasible-config pruning
  3. cost-model ranking
  4. EXP3 online selection
- Optional runtime benchmarking hooks for real GPU execution with user-provided callables.

## Install

```bash
pip install -e .
```

## CLI

### Detect current GPU

```bash
biocuda detect --json
```

### List known GPUs

```bash
biocuda list-gpus
```

### Plan Smith-Waterman on current GPU

```bash
biocuda plan smith-waterman --seq-len-a 4096 --seq-len-b 4096 --top-k 5 --json
```

### Plan PWM scoring

```bash
biocuda plan pwm --sequence-length 100000 --motif-length 19 --batches 512
```

### Autotune from the generated search space

```bash
biocuda autotune smith-waterman --seq-len-a 4096 --seq-len-b 4096 --rounds 50 --top-k 8 --json
```

## Examples

```bash
python examples/plan_sw.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Design goals

- production-oriented package structure
- CPU-safe by default, GPU-aware when available
- no notebook dependency
- deterministic planners and tests
- easy to integrate into larger CUDA/CuPy/PyTorch pipelines
