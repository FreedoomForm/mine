from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class BackendAvailability:
    torch: bool
    cupy: bool

    def to_dict(self) -> dict:
        return {"torch": self.torch, "cupy": self.cupy}


@dataclass(frozen=True)
class TimingResult:
    warmup_runs: int
    measured_runs: int
    min_ms: float
    median_ms: float
    mean_ms: float

    def to_dict(self) -> dict:
        return {
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "min_ms": self.min_ms,
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
        }


def available_backends() -> BackendAvailability:
    try:
        import torch  # noqa: F401
        has_torch = True
    except Exception:
        has_torch = False
    try:
        import cupy  # noqa: F401
        has_cupy = True
    except Exception:
        has_cupy = False
    return BackendAvailability(torch=has_torch, cupy=has_cupy)


def benchmark_callable(fn: Callable[[], None], warmup: int = 5, repeat: int = 20, synchronize: Optional[Callable[[], None]] = None) -> TimingResult:
    for _ in range(warmup):
        fn()
        if synchronize:
            synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        if synchronize:
            synchronize()
        end = time.perf_counter()
        samples.append((end - start) * 1000.0)
    return TimingResult(
        warmup_runs=warmup,
        measured_runs=repeat,
        min_ms=min(samples),
        median_ms=statistics.median(samples),
        mean_ms=statistics.mean(samples),
    )
