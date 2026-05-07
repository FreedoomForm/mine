from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass
from typing import Optional

from .specs import GPU_DATABASE, GPUSpec


@dataclass(frozen=True)
class DetectionResult:
    raw_name: str
    mode: str
    matched_key: str
    gpu: GPUSpec
    compute_capability: Optional[tuple[int, int]] = None
    total_memory_mb: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "raw_name": self.raw_name,
            "mode": self.mode,
            "matched_key": self.matched_key,
            "compute_capability": list(self.compute_capability) if self.compute_capability else None,
            "total_memory_mb": self.total_memory_mb,
            "gpu": self.gpu.to_dict(),
        }


def _run_nvidia_smi() -> Optional[str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,memory.total",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return completed.stdout.strip()
    except Exception:
        return None


def _normalize_name(name: str) -> str:
    return " ".join(name.lower().replace("-", " ").split())


def _match_name(raw_name: str) -> str:
    name = _normalize_name(raw_name)
    match_map = {
        "t4": "T4",
        "v100": "V100",
        "a100": "A100",
        "a10g": "A10",
        "a10": "A10",
        "l4": "L4",
        "l40": "L40",
        "3090": "RTX3090",
        "4090": "RTX4090",
        "h100 pcie": "H100_PCIe",
        "h100": "H100_SXM",
    }
    for needle, key in match_map.items():
        if needle in name:
            return key
    return "T4"


def detect_current_gpu() -> DetectionResult:
    raw = _run_nvidia_smi()
    if not raw:
        return DetectionResult(
            raw_name="CPU_SIMULATION",
            mode="simulation",
            matched_key="T4",
            gpu=GPU_DATABASE["T4"],
        )

    row = next(csv.reader(io.StringIO(raw)))
    raw_name = row[0].strip()
    compute_capability = None
    total_memory_mb = None
    if len(row) >= 2:
        try:
            major, minor = row[1].strip().split(".")
            compute_capability = (int(major), int(minor))
        except Exception:
            compute_capability = None
    if len(row) >= 3:
        digits = "".join(ch for ch in row[2] if ch.isdigit())
        total_memory_mb = int(digits) if digits else None

    matched_key = _match_name(raw_name)
    gpu = GPU_DATABASE[matched_key]
    if compute_capability is not None:
        for key, candidate in GPU_DATABASE.items():
            if candidate.compute_capability == compute_capability and _normalize_name(candidate.name).split()[0] in _normalize_name(raw_name):
                matched_key = key
                gpu = candidate
                break

    return DetectionResult(
        raw_name=raw_name,
        mode="detected",
        matched_key=matched_key,
        gpu=gpu,
        compute_capability=compute_capability,
        total_memory_mb=total_memory_mb,
    )
