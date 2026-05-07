"""BioCUDA package."""

from .autotune import EXP3Tuner, autotune_plan
from .detect import DetectionResult, detect_current_gpu
from .formulas import FormulaEngine
from .planner import OptimizationPlanner
from .specs import GPUSpec, GPU_DATABASE
from .workloads import Workload, WorkloadType
from . import kernels
from . import tier_m

__all__ = [
    "EXP3Tuner",
    "FormulaEngine",
    "GPUSpec",
    "GPU_DATABASE",
    "DetectionResult",
    "OptimizationPlanner",
    "Workload",
    "WorkloadType",
    "autotune_plan",
    "detect_current_gpu",
    "kernels",
    "tier_m",
]

__version__ = "39.3.0"
