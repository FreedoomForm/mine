from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class GPUSpec:
    name: str
    arch: str
    compute_capability: Tuple[int, int]
    n_sm: int
    fp32_cores_per_sm: int
    fp64_cores_per_sm: int
    tensor_cores_per_sm: int
    warp_size: int = 32
    max_warps_per_sm: int = 64
    max_threads_per_sm: int = 2048
    max_blocks_per_sm: int = 32
    max_threads_per_block: int = 1024
    registers_per_sm: int = 65536
    register_granularity: int = 256
    max_registers_per_thread: int = 255
    shared_mem_per_sm_bytes: int = 65536
    l2_cache_bytes: int = 4 * 1024 * 1024
    hbm_bandwidth_bytes_per_sec: float = 320e9
    memory_type: str = "HBM2"
    memory_size_gb: int = 16
    cache_line_bytes: int = 128
    boost_clock_ghz: float = 1.5
    tc_shape: Tuple[int, int, int] = (16, 16, 16)
    has_dpx: bool = False
    has_tma: bool = False
    tdp_watts: int = 300
    tau_reg: int = 4
    tau_shfl: int = 4
    tau_smem: int = 23
    tau_l2: int = 193
    tau_hbm: int = 600
    tau_tc: int = 16
    tau_dpx: int = 0
    n_issue_ports: int = 4
    smem_banks: int = 32
    smem_bank_width_bytes: int = 4

    def to_dict(self) -> dict:
        result = asdict(self)
        result["compute_capability"] = list(self.compute_capability)
        result["tc_shape"] = list(self.tc_shape)
        return result


GPU_DATABASE: Dict[str, GPUSpec] = {
    "T4": GPUSpec(
        name="Tesla T4", arch="turing", compute_capability=(7, 5), n_sm=40,
        fp32_cores_per_sm=64, fp64_cores_per_sm=2, tensor_cores_per_sm=8,
        max_blocks_per_sm=16, shared_mem_per_sm_bytes=65536, l2_cache_bytes=4 * 1024 * 1024,
        hbm_bandwidth_bytes_per_sec=320e9, memory_type="GDDR6", memory_size_gb=16,
        boost_clock_ghz=1.59, tdp_watts=70, tau_shfl=5, tau_smem=28, tau_l2=200, tau_hbm=450,
    ),
    "V100": GPUSpec(
        name="Tesla V100 SXM2", arch="volta", compute_capability=(7, 0), n_sm=80,
        fp32_cores_per_sm=64, fp64_cores_per_sm=32, tensor_cores_per_sm=8,
        shared_mem_per_sm_bytes=98304, l2_cache_bytes=6 * 1024 * 1024,
        hbm_bandwidth_bytes_per_sec=900e9, memory_type="HBM2", memory_size_gb=16,
        boost_clock_ghz=1.53, tdp_watts=300, tau_smem=24, tau_l2=180, tau_hbm=550,
    ),
    "A100": GPUSpec(
        name="A100 SXM4 80GB", arch="ampere", compute_capability=(8, 0), n_sm=108,
        fp32_cores_per_sm=64, fp64_cores_per_sm=32, tensor_cores_per_sm=4,
        shared_mem_per_sm_bytes=167936, l2_cache_bytes=40 * 1024 * 1024,
        hbm_bandwidth_bytes_per_sec=2039e9, memory_type="HBM2e", memory_size_gb=80,
        boost_clock_ghz=1.41, tdp_watts=400, tau_smem=22, tau_l2=180, tau_hbm=500,
    ),
    "A10": GPUSpec(
        name="A10", arch="ampere", compute_capability=(8, 6), n_sm=72,
        fp32_cores_per_sm=128, fp64_cores_per_sm=2, tensor_cores_per_sm=4,
        max_blocks_per_sm=16, shared_mem_per_sm_bytes=102400, l2_cache_bytes=6 * 1024 * 1024,
        hbm_bandwidth_bytes_per_sec=600e9, memory_type="GDDR6", memory_size_gb=24,
        boost_clock_ghz=1.70, tdp_watts=150, tau_shfl=5, tau_smem=25, tau_l2=190, tau_hbm=400,
    ),
    "L4": GPUSpec(
        name="L4", arch="ada", compute_capability=(8, 9), n_sm=60,
        fp32_cores_per_sm=128, fp64_cores_per_sm=2, tensor_cores_per_sm=4,
        max_blocks_per_sm=24, shared_mem_per_sm_bytes=102400, l2_cache_bytes=48 * 1024 * 1024,
        hbm_bandwidth_bytes_per_sec=300e9, memory_type="GDDR6", memory_size_gb=24,
        boost_clock_ghz=2.04, tdp_watts=72, tau_shfl=5, tau_smem=24, tau_l2=160, tau_hbm=380, tau_tc=14,
    ),
    "L40": GPUSpec(
        name="L40", arch="ada", compute_capability=(8, 9), n_sm=142,
        fp32_cores_per_sm=128, fp64_cores_per_sm=2, tensor_cores_per_sm=4,
        max_blocks_per_sm=24, shared_mem_per_sm_bytes=102400, l2_cache_bytes=96 * 1024 * 1024,
        hbm_bandwidth_bytes_per_sec=864e9, memory_type="GDDR6X", memory_size_gb=48,
        boost_clock_ghz=2.49, tdp_watts=300, tau_shfl=5, tau_smem=22, tau_l2=150, tau_hbm=350, tau_tc=14,
    ),
    "RTX3090": GPUSpec(
        name="GeForce RTX 3090", arch="ampere", compute_capability=(8, 6), n_sm=82,
        fp32_cores_per_sm=128, fp64_cores_per_sm=2, tensor_cores_per_sm=4,
        max_blocks_per_sm=16, shared_mem_per_sm_bytes=102400, l2_cache_bytes=6 * 1024 * 1024,
        hbm_bandwidth_bytes_per_sec=936e9, memory_type="GDDR6X", memory_size_gb=24,
        boost_clock_ghz=1.70, tdp_watts=350, tau_shfl=5, tau_smem=25, tau_l2=190, tau_hbm=380,
    ),
    "RTX4090": GPUSpec(
        name="GeForce RTX 4090", arch="ada", compute_capability=(8, 9), n_sm=128,
        fp32_cores_per_sm=128, fp64_cores_per_sm=2, tensor_cores_per_sm=4,
        max_blocks_per_sm=24, shared_mem_per_sm_bytes=102400, l2_cache_bytes=72 * 1024 * 1024,
        hbm_bandwidth_bytes_per_sec=1008e9, memory_type="GDDR6X", memory_size_gb=24,
        boost_clock_ghz=2.52, tdp_watts=450, tau_shfl=5, tau_smem=22, tau_l2=150, tau_hbm=350, tau_tc=14,
    ),
    "H100_SXM": GPUSpec(
        name="H100 SXM5", arch="hopper", compute_capability=(9, 0), n_sm=132,
        fp32_cores_per_sm=128, fp64_cores_per_sm=64, tensor_cores_per_sm=4,
        shared_mem_per_sm_bytes=233472, l2_cache_bytes=50 * 1024 * 1024,
        hbm_bandwidth_bytes_per_sec=3350e9, memory_type="HBM3", memory_size_gb=80,
        boost_clock_ghz=1.83, has_dpx=True, has_tma=True, tdp_watts=700,
        tau_smem=23, tau_l2=193, tau_hbm=600, tau_dpx=2,
    ),
    "H100_PCIe": GPUSpec(
        name="H100 PCIe", arch="hopper", compute_capability=(9, 0), n_sm=114,
        fp32_cores_per_sm=128, fp64_cores_per_sm=64, tensor_cores_per_sm=4,
        shared_mem_per_sm_bytes=233472, l2_cache_bytes=50 * 1024 * 1024,
        hbm_bandwidth_bytes_per_sec=2000e9, memory_type="HBM2e", memory_size_gb=80,
        boost_clock_ghz=1.62, has_dpx=True, has_tma=True, tdp_watts=350,
        tau_smem=23, tau_l2=193, tau_hbm=600, tau_dpx=2,
    ),
}
