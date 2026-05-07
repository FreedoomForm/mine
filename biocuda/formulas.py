from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from .specs import GPUSpec


class FormulaEngine:
    def __init__(self, gpu: GPUSpec):
        self.gpu = gpu
        self.W = gpu.warp_size

    def complement_xor(self, values: list[float], mask: int) -> list[float]:
        return [values[index ^ mask] for index in range(len(values))]

    def verify_involution(self, values: list[float], mask: int) -> bool:
        return self.complement_xor(self.complement_xor(values, mask), mask) == values

    def optimal_stride1(self, element_size: int) -> int:
        return self.gpu.cache_line_bytes // element_size

    def transaction_count(self, k: int, stride: int, element_size: int, base_offset: int = 0) -> int:
        span = base_offset + k * element_size + (self.W - 1) * stride * element_size
        return math.ceil(span / self.gpu.cache_line_bytes)

    def hamming_2bit(self, x: int, y: int) -> int:
        xor = x ^ y
        combined = (xor | (xor >> 1)) & 0x5555555555555555
        return combined.bit_count()

    def hamming_latency(self) -> dict:
        return {
            "2bit_cycles": 5 * self.gpu.tau_reg,
            "binary_cycles": 2 * self.gpu.tau_reg,
        }

    def antidiagonal_width(self, d: int, m: int, n: int) -> int:
        return min(d + 1, m, n, m + n - d - 1)

    def wavefront_latency_lb(self) -> int:
        if self.gpu.has_dpx:
            return self.gpu.tau_shfl + self.gpu.tau_dpx + self.gpu.tau_reg
        return self.gpu.tau_shfl + 3 * self.gpu.tau_reg

    def pwm_intensity(self, motif_length: int) -> float:
        return motif_length / 2.0

    def tensor_core_peak_ops_per_cycle(self) -> int:
        if self.gpu.arch == "volta":
            return 64
        if self.gpu.arch in {"ampere", "ada"}:
            return 128
        if self.gpu.arch == "hopper":
            return 128
        return 64

    def compute_bound_threshold(self) -> float:
        phi_tc = (
            self.gpu.n_sm
            * self.gpu.tensor_cores_per_sm
            * self.tensor_core_peak_ops_per_cycle()
            * self.gpu.boost_clock_ghz
            * 1e9
        )
        return phi_tc / self.gpu.hbm_bandwidth_bytes_per_sec

    def scan_latency_lb(self) -> int:
        return int(math.log2(self.gpu.warp_size)) * self.gpu.tau_shfl

    def scan_latency_full(self) -> int:
        return 2 * self.scan_latency_lb()

    def affine_compose(self, a2: float, b2: float, a1: float, b1: float) -> tuple[float, float]:
        return (a2 * a1, a2 * b1 + b2)

    def hmm_intensity(self, n_states: int) -> float:
        return float(n_states)

    def occupancy(self, threads_per_block: int, regs_per_thread: int, smem_per_block: int) -> dict:
        warps_per_block = math.ceil(threads_per_block / self.gpu.warp_size)
        regs_per_warp = regs_per_thread * self.gpu.warp_size
        rounded_per_warp = math.ceil(regs_per_warp / self.gpu.register_granularity) * self.gpu.register_granularity
        regs_per_block = rounded_per_warp * warps_per_block
        limit_regs = self.gpu.registers_per_sm // regs_per_block if regs_per_block else self.gpu.max_blocks_per_sm
        limit_smem = self.gpu.shared_mem_per_sm_bytes // smem_per_block if smem_per_block else self.gpu.max_blocks_per_sm
        limit_blocks = self.gpu.max_blocks_per_sm
        limit_threads = self.gpu.max_threads_per_sm // threads_per_block if threads_per_block else 0
        limits = {
            "registers": limit_regs,
            "shared_memory": limit_smem,
            "max_blocks": limit_blocks,
            "max_threads": limit_threads,
        }
        b_res = min(limits.values())
        active_warps = b_res * warps_per_block
        return {
            "B_res": b_res,
            "active_warps": active_warps,
            "rho_warp": min(active_warps / self.gpu.max_warps_per_sm, 1.0),
            "limiting_factor": min(limits, key=limits.get),
            "limits": limits,
        }

    def exp3_update(self, weights: list[float], losses: list[float], eta: float) -> list[float]:
        scaled = [w * math.exp(-eta * loss) for w, loss in zip(weights, losses)]
        total = sum(scaled)
        return [value / total for value in scaled] if total else [1.0 / len(weights)] * len(weights)

    def optimal_eta(self, k: int, t_rounds: int, max_loss: float = 1.0) -> float:
        return math.sqrt(2.0 * math.log(k) / (t_rounds * max_loss * max_loss))

    def regret_bound(self, k: int, t_rounds: int) -> float:
        return math.sqrt(2.0 * t_rounds * k * math.log(k))

    def bank_conflicts(self, addresses: Iterable[int]) -> dict:
        banks = [((address // self.gpu.smem_bank_width_bytes) % self.gpu.smem_banks) for address in addresses]
        counts = Counter(banks)
        max_conflict = max(counts.values()) if counts else 1
        total_pairs = sum(n * (n - 1) for n in counts.values())
        return {
            "max_way_conflict": max_conflict,
            "serialization_factor": max_conflict,
            "total_exclusive_pairs": total_pairs,
        }

    def digit_sort_latency_lb(self) -> int:
        return self.scan_latency_lb() + self.gpu.tau_smem

    def digit_sort_latency_full(self) -> int:
        return self.scan_latency_full() + self.gpu.tau_smem

    def partial_tile_efficiency(self, rows: int, cols: int) -> float:
        tile_m, tile_n, _ = self.gpu.tc_shape
        padded_r = tile_m * math.ceil(rows / tile_m)
        padded_c = tile_n * math.ceil(cols / tile_n)
        return (rows * cols) / (padded_r * padded_c)

    def staging_thresholds(self) -> dict:
        tau_g = self.gpu.tau_hbm
        return {
            "A_min_smem": (tau_g - self.gpu.tau_smem) / self.gpu.tau_smem,
            "A_min_shfl": (tau_g - self.gpu.tau_shfl) / self.gpu.tau_shfl,
            "A_min_L2": (tau_g - self.gpu.tau_l2) / self.gpu.tau_l2,
        }
