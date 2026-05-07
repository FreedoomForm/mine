from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import List

from .formulas import FormulaEngine
from .specs import GPUSpec
from .workloads import Workload, WorkloadType


@dataclass
class CandidateConfig:
    variant: str
    block_size: int
    layout: str
    pipeline: int
    vector_width: int
    regs_per_thread: int
    smem_per_block: int
    occupancy: dict
    predicted_cycles: float
    predicted_score: float
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


@dataclass
class OptimizationPlan:
    gpu_key: str
    gpu_name: str
    workload: dict
    candidates_considered: int
    candidates_feasible: int
    best: CandidateConfig | None
    top_candidates: List[CandidateConfig]

    def to_dict(self) -> dict:
        return {
            "gpu_key": self.gpu_key,
            "gpu_name": self.gpu_name,
            "workload": self.workload,
            "candidates_considered": self.candidates_considered,
            "candidates_feasible": self.candidates_feasible,
            "best": self.best.to_dict() if self.best else None,
            "top_candidates": [candidate.to_dict() for candidate in self.top_candidates],
        }


class OptimizationPlanner:
    def __init__(self, gpu_key: str, gpu: GPUSpec):
        self.gpu_key = gpu_key
        self.gpu = gpu
        self.engine = FormulaEngine(gpu)

    def _variants(self, workload: Workload) -> list[str]:
        if workload.kind == WorkloadType.SMITH_WATERMAN:
            variants = ["warp_affine", "striped_affine"]
            if self.gpu.has_dpx:
                variants.append("dpx_native")
            return variants
        if workload.kind in {WorkloadType.PWM, WorkloadType.HMM_FORWARD}:
            variants = ["fp32_cuda", "fp16_tc"]
            if self.gpu.compute_capability >= (8, 0):
                variants.append("tf32_tc")
            return variants
        if workload.kind == WorkloadType.HAMMING:
            return ["popc_2bit", "xor_popc_binary"]
        if workload.kind == WorkloadType.SUFFIX_ARRAY:
            return ["radix_scan", "radix_scan_padded"]
        raise ValueError(f"Unsupported workload: {workload.kind}")

    def _block_sizes(self) -> list[int]:
        return [size for size in (64, 128, 256, 512, 1024) if size <= self.gpu.max_threads_per_block]

    def _layouts(self) -> list[str]:
        base = ["row_major", "padded"]
        if self.gpu.has_tma or self.gpu.l2_cache_bytes >= 40 * 1024 * 1024:
            base.append("swizzled")
        return base

    def _pipelines(self) -> list[int]:
        values = [1, 2, 4]
        if self.gpu.has_tma:
            values.append(8)
        return values

    def _regs_and_smem(self, variant: str, workload: Workload, block_size: int, pipeline: int) -> tuple[int, int]:
        if workload.kind == WorkloadType.SMITH_WATERMAN:
            base_regs = 40 if "striped" in variant else 32
            if variant == "dpx_native":
                base_regs -= 4
            smem = 2048 + pipeline * 1024 + block_size * 4
            return base_regs, smem
        if workload.kind == WorkloadType.PWM:
            motif_length = int(workload.params["motif_length"])
            tc_bonus = 8 if "tc" in variant else 0
            return 28 + tc_bonus, 4096 + pipeline * 2048 + motif_length * 16
        if workload.kind == WorkloadType.HMM_FORWARD:
            n_states = int(workload.params["n_states"])
            return 36 + (8 if "tc" in variant else 0), 8192 + pipeline * 2048 + min(n_states, 256) * 32
        if workload.kind == WorkloadType.HAMMING:
            return 16, 1024
        if workload.kind == WorkloadType.SUFFIX_ARRAY:
            return 24, 4096 + pipeline * 1024 + block_size * 8
        raise ValueError(f"Unsupported workload: {workload.kind}")

    def _predict(self, workload: Workload, variant: str, block_size: int, layout: str, pipeline: int, vector_width: int, occ: dict) -> tuple[float, float, list[str]]:
        rho = max(occ["rho_warp"], 1e-6)
        rationale: list[str] = [f"occupancy={rho:.2f}", f"limiter={occ['limiting_factor']}"]
        if workload.kind == WorkloadType.SMITH_WATERMAN:
            m = int(workload.params["seq_len_a"])
            n = int(workload.params["seq_len_b"])
            antidiags = m + n - 1
            cycles = antidiags * self.engine.wavefront_latency_lb() / rho
            if variant == "dpx_native":
                cycles *= 0.85
                rationale.append("dpx_bonus")
            if layout == "swizzled":
                cycles *= 0.96
                rationale.append("swizzled_reuse")
            cycles /= min(vector_width, 4)
            score = 1.0 / cycles
            return cycles, score, rationale
        if workload.kind == WorkloadType.PWM:
            seq_len = int(workload.params["sequence_length"])
            motif = int(workload.params["motif_length"])
            batches = int(workload.params["batches"])
            intensity = self.engine.pwm_intensity(motif)
            threshold = self.engine.compute_bound_threshold()
            tc_eff = self.engine.partial_tile_efficiency(4, motif)
            cycles = (seq_len * max(1, batches) * motif) / (rho * block_size)
            if intensity >= threshold:
                cycles *= 0.82
                rationale.append("compute_bound")
            if "tc" in variant:
                cycles *= max(0.55, 1.0 - 0.5 * tc_eff)
                rationale.append(f"tc_eff={tc_eff:.2f}")
            score = (intensity + tc_eff + pipeline / 8.0) / cycles
            return cycles, score, rationale
        if workload.kind == WorkloadType.HMM_FORWARD:
            n_states = int(workload.params["n_states"])
            seq_len = int(workload.params["sequence_length"])
            batch = int(workload.params["batch_size"])
            intensity = self.engine.hmm_intensity(n_states)
            tc_eff = self.engine.partial_tile_efficiency(n_states, n_states)
            cycles = (seq_len * n_states * n_states * max(batch, 1)) / (rho * block_size)
            if "tc" in variant:
                cycles *= max(0.4, 1.0 - 0.6 * tc_eff)
                rationale.append(f"tc_eff={tc_eff:.2f}")
            if self.gpu.l2_cache_bytes >= 40 * 1024 * 1024:
                cycles *= 0.93
                rationale.append("large_l2")
            score = (intensity + pipeline / 4.0) / cycles
            return cycles, score, rationale
        if workload.kind == WorkloadType.HAMMING:
            seq_len = int(workload.params["sequence_length"])
            lanes = math.ceil(seq_len / 32)
            cycles = lanes * self.engine.hamming_latency()["2bit_cycles"] / (rho * vector_width)
            if variant == "popc_2bit":
                cycles *= 0.9
            score = 1.0 / cycles
            return cycles, score, rationale
        if workload.kind == WorkloadType.SUFFIX_ARRAY:
            text_len = int(workload.params["text_length"])
            passes = 2 * math.ceil(math.log2(max(text_len, 2)))
            cycles = passes * self.engine.digit_sort_latency_full() / rho
            if layout == "padded":
                conflicts = self.engine.bank_conflicts([i * 8 for i in range(32)])
                cycles *= 1.0 - min(0.15, 0.03 * max(0, conflicts["serialization_factor"] - 1))
                rationale.append("bank_conflict_mitigation")
            score = 1.0 / cycles
            return cycles, score, rationale
        raise ValueError(f"Unsupported workload: {workload.kind}")

    def generate_candidates(self, workload: Workload) -> list[CandidateConfig]:
        candidates: list[CandidateConfig] = []
        for variant in self._variants(workload):
            for block_size in self._block_sizes():
                for layout in self._layouts():
                    for pipeline in self._pipelines():
                        for vector_width in (1, 2, 4):
                            regs, smem = self._regs_and_smem(variant, workload, block_size, pipeline)
                            occ = self.engine.occupancy(block_size, regs, smem)
                            if occ["B_res"] <= 0:
                                continue
                            cycles, score, rationale = self._predict(workload, variant, block_size, layout, pipeline, vector_width, occ)
                            candidates.append(CandidateConfig(
                                variant=variant,
                                block_size=block_size,
                                layout=layout,
                                pipeline=pipeline,
                                vector_width=vector_width,
                                regs_per_thread=regs,
                                smem_per_block=smem,
                                occupancy=occ,
                                predicted_cycles=cycles,
                                predicted_score=score,
                                rationale=rationale,
                            ))
        return candidates

    def plan(self, workload: Workload, top_k: int = 5) -> OptimizationPlan:
        candidates = self.generate_candidates(workload)
        ranked = sorted(candidates, key=lambda cfg: cfg.predicted_score, reverse=True)
        top = ranked[:top_k]
        return OptimizationPlan(
            gpu_key=self.gpu_key,
            gpu_name=self.gpu.name,
            workload=workload.to_dict(),
            candidates_considered=len(candidates),
            candidates_feasible=len(candidates),
            best=top[0] if top else None,
            top_candidates=top,
        )
