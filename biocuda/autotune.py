from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional

from .formulas import FormulaEngine
from .planner import CandidateConfig, OptimizationPlan
from .specs import GPUSpec


@dataclass
class EXP3Report:
    best: CandidateConfig
    final_weights: list[float]
    selections: list[int]
    eta: float
    regret_bound: float
    rounds: int

    def to_dict(self) -> dict:
        return {
            "best": self.best.to_dict(),
            "final_weights": self.final_weights,
            "selections": self.selections,
            "eta": self.eta,
            "regret_bound": self.regret_bound,
            "rounds": self.rounds,
        }


class EXP3Tuner:
    def __init__(self, gpu: GPUSpec, seed: int = 7):
        self.engine = FormulaEngine(gpu)
        self.random = random.Random(seed)

    def tune(
        self,
        candidates: List[CandidateConfig],
        rounds: int = 50,
        measure: Optional[Callable[[CandidateConfig], float]] = None,
    ) -> EXP3Report:
        if not candidates:
            raise ValueError("No candidate configurations to tune")
        k = len(candidates)
        weights = [1.0 / k] * k
        selections = [0] * k
        eta = self.engine.optimal_eta(k, rounds)

        for _ in range(rounds):
            choice = self._sample(weights)
            selections[choice] += 1
            observed_cost = measure(candidates[choice]) if measure else self._synthetic_cost(candidates[choice])
            observed_cost = max(observed_cost, 1e-12)
            estimated_loss = min(1.0, observed_cost / max(candidate.predicted_cycles for candidate in candidates))
            losses = [0.0] * k
            losses[choice] = estimated_loss / max(weights[choice], 1e-12)
            weights = self.engine.exp3_update(weights, losses, eta)

        best_index = max(range(k), key=lambda idx: weights[idx])
        return EXP3Report(
            best=candidates[best_index],
            final_weights=weights,
            selections=selections,
            eta=eta,
            regret_bound=self.engine.regret_bound(k, rounds),
            rounds=rounds,
        )

    def _sample(self, weights: list[float]) -> int:
        threshold = self.random.random()
        prefix = 0.0
        for index, weight in enumerate(weights):
            prefix += weight
            if threshold <= prefix:
                return index
        return len(weights) - 1

    def _synthetic_cost(self, candidate: CandidateConfig) -> float:
        noise = self.random.uniform(0.97, 1.03)
        return candidate.predicted_cycles * noise


def autotune_plan(
    plan: OptimizationPlan,
    gpu: GPUSpec,
    rounds: int = 50,
    top_k: Optional[int] = None,
    measure: Optional[Callable[[CandidateConfig], float]] = None,
) -> EXP3Report:
    candidates = plan.top_candidates if top_k is None else plan.top_candidates[:top_k]
    tuner = EXP3Tuner(gpu)
    return tuner.tune(candidates, rounds=rounds, measure=measure)
