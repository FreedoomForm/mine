from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class WorkloadType(str, Enum):
    SMITH_WATERMAN = "smith-waterman"
    PWM = "pwm"
    HMM_FORWARD = "hmm-forward"
    HAMMING = "hamming"
    SUFFIX_ARRAY = "suffix-array"


@dataclass(frozen=True)
class Workload:
    kind: WorkloadType
    params: Dict[str, Any]

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "params": dict(self.params)}

    @staticmethod
    def smith_waterman(seq_len_a: int, seq_len_b: int, affine_gaps: bool = True) -> "Workload":
        return Workload(WorkloadType.SMITH_WATERMAN, {
            "seq_len_a": seq_len_a,
            "seq_len_b": seq_len_b,
            "affine_gaps": affine_gaps,
        })

    @staticmethod
    def pwm(sequence_length: int, motif_length: int, batches: int = 1) -> "Workload":
        return Workload(WorkloadType.PWM, {
            "sequence_length": sequence_length,
            "motif_length": motif_length,
            "batches": batches,
        })

    @staticmethod
    def hmm_forward(n_states: int, sequence_length: int, batch_size: int = 1) -> "Workload":
        return Workload(WorkloadType.HMM_FORWARD, {
            "n_states": n_states,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
        })

    @staticmethod
    def hamming(sequence_length: int) -> "Workload":
        return Workload(WorkloadType.HAMMING, {"sequence_length": sequence_length})

    @staticmethod
    def suffix_array(text_length: int, alphabet_size: int = 4) -> "Workload":
        return Workload(WorkloadType.SUFFIX_ARRAY, {
            "text_length": text_length,
            "alphabet_size": alphabet_size,
        })
