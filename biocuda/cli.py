from __future__ import annotations

import argparse
import json
from typing import Any

from .autotune import autotune_plan
from .detect import detect_current_gpu
from .planner import OptimizationPlanner
from .runtime import available_backends
from .specs import GPU_DATABASE
from .workloads import Workload


def _print(data: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=False))
    else:
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{key}: {value}")
        else:
            print(data)


def _planner_from_detection() -> tuple[str, OptimizationPlanner]:
    detection = detect_current_gpu()
    return detection.matched_key, OptimizationPlanner(detection.matched_key, detection.gpu)


def _build_workload(args: argparse.Namespace) -> Workload:
    if args.workload == "smith-waterman":
        return Workload.smith_waterman(args.seq_len_a, args.seq_len_b, affine_gaps=not args.linear_gap)
    if args.workload == "pwm":
        return Workload.pwm(args.sequence_length, args.motif_length, batches=args.batches)
    if args.workload == "hmm-forward":
        return Workload.hmm_forward(args.n_states, args.sequence_length, batch_size=args.batch_size)
    if args.workload == "hamming":
        return Workload.hamming(args.sequence_length)
    if args.workload == "suffix-array":
        return Workload.suffix_array(args.text_length, alphabet_size=args.alphabet_size)
    raise ValueError(f"Unsupported workload {args.workload}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biocuda", description="GPU-aware optimization planner for bioinformatics kernels")
    sub = parser.add_subparsers(dest="command", required=True)

    detect_p = sub.add_parser("detect", help="Detect current GPU")
    detect_p.add_argument("--json", action="store_true")

    list_p = sub.add_parser("list-gpus", help="List built-in GPU database")
    list_p.add_argument("--json", action="store_true")

    back_p = sub.add_parser("backends", help="Show optional runtime backends")
    back_p.add_argument("--json", action="store_true")

    for name in ("plan", "autotune"):
        cmd = sub.add_parser(name, help=f"{name.title()} a workload")
        cmd.add_argument("workload", choices=["smith-waterman", "pwm", "hmm-forward", "hamming", "suffix-array"])
        cmd.add_argument("--json", action="store_true")
        cmd.add_argument("--top-k", type=int, default=5)
        cmd.add_argument("--seq-len-a", type=int, default=2048)
        cmd.add_argument("--seq-len-b", type=int, default=2048)
        cmd.add_argument("--linear-gap", action="store_true")
        cmd.add_argument("--sequence-length", type=int, default=8192)
        cmd.add_argument("--motif-length", type=int, default=19)
        cmd.add_argument("--batches", type=int, default=1)
        cmd.add_argument("--n-states", type=int, default=128)
        cmd.add_argument("--batch-size", type=int, default=1)
        cmd.add_argument("--text-length", type=int, default=1_000_000)
        cmd.add_argument("--alphabet-size", type=int, default=4)
        if name == "autotune":
            cmd.add_argument("--rounds", type=int, default=50)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "detect":
        detection = detect_current_gpu()
        _print(detection.to_dict(), args.json)
        return

    if args.command == "list-gpus":
        payload = {key: spec.to_dict() for key, spec in GPU_DATABASE.items()}
        _print(payload, args.json)
        return

    if args.command == "backends":
        _print(available_backends().to_dict(), args.json)
        return

    _, planner = _planner_from_detection()
    workload = _build_workload(args)
    plan = planner.plan(workload, top_k=args.top_k)

    if args.command == "plan":
        _print(plan.to_dict(), args.json)
        return

    if args.command == "autotune":
        report = autotune_plan(plan, planner.gpu, rounds=args.rounds)
        payload = {"plan": plan.to_dict(), "autotune": report.to_dict()}
        _print(payload, args.json)
        return


if __name__ == "__main__":
    main()
