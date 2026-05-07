from biocuda.detect import detect_current_gpu
from biocuda.planner import OptimizationPlanner
from biocuda.workloads import Workload


def main() -> None:
    detection = detect_current_gpu()
    planner = OptimizationPlanner(detection.matched_key, detection.gpu)
    workload = Workload.smith_waterman(seq_len_a=4096, seq_len_b=4096)
    plan = planner.plan(workload, top_k=3)
    print(plan.to_dict())


if __name__ == "__main__":
    main()
