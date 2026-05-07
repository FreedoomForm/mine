import unittest

from biocuda.planner import OptimizationPlanner
from biocuda.specs import GPU_DATABASE
from biocuda.workloads import Workload


class PlannerTests(unittest.TestCase):
    def test_sw_plan_prefers_dpx_on_hopper(self):
        planner = OptimizationPlanner("H100_SXM", GPU_DATABASE["H100_SXM"])
        plan = planner.plan(Workload.smith_waterman(4096, 4096), top_k=5)
        self.assertIsNotNone(plan.best)
        self.assertEqual(plan.best.variant, "dpx_native")

    def test_pwm_plan_is_generated(self):
        planner = OptimizationPlanner("A100", GPU_DATABASE["A100"])
        plan = planner.plan(Workload.pwm(sequence_length=100000, motif_length=19, batches=128), top_k=4)
        self.assertGreater(plan.candidates_feasible, 0)
        self.assertIsNotNone(plan.best)


if __name__ == "__main__":
    unittest.main()
