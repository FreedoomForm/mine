import unittest

from biocuda.formulas import FormulaEngine
from biocuda.specs import GPU_DATABASE


class FormulaTests(unittest.TestCase):
    def setUp(self):
        self.engine = FormulaEngine(GPU_DATABASE["H100_SXM"])

    def test_complement_involution(self):
        values = list(range(32))
        self.assertTrue(self.engine.verify_involution(values, 31))

    def test_partial_tile_efficiency(self):
        self.assertAlmostEqual(self.engine.partial_tile_efficiency(16, 16), 1.0)
        self.assertAlmostEqual(self.engine.partial_tile_efficiency(8, 16), 0.5)

    def test_wavefront_latency_uses_dpx(self):
        self.assertEqual(self.engine.wavefront_latency_lb(), 10)

    def test_occupancy_positive(self):
        occ = self.engine.occupancy(256, 32, 8192)
        self.assertGreater(occ["B_res"], 0)
        self.assertLessEqual(occ["rho_warp"], 1.0)


if __name__ == "__main__":
    unittest.main()
