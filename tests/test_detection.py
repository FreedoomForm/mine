import unittest

from biocuda.detect import detect_current_gpu


class DetectionTests(unittest.TestCase):
    def test_detect_has_fallback(self):
        result = detect_current_gpu()
        self.assertIn(result.mode, {"simulation", "detected"})
        self.assertTrue(result.gpu.name)
        self.assertTrue(result.matched_key)


if __name__ == "__main__":
    unittest.main()
