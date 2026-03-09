import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest

from ai_labeller.features.golden import normalize_golden_mode


class GoldenModeTests(unittest.TestCase):
    def test_normalize_legacy_values(self):
        self.assertEqual(normalize_golden_mode("class"), "class")
        self.assertEqual(normalize_golden_mode("position"), "position")
        self.assertEqual(normalize_golden_mode("both"), "both")

    def test_normalize_professional_labels(self):
        self.assertEqual(normalize_golden_mode("Label Count Match"), "class")
        self.assertEqual(normalize_golden_mode("Spatial Match"), "position")
        self.assertEqual(normalize_golden_mode("Strict Match"), "both")


if __name__ == "__main__":
    unittest.main()
