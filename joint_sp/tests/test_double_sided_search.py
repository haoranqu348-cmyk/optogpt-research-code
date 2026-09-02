import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from joint_sp.scripts.search_double_sided_glass import (  # noqa: E402
    build_stack,
    merge_adjacent,
    simulate_inc,
    simulate_search_fast,
)


class DoubleSidedSearchPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.wavelengths = np.asarray([400.0, 700.0, 1100.0])
        self.nk = {
            "Glass_Substrate": np.full(3, 1.50 + 0j),
            "MgF2": np.full(3, 1.38 + 0j),
            "Al2O3": np.full(3, 1.72 + 0j),
            "SiO2": np.full(3, 1.46 + 0j),
        }

    def test_independent_back_order_is_glass_to_air(self):
        n_list, d_list, c_list = build_stack(
            [(1.38, 100.0), (1.72, 80.0)], [(1.46, 90.0), (1.38, 120.0)], 1.5
        )
        self.assertEqual(n_list, [1.0, 1.38, 1.72, 1.5, 1.46, 1.38, 1.0])
        self.assertEqual(d_list[1:-1], [100.0, 80.0, 500000, 90.0, 120.0])
        self.assertEqual(c_list, ["i", "c", "c", "i", "c", "c", "i"])

    def test_fast_incoherent_series_matches_inc_tmm(self):
        front = [("MgF2", 137.2), ("Al2O3", 91.4)]
        back = [("SiO2", 126.8), ("MgF2", 104.2)]
        exact = simulate_inc(front, back, 60.0, self.wavelengths, self.nk)
        fast = simulate_search_fast(front, back, 60.0, self.wavelengths, self.nk)
        for pol in ("s", "p"):
            for key in ("R", "T", "A"):
                np.testing.assert_allclose(exact[pol][key], fast[pol][key], atol=2e-12)

    def test_adjacent_equal_layers_merge_exactly(self):
        original = [("MgF2", 100), ("MgF2", 40), ("Al2O3", 80)]
        merged = merge_adjacent(original)
        self.assertEqual(merged, [("MgF2", 140.0), ("Al2O3", 80.0)])
        before = simulate_inc(original, [], 60.0, self.wavelengths, self.nk)
        after = simulate_inc(merged, [], 60.0, self.wavelengths, self.nk)
        for pol in ("s", "p"):
            np.testing.assert_allclose(before[pol]["R"], after[pol]["R"], atol=2e-12)


if __name__ == "__main__":
    unittest.main()
