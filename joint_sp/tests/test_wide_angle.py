"""Regression tests for wide-angle transmission metrics."""

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from joint_sp.scripts.evaluate_wide_angle import (
    _candidate_from_mapping,
    deduplicate_candidates,
    load_structure_candidates,
    parse_angle_grid,
    summarize_grid,
)


class TestAngleGrid(unittest.TestCase):
    def test_range_includes_requested_stop(self):
        np.testing.assert_allclose(parse_angle_grid("0:80:30"), [0, 30, 60, 80])

    def test_rejects_grazing_or_invalid_range(self):
        for value in ("0:90:5", "80:0:5", "0:80:0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_angle_grid(value)

    def test_structure_input_rejects_fractional_token_thickness(self):
        with self.assertRaisesRegex(ValueError, "positive integers"):
            _candidate_from_mapping({"materials": ["SiO2"], "thicknesses": [100.5]})

    def test_multiple_files_can_be_merged_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.json"
            second = Path(temp_dir) / "second.json"
            row = {"materials": ["SiO2"], "thicknesses": [100]}
            first.write_text(json.dumps({"top_results": [row]}), encoding="utf-8")
            second.write_text(json.dumps({"structures": [row]}), encoding="utf-8")
            merged = deduplicate_candidates(
                load_structure_candidates(first) + load_structure_candidates(second)
            )
            self.assertEqual(len(merged), 1)


class TestWideAngleMetrics(unittest.TestCase):
    def setUp(self):
        self.angles = np.asarray([0.0, 40.0, 80.0])
        self.thresholds = {
            "mean_threshold": 0.85,
            "p05_threshold": 0.80,
            "min_threshold": 0.70,
        }

    def test_uniform_high_transmission_passes(self):
        metrics = summarize_grid(
            {"Ts": np.full((3, 71), 0.92), "Tp": np.full((3, 71), 0.95)},
            self.angles,
            **self.thresholds,
        )
        self.assertTrue(metrics["all_angles_pass"])
        self.assertEqual(metrics["angle_pass_rate"], 1.0)
        self.assertAlmostEqual(metrics["worst_pol_angle_mean_T"], 0.92)

    def test_bad_high_angle_tail_fails_despite_good_global_mean(self):
        ts = np.full((3, 71), 0.95)
        tp = np.full((3, 71), 0.96)
        ts[-1, :8] = 0.50
        metrics = summarize_grid({"Ts": ts, "Tp": tp}, self.angles, **self.thresholds)
        self.assertGreater(metrics["mean_Ts"], 0.90)
        self.assertFalse(metrics["all_angles_pass"])
        self.assertEqual(metrics["worst_p05_angle_deg"], 80.0)
        self.assertFalse(metrics["per_angle"][-1]["passes"])

    def test_worst_polarization_is_not_hidden_by_unpolarized_average(self):
        metrics = summarize_grid(
            {"Ts": np.full((3, 71), 0.74), "Tp": np.full((3, 71), 0.99)},
            self.angles,
            **self.thresholds,
        )
        self.assertFalse(metrics["all_angles_pass"])
        self.assertAlmostEqual(metrics["worst_pol_angle_mean_T"], 0.74)


if __name__ == "__main__":
    unittest.main()
