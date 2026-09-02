"""Tests for per-angle error exports used by wide-angle plots."""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from joint_sp.scripts.plot_wide_angle_errors import (
    candidate_summary,
    compute_error_rows,
    load_attempt_candidates,
)


class TestErrorRows(unittest.TestCase):
    def test_attempt_loader_retains_duplicate_seed_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "seed_42.json"
            row = {"materials": ["SiO2"], "thicknesses": [100]}
            path.write_text(json.dumps({"top_results": [row, row]}), encoding="utf-8")
            candidates = load_attempt_candidates([path])
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["candidate_id"], "S42-R01")
        self.assertEqual(candidates[1]["candidate_id"], "S42-R02")
        self.assertEqual(candidates[0]["structure_hash"], candidates[1]["structure_hash"])

    def test_errors_are_relative_to_ideal_transmission(self):
        candidate = {
            "tokens": ["SiO2_100"],
            "materials": ["SiO2"],
            "thicknesses": [100],
            "n_layers": 1,
            "structure_hash": "abc",
        }
        matrices = {
            "Ts": np.asarray([[0.8, 0.6], [0.4, 0.2]]),
            "Tp": np.asarray([[1.0, 0.8], [0.9, 0.7]]),
        }
        with patch(
            "joint_sp.scripts.plot_wide_angle_errors.simulate_angle_grid",
            return_value=matrices,
        ):
            rows, _spectra, failures = compute_error_rows(
                [candidate], np.asarray([0.0, 80.0]), {}
            )
        self.assertFalse(failures)
        self.assertAlmostEqual(rows[0]["E_s"], 0.3)
        self.assertAlmostEqual(rows[0]["E_p"], 0.1)
        self.assertAlmostEqual(rows[0]["E_joint"], 0.2)
        self.assertAlmostEqual(rows[1]["E_s"], 0.7)
        self.assertAlmostEqual(rows[1]["E_p"], 0.2)
        self.assertAlmostEqual(rows[1]["E_joint"], 0.45)

    def test_candidate_summary_uses_worst_angle(self):
        rows = [
            {
                "candidate_id": "C01", "structure_hash": "abc", "tokens": "SiO2_100",
                "n_layers": 1, "angle_deg": 0.0, "E_s": 0.1, "E_p": 0.1,
                "E_joint": 0.1,
            },
            {
                "candidate_id": "C01", "structure_hash": "abc", "tokens": "SiO2_100",
                "n_layers": 1, "angle_deg": 80.0, "E_s": 0.7, "E_p": 0.3,
                "E_joint": 0.5,
            },
        ]
        summary = candidate_summary(rows)
        self.assertEqual(summary[0]["worst_angle_deg"], 80.0)
        self.assertAlmostEqual(summary[0]["worst_E_joint"], 0.5)


if __name__ == "__main__":
    unittest.main()
