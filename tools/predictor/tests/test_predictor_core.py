"""Core regression tests for input validation, decoding constraints, and TMM."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PREDICTOR_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PREDICTOR_ROOT.parent
PKG_ROOT = PROJECT_ROOT / "optogpt" / "optogpt"
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(PREDICTOR_ROOT))

from interactive_predictor import (
    ALLOWED_DIELECTRIC,
    BANNED_DIELECTRIC,
    InteractivePredictor,
    WAVELENGTHS_NM,
    _parse_and_interpolate_csv,
    _parse_joint_csv,
    _sample_tokens,
    build_logits_mask,
    generate_candidates,
    select_test_data_dir,
    tmm_simulate,
)
from run_prediction import resolve_csv_path


class DummyGenerator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.proj = nn.Linear(vocab_size, vocab_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(vocab_size))


class DummyModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.generator = DummyGenerator(vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        batch, length = tgt.shape
        vocab = self.generator.proj.out_features
        output = torch.full((batch, length, vocab), -4.0, device=tgt.device)
        output[:, -1, 2] = -10.0  # BOS must never win after the initial token.
        output[:, -1, 3] = 0.2   # EOS remains possible.
        output[:, -1, 4:] = torch.linspace(0.5, 2.0, vocab - 4, device=tgt.device)
        return output


class TestCsvValidation(unittest.TestCase):
    def test_joint_csv_creates_284_dimensional_spectrum(self):
        text = "400,0.1,0.8,0.2,0.7\n1100,0.1,0.8,0.2,0.7"
        spectrum = _parse_joint_csv(text)
        self.assertEqual(spectrum.shape, (284,))

    def test_readme_style_input_path_is_not_double_prefixed(self):
        path = resolve_csv_path("inputs/target_spectrum.csv")
        self.assertEqual(path, PREDICTOR_ROOT / "inputs" / "target_spectrum.csv")

    def test_valid_unsorted_csv_is_sorted_and_interpolated(self):
        lines = [f"{wl},0.2,0.7" for wl in reversed(WAVELENGTHS_NM)]
        spectrum = _parse_and_interpolate_csv("\n".join(lines))
        self.assertEqual(spectrum.shape, (142,))
        self.assertTrue(np.isfinite(spectrum).all())

    def test_duplicate_wavelength_is_rejected(self):
        text = "400,0.1,0.8\n400,0.2,0.7\n1100,0.1,0.8"
        with self.assertRaisesRegex(ValueError, "重复波长"):
            _parse_and_interpolate_csv(text)

    def test_nan_row_is_rejected_instead_of_skipped(self):
        text = "400,0.1,0.8\n500,nan,0.7\n1100,0.1,0.8"
        with self.assertRaisesRegex(ValueError, "NaN 或 Inf"):
            _parse_and_interpolate_csv(text)

    def test_nonphysical_rt_is_rejected(self):
        text = "400,0.7,0.7\n1100,0.7,0.7"
        with self.assertRaisesRegex(ValueError, r"R\+T>1"):
            _parse_and_interpolate_csv(text)

    def test_missing_range_is_rejected(self):
        text = "500,0.1,0.8\n900,0.1,0.8"
        with self.assertRaisesRegex(ValueError, "覆盖 400-1100"):
            _parse_and_interpolate_csv(text)


class TestCandidateGeneration(unittest.TestCase):
    def setUp(self):
        self.word_dict = {
            "UNK": 0, "PAD": 1, "BOS": 2, "EOS": 3,
            "SiO2_100": 4, "TiO2_100": 5, "Ag_100": 6, "MgF2_100": 7,
        }
        self.index_dict = {value: key for key, value in self.word_dict.items()}
        self.model = DummyModel(len(self.word_dict))
        self.spectrum = np.r_[np.zeros(71), np.ones(71)].astype(np.float32)

    def test_sampling_generates_multiple_unique_candidates(self):
        mask = torch.ones(len(self.word_dict), dtype=torch.bool)
        mask[:3] = False
        candidates = generate_candidates(
            self.model, self.spectrum, self.word_dict, self.index_dict,
            num_candidates=16, max_layers=3, top_k=4, top_p=1.0,
            logits_mask=mask, seed=11,
        )
        self.assertGreater(len(candidates), 1)
        self.assertTrue(all(candidate["n_layers"] <= 3 for candidate in candidates))

    def test_dielectric_mask_blocks_banned_materials(self):
        mask = build_logits_mask(
            self.word_dict,
            allowed_materials=ALLOWED_DIELECTRIC,
            banned_materials=BANNED_DIELECTRIC,
        )
        self.assertFalse(bool(mask[self.word_dict["Ag_100"]]))
        self.assertTrue(bool(mask[self.word_dict["SiO2_100"]]))

    def test_negative_top_k_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            _sample_tokens(torch.zeros(1, 8), top_k=-1)

    def test_sampling_parameters_are_checked_for_single_candidate(self):
        with self.assertRaisesRegex(ValueError, "top_p"):
            generate_candidates(
                self.model, self.spectrum, self.word_dict, self.index_dict,
                num_candidates=1, top_p=0,
            )

    def test_direct_nonphysical_spectrum_is_rejected(self):
        predictor = InteractivePredictor()
        spectrum = np.r_[np.full(71, 0.7), np.full(71, 0.7)]
        with self.assertRaisesRegex(ValueError, r"R\+T>1"):
            predictor.predict_and_validate(spectrum)


class TestTmmAndDatasetRouting(unittest.TestCase):
    def test_inc_tmm_result_conserves_energy(self):
        R, T, A = tmm_simulate(["SiO2", "TiO2"], [100, 50], 60, "s")
        self.assertEqual(R.shape, (71,))
        self.assertTrue(np.all(R >= 0))
        self.assertTrue(np.all(T >= 0))
        self.assertTrue(np.all(A >= 0))
        np.testing.assert_allclose(R + T + A, 1.0, atol=1e-8)

    def test_dataset_route_matches_model(self):
        self.assertEqual(
            select_test_data_dir("dielectric_best.pt", 60, True).name, "data"
        )
        self.assertEqual(
            select_test_data_dir("optogpt_60deg_s_best.pt", 60, False).name,
            "data_60deg_s",
        )
        self.assertEqual(
            select_test_data_dir("optogpt.pt", 0, False).name, "data"
        )


if __name__ == "__main__":
    unittest.main()
