import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from double_sided.contract import DoubleSidedStructure, Layer, assign_split
from double_sided.scripts.compare_model_evaluations import selection_key, summarize_evaluation
from double_sided.scripts.generate_formal_data import load_elite_templates, perturb_elite
from double_sided.scripts.merge_formal_datasets import merge_datasets
from double_sided.scripts.predict_band import (
    build_target_spectrum, evaluate_structures, parse_out_of_band_reflectances,
)
from double_sided.config import DoubleSidedConfig
from optogpt.core.datasets.sim import load_materials


def write_dataset(root, rows, value_offset=0.0):
    root.mkdir(parents=True)
    by_split = {split: [] for split in ("train", "dev", "test")}
    for row in rows:
        by_split[row["split"]].append(row)
    for split, split_rows in by_split.items():
        with (root / f"structures_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row) + "\n")
        values = np.asarray([
            np.full(284, index + value_offset, dtype=np.float32)
            for index, _ in enumerate(split_rows)
        ]).reshape((-1, 284))
        np.savez_compressed(root / f"spectra_ABC_{split}.npz", A=values, B=values, C=values)


def record(structure, family="random"):
    merged = structure.merged()
    split = assign_split(structure.split_group_hash())
    return {
        "tokens": structure.to_tokens(),
        "merged_tokens": merged.to_tokens(),
        "front_layers_raw": len(structure.front),
        "back_layers_raw": len(structure.back),
        "front_layers_physical": len(merged.front),
        "back_layers_physical": len(merged.back),
        "physical_hash": structure.physical_hash(),
        "split_group_hash": structure.split_group_hash(),
        "source_family": family,
        "split": split,
    }


class ProgressiveGenerationTests(unittest.TestCase):
    def test_short_elite_is_extended_to_requested_physical_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            elite = Path(temporary) / "rankings.csv"
            with elite.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "objective", "front_tokens", "back_tokens",
                ])
                writer.writeheader()
                writer.writerow({
                    "objective": 0.1,
                    "front_tokens": "MgF2_100/SiO2_100/Al2O3_100",
                    "back_tokens": "Al2O3_100/SiO2_100/MgF2_100",
                })
            template = load_elite_templates(elite, maximum_layers=16)[0]
            for seed in range(10):
                structure = perturb_elite(
                    np.random.RandomState(seed), template,
                    minimum_layers=9, maximum_layers=16,
                )
                front, back = structure.physical_layer_counts
                self.assertTrue(9 <= front <= 16)
                self.assertTrue(9 <= back <= 16)


class DatasetMergeTests(unittest.TestCase):
    def test_base_duplicate_is_retained_and_spectra_are_verified(self):
        structure = DoubleSidedStructure(
            (Layer("SiO2", 100),), (Layer("TiO2", 100),)
        )
        extra_structure = DoubleSidedStructure(
            (Layer("MgF2", 100),), (Layer("Al2O3", 100),)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, extra, output = root / "base", root / "extra", root / "merged"
            write_dataset(base, [record(structure, "base")])
            write_dataset(extra, [record(structure, "duplicate"), record(extra_structure, "extra")])
            contract = merge_datasets(base, [extra], output)
            self.assertEqual(contract["duplicates_removed"], 1)
            self.assertEqual(contract["unique_physical_samples"], 2)
            self.assertEqual(contract["maximum_duplicate_spectrum_difference"], 0.0)
            split = assign_split(structure.split_group_hash())
            rows = [json.loads(line) for line in
                    (output / f"structures_{split}.jsonl").read_text().splitlines()]
            retained = next(row for row in rows if row["physical_hash"] == structure.physical_hash())
            self.assertEqual(retained["source_family"], "base")

    def test_inconsistent_duplicate_spectrum_is_rejected(self):
        structure = DoubleSidedStructure(
            (Layer("SiO2", 100),), (Layer("TiO2", 100),)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, extra = root / "base", root / "extra"
            write_dataset(base, [record(structure)], value_offset=0.0)
            write_dataset(extra, [record(structure)], value_offset=0.1)
            with self.assertRaises(RuntimeError):
                merge_datasets(base, [extra], root / "merged")


class ComparisonTests(unittest.TestCase):
    def test_robust_selection_precedes_nominal_objective(self):
        max8 = {
            "worst_envelope_passes_strict": False,
            "best_passes_strict": False,
            "worst_envelope_objective": 0.12,
            "best_objective": 0.11,
        }
        max16 = {
            "worst_envelope_passes_strict": False,
            "best_passes_strict": False,
            "worst_envelope_objective": 0.13,
            "best_objective": 0.10,
        }
        self.assertLess(selection_key(max8), selection_key(max16))

    def test_evaluation_and_robustness_files_are_combined(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation, robustness = root / "evaluation", root / "robustness"
            evaluation.mkdir(); robustness.mkdir()
            ranking = {
                "method": "model_tmm_topk", "objective": 0.11, "passes_strict": False,
                "front_physical_layers": 9, "back_physical_layers": 8,
                "front_tokens": "SiO2_100", "back_tokens": "TiO2_100",
                "mean_Rs": 0.18, "mean_Rp": 0.01, "p95_Rs": 0.19,
                "p95_Rp": 0.02, "mean_Ts": 0.82, "mean_Tp": 0.99,
            }
            with (evaluation / "rankings.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(ranking))
                writer.writeheader(); writer.writerow(ranking)
            (evaluation / "manifest.json").write_text(json.dumps({
                "requested_candidates": 256, "unique_physical_structures": 200,
                "total_tmm_calls": 12345,
            }))
            robust = {
                "rank": 1, "worst_envelope_objective": 0.12,
                "worst_envelope_passes_strict": False,
                "worst_envelope_mean_Rs": 0.2, "worst_envelope_mean_Rp": 0.02,
            }
            with (robustness / "robustness_top20_summary.csv").open(
                    "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(robust))
                writer.writeheader(); writer.writerow(robust)
            summary = summarize_evaluation("max16", evaluation, robustness)
            self.assertEqual(summary["total_tmm_calls"], 12345)
            self.assertEqual(summary["generated_structures_using_layers_9_to_16"], 1)
            self.assertAlmostEqual(summary["worst_envelope_objective"], 0.12)


class BandPredictionTests(unittest.TestCase):
    def test_target_keeps_284_value_checkpoint_contract(self):
        wavelengths = np.arange(400.0, 1101.0, 10.0)
        target = build_target_spectrum(wavelengths, 400.0, 800.0, 0.25)
        self.assertEqual(target.shape, (284,))
        n = len(wavelengths)
        in_band = wavelengths <= 800.0
        self.assertTrue(np.all(target[:n][in_band] == 0.0))
        self.assertTrue(np.all(target[n:2*n][in_band] == 1.0))
        self.assertTrue(np.all(target[:n][~in_band] == 0.25))
        self.assertTrue(np.all(target[n:2*n][~in_band] == 0.75))

    def test_out_of_band_profiles_are_validated(self):
        self.assertEqual(parse_out_of_band_reflectances("0, 0.1, 0.1"), [0.0, 0.1])
        with self.assertRaises(ValueError):
            parse_out_of_band_reflectances("1.1")

    def test_band_tmm_evaluation_uses_only_selected_points(self):
        root = Path(__file__).resolve().parents[2]
        config = DoubleSidedConfig(
            wavelengths_nm=np.arange(400.0, 801.0, 10.0)
        )
        materials = [config.substrate, "SiO2", "TiO2"]
        nk = load_materials(
            all_mats=materials,
            wavelengths=config.wavelengths_nm / 1000.0,
            DATABASE=str(root / "optogpt" / "nk"),
        )
        structure = DoubleSidedStructure(
            (Layer("SiO2", 100),), (Layer("TiO2", 100),)
        )
        row = evaluate_structures([structure], nk, config)[0]
        self.assertEqual(row["tmm_calls"], 82)
        self.assertTrue(np.isfinite(row["metrics"]["objective"]))


if __name__ == "__main__":
    unittest.main()
