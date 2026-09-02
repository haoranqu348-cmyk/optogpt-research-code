"""Regression tests for production-safety fixes in the joint s+p pipeline."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from joint_sp.constants import ALLOWED_MATERIALS, SPEC_DIM, validate_disk_structure_tokens
from joint_sp.decoder import tmm_rerank_joint
from joint_sp.model import (
    ARCH_JOINT_SP_LEGACY_V1,
    load_joint_sp_checkpoint,
    make_model_SP,
    save_sp_checkpoint,
)
from joint_sp.scripts.finetune import validate_data_strict


def _vocab(size=20):
    word_dict = {"UNK": 0, "PAD": 1, "BOS": 2, "EOS": 3}
    for index in range(4, size):
        word_dict[f"SiO2_{index * 10}"] = index
    return word_dict


class TestStrictThicknessTokens(unittest.TestCase):
    def test_rejects_zero_and_leading_zero(self):
        for token in ("SiO2_0", "SiO2_001"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                validate_disk_structure_tokens([token])


class TestConfiguredDropout(unittest.TestCase):
    def test_attention_uses_model_dropout(self):
        model = make_model_SP(tgt_vocab=20, N=1, d_model=32, d_ff=16, h=4, dropout=0.37)
        layer = model.decoder.layers[0]
        self.assertAlmostEqual(layer.self_attn.dropout.p, 0.37)
        self.assertAlmostEqual(layer.src_attn.dropout.p, 0.37)


class TestInferenceLoaderMode(unittest.TestCase):
    def test_loader_returns_eval_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            word_dict = _vocab()
            configs = {
                "architecture_version": ARCH_JOINT_SP_LEGACY_V1,
                "N": 1,
                "d_model": 32,
                "d_ff": 16,
                "head_num": 4,
                "dropout": 0.25,
                "struc_word_dict": word_dict,
                "struc_index_dict": {value: key for key, value in word_dict.items()},
                "pretrained_sha256": "a" * 64,
            }
            model = make_model_SP(tgt_vocab=len(word_dict), N=1, d_model=32,
                                  d_ff=16, h=4, dropout=0.25)
            optimizer = torch.optim.Adam(model.parameters())
            checkpoint = Path(temp_dir) / "model.pt"
            save_sp_checkpoint(model, optimizer, 1, {"train_loss": []}, checkpoint, configs)
            loaded, _word, _index, _config = load_joint_sp_checkpoint(checkpoint)
            self.assertFalse(loaded.training)


class TestDataPublicationGate(unittest.TestCase):
    def _write_minimal_split(self, directory, name):
        with open(directory / f"Structure_{name}.pkl", "wb") as handle:
            pickle.dump([["SiO2_40"]], handle)
        with open(directory / f"Spectrum_{name}.pkl", "wb") as handle:
            pickle.dump(np.zeros((1, SPEC_DIM), dtype=np.float32), handle)

    def test_training_rejects_unpublished_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            self._write_minimal_split(directory, "train")
            self._write_minimal_split(directory, "dev")
            with self.assertRaisesRegex(ValueError, "BUILD_COMPLETE"):
                validate_data_strict(directory, _vocab())

    def test_training_accepts_published_nonleaking_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            structures = {"train": "SiO2_40", "dev": "SiO2_50", "test": "SiO2_60"}
            for name, token in structures.items():
                with open(directory / f"Structure_{name}.pkl", "wb") as handle:
                    pickle.dump([[token]], handle)
                with open(directory / f"Spectrum_{name}.pkl", "wb") as handle:
                    pickle.dump(np.zeros((1, SPEC_DIM), dtype=np.float32), handle)
            (directory / "BUILD_COMPLETE.json").write_text('{"status":"complete"}', encoding="utf-8")
            (directory / "generation_config.json").write_text(json.dumps({
                "spec_layout": ["Rs", "Ts", "Rp", "Tp"],
                "spec_dim": SPEC_DIM,
                "theta_deg": 60,
            }), encoding="utf-8")
            validate_data_strict(directory, _vocab())


class TestRankingObjective(unittest.TestCase):
    @staticmethod
    def _fake_spectrum(materials, thicknesses, **_kwargs):
        transmission = 0.5 if materials[0] == "SiO2" else 0.9
        return [1.0 - transmission] * 71 + [transmission] * 71

    def test_high_transmission_metrics_change_final_order(self):
        target = np.concatenate([
            np.full(71, 0.5), np.full(71, 0.5),
            np.full(71, 0.5), np.full(71, 0.5),
        ])
        candidates = [
            {"tokens": ["SiO2_100"], "materials": ["SiO2"],
             "thicknesses": [100], "n_layers": 1},
            {"tokens": ["TiO2_100"], "materials": ["TiO2"],
             "thicknesses": [100], "n_layers": 1},
        ]
        with patch("optogpt.core.datasets.sim.spectrum", side_effect=self._fake_spectrum):
            error_ranked, _ = tmm_rerank_joint(candidates, target, {}, objective="joint_error")
            high_t_ranked, _ = tmm_rerank_joint(
                candidates, target, {}, objective="high_transmission",
                high_T_objective_weight=2.0,
            )
        self.assertEqual(error_ranked[0]["materials"], ["SiO2"])
        self.assertEqual(high_t_ranked[0]["materials"], ["TiO2"])
        self.assertLess(high_t_ranked[0]["ranking_score"], high_t_ranked[1]["ranking_score"])


if __name__ == "__main__":
    unittest.main()
