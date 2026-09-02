import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from double_sided.config import DoubleSidedConfig, LayerStage
from double_sided.contract import DoubleSidedStructure, Layer, SIDE_SEP, assign_split
from double_sided.decoder import DecodeState, advance_state, allowed_next_ids, layer_token_ids
from double_sided.model import extend_vocabulary, migrate_joint_sp_model
from double_sided.physics import simulate_abc, simulate_c, spectrum_vector, summarize, verify_merge_equivalence
from double_sided.training import (
    DoubleSidedDataset, build_output_mask, collate_double_sided,
    masked_label_smoothed_loss,
)
from joint_sp.model import ARCH_JOINT_SP_LEGACY_V1, make_model_SP
from optogpt.core.datasets.sim import load_materials


ROOT = Path(__file__).resolve().parents[2]


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.allowed = ["SiO2", "TiO2"]

    def test_strict_contract(self):
        structure = DoubleSidedStructure.from_tokens(
            ["BOS", "SiO2_100", SIDE_SEP, "TiO2_50", "EOS"], self.allowed, 4
        )
        self.assertEqual(structure.physical_layer_counts, (1, 1))
        for invalid in (
            ["BOS", SIDE_SEP, "TiO2_50", "EOS"],
            ["BOS", "SiO2_100", "EOS"],
            ["BOS", "SiO2_100", SIDE_SEP, SIDE_SEP, "TiO2_50", "EOS"],
        ):
            with self.assertRaises(ValueError):
                DoubleSidedStructure.from_tokens(invalid, self.allowed, 4)

    def test_merge_and_mirror_grouping(self):
        original = DoubleSidedStructure(
            (Layer("SiO2", 20), Layer("SiO2", 30)), (Layer("TiO2", 40),)
        )
        merged = DoubleSidedStructure((Layer("SiO2", 50),), (Layer("TiO2", 40),))
        mirror = DoubleSidedStructure((Layer("TiO2", 40),), (Layer("SiO2", 50),))
        self.assertEqual(original.physical_hash(), merged.physical_hash())
        self.assertEqual(original.split_group_hash(), mirror.split_group_hash())
        self.assertEqual(assign_split(original.split_group_hash()), assign_split(mirror.split_group_hash()))

    def test_staged_stop_rule(self):
        config = DoubleSidedConfig(
            layer_stages=[LayerStage(1, 4), LayerStage(5, 8), LayerStage(9, 16)],
            technical_max_layers_per_side=16,
            stage_improvement_threshold=0.01,
        ).validate()
        self.assertFalse(config.should_stop([0.20, 0.195]))
        self.assertTrue(config.should_stop([0.20, 0.195, 0.191]))


class DecoderStateTests(unittest.TestCase):
    def setUp(self):
        self.word = {"UNK": 0, "PAD": 1, "BOS": 2, "EOS": 3, SIDE_SEP: 4,
                     "SiO2_10": 5, "TiO2_20": 6, "Ag_10": 7}
        self.physical = layer_token_ids(self.word, ["SiO2", "TiO2"])

    def test_state_machine_forces_separator_and_eos(self):
        state = DecodeState()
        self.assertNotIn(self.word[SIDE_SEP], allowed_next_ids(state, self.word, self.physical, 1))
        advance_state(state, self.word["SiO2_10"], self.word, set(self.physical))
        self.assertEqual(allowed_next_ids(state, self.word, self.physical, 1), [self.word[SIDE_SEP]])
        advance_state(state, self.word[SIDE_SEP], self.word, set(self.physical))
        advance_state(state, self.word["TiO2_20"], self.word, set(self.physical))
        self.assertEqual(allowed_next_ids(state, self.word, self.physical, 1), [self.word["EOS"]])
        advance_state(state, self.word["EOS"], self.word, set(self.physical))
        self.assertTrue(state.finished)


class ModelMigrationTests(unittest.TestCase):
    def test_named_vocab_rows_are_exactly_inherited(self):
        source_vocab = {"UNK": 0, "PAD": 1, "BOS": 2, "EOS": 3, "SiO2_10": 4}
        target_vocab, _ = extend_vocabulary(source_vocab, ["Sc2O3_10"])
        source = make_model_SP(tgt_vocab=len(source_vocab), N=1, d_model=32, d_ff=16,
                               h=4, dropout=0.0,
                               architecture_version=ARCH_JOINT_SP_LEGACY_V1)
        migrated = migrate_joint_sp_model(source, source_vocab, target_vocab, {
            "N": 1, "d_model": 32, "d_ff": 16, "head_num": 4, "dropout": 0.0,
            "architecture_version": ARCH_JOINT_SP_LEGACY_V1,
        })
        self.assertIn(SIDE_SEP, target_vocab)
        self.assertEqual(migrated.vocabulary_transfer["inherited_token_rows"], len(source_vocab))


class TrainingDataTests(unittest.TestCase):
    def test_collate_keeps_single_bos_eos_and_separator(self):
        word = {"UNK": 0, "PAD": 1, "BOS": 2, "EOS": 3, SIDE_SEP: 4,
                "SiO2_10": 5, "TiO2_20": 6}
        batch = collate_double_sided([
            (np.zeros(284, dtype=np.float32),
             [word["BOS"], word["SiO2_10"], word[SIDE_SEP], word["TiO2_20"], word["EOS"]])
        ], word["PAD"])
        self.assertEqual(batch["decoder_input"].tolist()[0], [2, 5, 4, 6])
        self.assertEqual(batch["target_y"].tolist()[0], [5, 4, 6, 3])
        self.assertEqual(int(batch["ntokens"]), 4)

    def test_output_mask_excludes_legacy_banned_materials(self):
        word = {"UNK": 0, "PAD": 1, "BOS": 2, "EOS": 3, SIDE_SEP: 4}
        for material in ("SiO2", "TiO2"):
            for thickness in range(10, 501, 10):
                word[f"{material}_{thickness}"] = len(word)
        word["Ag_10"] = len(word)
        mask = build_output_mask(word, ["SiO2", "TiO2"])
        self.assertEqual(int(mask.sum()), 102)
        self.assertFalse(bool(mask[word["Ag_10"]]))
        self.assertFalse(bool(mask[word["BOS"]]))
        self.assertTrue(bool(mask[word[SIDE_SEP]]))

    def test_masked_label_smoothing_is_finite(self):
        logits = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.5, 2.0]])
        allowed = torch.tensor([False, True, True])
        logits = logits.masked_fill(~allowed, float("-inf"))
        log_probabilities = torch.log_softmax(logits, dim=-1)
        loss = masked_label_smoothed_loss(
            log_probabilities, torch.tensor([1, 2]), pad_id=0,
            smoothing=0.1, output_mask=allowed,
        )
        self.assertTrue(torch.isfinite(loss))


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DoubleSidedConfig().validate()
        cls.nk = load_materials(
            all_mats=["Glass_Substrate", "SiO2", "TiO2"],
            wavelengths=cls.config.wavelengths_nm / 1000.0,
            DATABASE=str(ROOT / "optogpt" / "nk"),
        )

    def test_abc_shapes_energy_and_merge_equivalence(self):
        structure = DoubleSidedStructure(
            (Layer("SiO2", 40), Layer("SiO2", 60)),
            (Layer("TiO2", 50), Layer("SiO2", 80)),
        )
        labels = simulate_abc(structure, self.nk, self.config)
        c_only = simulate_c(structure, self.nk, self.config)
        for definition in ("A", "B", "C"):
            vector = spectrum_vector(labels[definition])
            self.assertEqual(vector.shape, (284,))
            metrics = summarize(labels[definition])
            self.assertTrue(np.isfinite(metrics["objective"]))
        for pol in ("s", "p"):
            for key in ("R", "T", "A"):
                self.assertTrue(np.array_equal(labels["C"][pol][key], c_only[pol][key]))
        check = verify_merge_equivalence(structure, self.nk, self.config)
        self.assertTrue(check["equivalent"])


if __name__ == "__main__":
    unittest.main()
