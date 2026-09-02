"""Regression tests for OptoGPT checkpoint forward semantics."""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = ROOT / "optogpt" / "optogpt"
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The architecture loader does not use the simulator; avoid making tmm a test dependency.
fake_sim = types.ModuleType("core.datasets.sim")
fake_sim.load_materials = None
fake_sim.wavelengths = None
sys.modules.setdefault("core.datasets.sim", fake_sim)

from core.models.transformer import make_model_I
from interactive_predictor import (
    ARCH_LEGACY,
    ARCH_RELU,
    ORIGINAL_OPTOGPT_SHA256,
    _apply_model_architecture,
    _checkpoint_model_type,
    _resolve_architecture,
)


class TestArchitectureResolution(unittest.TestCase):
    def test_joint_checkpoint_is_detected_from_state_keys(self):
        state = {
            "fc_s.fc1.weight": torch.zeros(1),
            "fc_p.fc1.weight": torch.zeros(1),
            "fusion.0.weight": torch.zeros(1),
        }
        self.assertEqual(_checkpoint_model_type({}, state), "joint_sp")

    def test_original_hash_selects_legacy(self):
        with patch(
            "interactive_predictor._sha256_file",
            return_value=ORIGINAL_OPTOGPT_SHA256,
        ):
            resolved = _resolve_architecture("optogpt.pt", {}, "auto")
        self.assertEqual(resolved, ARCH_LEGACY)

    def test_saved_relu_version_is_respected(self):
        resolved = _resolve_architecture(
            "fine_tuned.pt", {"architecture_version": "joint_sp_relu_v0"}, "auto"
        )
        self.assertEqual(resolved, ARCH_RELU)

    def test_explicit_override_wins(self):
        resolved = _resolve_architecture(
            "unknown.pt", {"architecture_version": "joint_sp_relu_v0"}, ARCH_LEGACY
        )
        self.assertEqual(resolved, ARCH_LEGACY)


class TestLegacyForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = make_model_I(
            src_vocab=4, tgt_vocab=6, N=1, d_model=4, d_ff=3, h=1, dropout=0.0
        ).eval()

    def test_legacy_fc_uses_historical_formula(self):
        _apply_model_architecture(self.model, ARCH_LEGACY)
        value = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
        expected = self.model.fc.fc2(self.model.fc.norm(self.model.fc.fc1(value)))
        torch.testing.assert_close(self.model.fc(value), expected)

    def test_legacy_ff_has_no_relu(self):
        _apply_model_architecture(self.model, ARCH_LEGACY)
        ff = self.model.decoder.layers[0].feed_forward
        value = torch.tensor([[[-1.0, 0.5, 2.0, -0.25]]])
        expected = ff.w_2(ff.dropout(ff.w_1(value)))
        torch.testing.assert_close(ff(value), expected)


if __name__ == "__main__":
    unittest.main()
