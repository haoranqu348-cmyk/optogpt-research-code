"""
Checkpoint Optimization for OptoGPT.

Problem: current checkpoints are ~692 MB each because they include
Adam optimizer states (2x model params). This script:

1. Loads existing checkpoints
2. Creates lightweight 'best' versions (inference-only: model weights + vocab + configs)
3. Creates compact 'resume' versions (model + optimizer + epoch + loss history)
4. Validates that lightweight checkpoints produce identical outputs
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.models.transformer import make_model_I
from multi_candidate_decoder import batch_greedy_decode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_full_checkpoint(path):
    """Load a full checkpoint with all metadata."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def create_lightweight_best(full_ckpt, output_path, pretrained_ckpt=None):
    """
    Create inference-only checkpoint.
    Contains: model_state_dict, model hyperparams, vocab, physical conditions.
    """
    configs = full_ckpt.get("configs", {})

    # Resolve vocab - needed for standalone loading
    word_dict = None
    index_dict = None
    if isinstance(configs, dict):
        word_dict = configs.get("struc_word_dict")
        index_dict = configs.get("struc_index_dict")
    else:
        word_dict = getattr(configs, "struc_word_dict", None)
        index_dict = getattr(configs, "struc_index_dict", None)

    # Fall back to pretrained checkpoint for vocab
    if word_dict is None and pretrained_ckpt is not None:
        p_cfg = pretrained_ckpt.get("configs", {})
        if isinstance(p_cfg, dict):
            word_dict = p_cfg.get("struc_word_dict", {})
            index_dict = p_cfg.get("struc_index_dict", {})
        else:
            word_dict = getattr(p_cfg, "struc_word_dict", {})
            index_dict = getattr(p_cfg, "struc_index_dict", {})

    # Extract hyperparams
    def _get(cfg, key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    p_cfg = pretrained_ckpt.get("configs", {}) if pretrained_ckpt else {}

    light_config = {
        "description": "Lightweight inference-only checkpoint for OptoGPT 60deg s-pol",
        "spec_dim": _get(configs, "spec_dim", _get(p_cfg, "spec_dim", 142)),
        "struc_dim": _get(configs, "struc_dim", _get(p_cfg, "struc_dim", 904)),
        "layers": _get(configs, "layers", _get(p_cfg, "layers", 6)),
        "d_model": _get(configs, "d_model", _get(p_cfg, "d_model", 1024)),
        "d_ff": _get(configs, "d_ff", _get(p_cfg, "d_ff", 512)),
        "head_num": _get(configs, "head_num", _get(p_cfg, "head_num", 8)),
        "dropout": _get(configs, "dropout", _get(p_cfg, "dropout", 0.1)),
        "max_len": _get(configs, "max_len", _get(p_cfg, "max_len", 22)),
        "struc_word_dict": word_dict,
        "struc_index_dict": index_dict,
        "theta_deg": _get(configs, "theta_deg", 60),
        "polarization": _get(configs, "polarization", "s"),
    }

    # Remove optimizer state to save space
    light_ckpt = {
        "model_state_dict": full_ckpt["model_state_dict"],
        "configs": light_config,
        "original_source": str(output_path.parent / output_path.name.replace("_light.pt", ".pt")),
    }

    torch.save(light_ckpt, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Lightweight checkpoint saved: {output_path} ({size_mb:.1f} MB)")
    return light_ckpt


def create_resume_checkpoint(full_ckpt, output_path):
    """
    Create compact resume checkpoint.
    Contains: model_state_dict, optimizer_state_dict, epoch, loss_history, configs.
    """
    light_ckpt = {
        "model_state_dict": full_ckpt["model_state_dict"],
        "optimizer_state_dict": full_ckpt.get("optimizer_state_dict"),
        "epoch": full_ckpt.get("epoch"),
        "loss_all": full_ckpt.get("loss_all"),
        "configs": full_ckpt.get("configs"),
    }

    torch.save(light_ckpt, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Resume checkpoint saved: {output_path} ({size_mb:.1f} MB)")
    return light_ckpt


def verify_output_equivalence(original_path, new_path, pretrained_path, n_test=5):
    """Verify that old and new checkpoints produce identical outputs."""
    # Load original
    orig_ckpt = load_full_checkpoint(original_path)
    new_ckpt = load_full_checkpoint(new_path)

    # Load pretrained for vocab
    pretrained_ckpt = load_full_checkpoint(pretrained_path) if pretrained_path else None

    def build_model_from_ckpt(ckpt, pt_ckpt):
        cfg = ckpt.get("configs", {})
        p_cfg = pt_ckpt.get("configs", {}) if pt_ckpt else {}

        def _get(key, default):
            val = cfg.get(key) if isinstance(cfg, dict) else getattr(cfg, key, None)
            if val is None and pt_ckpt:
                val = p_cfg.get(key) if isinstance(p_cfg, dict) else getattr(p_cfg, key, None)
            return val if val is not None else default

        word_dict = _get("struc_word_dict", {})
        model = make_model_I(
            src_vocab=_get("spec_dim", 142),
            tgt_vocab=_get("struc_dim", len(word_dict)),
            N=_get("layers", 6),
            d_model=_get("d_model", 1024),
            d_ff=_get("d_ff", 512),
            h=_get("head_num", 8),
            dropout=_get("dropout", 0.1),
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        return model, word_dict, _get("struc_index_dict", {})

    model_orig, wd, idict = build_model_from_ckpt(orig_ckpt, pretrained_ckpt)
    model_new, _, _ = build_model_from_ckpt(new_ckpt, pretrained_ckpt)

    # Generate random test spectra
    rng = np.random.RandomState(42)
    test_specs = [rng.rand(142).tolist() for _ in range(n_test)]

    for i, spec in enumerate(test_specs):
        tokens_orig = batch_greedy_decode(
            model_orig, [spec], wd, idict, max_len=22, device=DEVICE)[0]
        tokens_new = batch_greedy_decode(
            model_new, [spec], wd, idict, max_len=22, device=DEVICE)[0]
        match = tokens_orig == tokens_new
        if match:
            print(f"  Test {i+1}: MATCH")
        else:
            print(f"  Test {i+1}: MISMATCH!")
            print(f"    Orig: {tokens_orig}")
            print(f"    New:  {tokens_new}")
            return False

    print("  All tests passed - outputs are identical!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Optimize OptoGPT checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint to optimize")
    parser.add_argument("--pretrained", type=str, default="../model/optogpt.pt",
                        help="Path to pretrained checkpoint (for vocab fallback)")
    parser.add_argument("--output_dir", type=str, default="../model",
                        help="Output directory for optimized checkpoints")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Verify output equivalence after optimization")
    args = parser.parse_args()

    ckpt_path = (PROJECT_ROOT / args.checkpoint).resolve()
    pretrained_path = (PROJECT_ROOT / args.pretrained).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Optimizing: {ckpt_path}")
    full_ckpt = load_full_checkpoint(str(ckpt_path))

    original_size = os.path.getsize(str(ckpt_path)) / 1024 / 1024
    print(f"  Original size: {original_size:.1f} MB")
    print(f"  Has optimizer: {'optimizer_state_dict' in full_ckpt}")
    print(f"  Epoch: {full_ckpt.get('epoch', 'N/A')}")

    # Load pretrained for vocab fallback
    pretrained_ckpt = None
    if pretrained_path.exists():
        pretrained_ckpt = load_full_checkpoint(str(pretrained_path))

    stem = ckpt_path.stem

    # Create lightweight best
    light_path = output_dir / f"{stem}_light.pt"
    # Only create if it doesn't already exist
    if not light_path.exists():
        print("\nCreating lightweight inference checkpoint...")
        create_lightweight_best(full_ckpt, str(light_path), pretrained_ckpt)
    else:
        print(f"\nLightweight checkpoint already exists: {light_path}")

    # Verify
    if args.verify and light_path.exists():
        print("\nVerifying output equivalence...")
        verify_output_equivalence(
            str(ckpt_path), str(light_path),
            str(pretrained_path) if pretrained_path.exists() else None,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
