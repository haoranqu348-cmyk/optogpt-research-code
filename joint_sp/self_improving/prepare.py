"""
joint_sp/self_improving/prepare.py — Generate candidate structures for OOD targets.

Uses the joint TransformerSP model to decode structures for OOD targets,
then re-ranks via TMM. Outputs structure-spectrum pairs for perturbation.

Usage (standalone):
    python joint_sp/self_improving/prepare.py \
        --model joint_sp/models/optogpt_60deg_sp_best.pt \
        --targets ood_targets.npy \
        --output prepared_candidates.pkl
"""

import os
import sys
import json
import pickle
import argparse
import hashlib
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_JOINT_SP = _SCRIPT_DIR.parent
_PKG_ROOT = _JOINT_SP.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from optogpt.core.datasets.sim import load_materials
from joint_sp.constants import (
    SPEC_DIM, ALLOWED_MATERIALS, BANNED_MATERIALS,
    THETA_DEG, SUBSTRATE, SUBSTRATE_THICK_NM, MAX_LAYERS,
)
from joint_sp.model import make_model_SP, load_sp_from_pretrained, load_joint_sp_checkpoint
from joint_sp.io_utils import atomic_pickle_dump
from joint_sp.decoder import (
    generate_candidates_sp, tmm_rerank_joint, build_joint_logits_mask,
    structure_to_tuple,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_WL = np.arange(0.4, 1.101, 0.01)


def structure_hash(tokens):
    return hashlib.sha256("|".join(tokens).encode()).hexdigest()


def prepare_candidates(model, word_dict, index_dict, ood_targets, nk_dict,
                       num_candidates=32, logits_mask=None, device=None,
                       theta=THETA_DEG, target_labels=None):
    """
    Generate and TMM-rank candidates for each OOD target.

    Args:
        model: TransformerSP
        word_dict, index_dict: vocabulary
        ood_targets: (N, 284) array of OOD target spectra
        nk_dict: material nk
        num_candidates: candidates per target
        logits_mask: vocab mask
        device: torch device

    Returns:
        list of dicts: {target_idx, target_spec, best_structure, best_spec_joint,
                        structure_hash, joint_error, mean_Ts, mean_Tp, ...}
    """
    if device is None:
        device = DEVICE
    model.eval()

    results = []

    for idx in tqdm(range(len(ood_targets)), desc="Preparing"):
        spec_joint = ood_targets[idx]

        candidates = generate_candidates_sp(
            model, spec_joint, word_dict, index_dict,
            num_candidates=num_candidates,
            max_len=MAX_LAYERS + 2,
            device=device, logits_mask=logits_mask,
        )

        if len(candidates) == 0:
            continue

        label = target_labels[idx] if target_labels is not None else None
        objective = "high_transmission" if label == "broadband_high_T" else "joint_error"
        ranked, _failures = tmm_rerank_joint(
            candidates, spec_joint, nk_dict,
            wavelengths=DEFAULT_WL, theta=theta, objective=objective,
        )

        if len(ranked) == 0:
            continue

        best = ranked[0]

        # Build joint spectrum from TMM
        best_joint = np.concatenate([
            np.array(best['sim_Rs']), np.array(best['sim_Ts']),
            np.array(best['sim_Rp']), np.array(best['sim_Tp']),
        ]).astype(np.float32)

        results.append({
            'target_idx': idx,
            'target_spec_joint': spec_joint,
            'best_tokens': best['tokens'],
            'best_materials': best['materials'],
            'best_thicknesses': best['thicknesses'],
            'best_spec_joint': best_joint,
            'structure_hash': structure_hash(best['tokens']),
            'E_joint': best['E_joint'],
            'mean_Ts': best['mean_Ts'],
            'mean_Tp': best['mean_Tp'],
            'worst_pol_mean_T': best['worst_pol_mean_T'],
            'target_label': label,
            'ranking_objective': best['ranking_objective'],
            'ranking_score': best['ranking_score'],
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Prepare candidates for self-improving")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--targets", type=str, required=True,
                        help="OOD targets .npy file")
    parser.add_argument("--output", type=str, default="prepared_candidates.pkl")
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--architecture_override", type=str, default=None,
                        choices=["joint_sp_legacy_v1", "joint_sp_relu_v0"],
                        help="Only for unversioned legacy joint checkpoints. "
                             "Cannot replace a checkpoint's saved architecture_version.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("Self-Improving: Prepare Candidates")
    print(f"  Model: {args.model}")
    print(f"  Targets: {args.targets}")
    print(f"  Candidates per target: {args.num_candidates}")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading model...")
    model, word_dict, index_dict, sp_configs = load_joint_sp_checkpoint(
        args.model, device=DEVICE,
        architecture_override=args.architecture_override,
    )
    model.eval()

    # Load NK
    nk_dict = load_materials(
        all_mats=[SUBSTRATE] + ALLOWED_MATERIALS + list(BANNED_MATERIALS),
        wavelengths=DEFAULT_WL,
        DATABASE=str(_PKG_ROOT / "optogpt" / "nk"),
    )

    # Load targets
    ood_targets = np.load(args.targets).astype(np.float32)
    if ood_targets.ndim == 1:
        ood_targets = ood_targets.reshape(1, -1)
    print(f"  Loaded {len(ood_targets)} OOD targets")

    # Build mask
    logits_mask, special_ids = build_joint_logits_mask(word_dict, ALLOWED_MATERIALS)

    # Generate
    results = prepare_candidates(
        model, word_dict, index_dict, ood_targets, nk_dict,
        num_candidates=args.num_candidates,
        logits_mask=logits_mask,
    )

    print(f"\n  Generated {len(results)} valid candidates")

    # Save
    atomic_pickle_dump(results, args.output)
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
