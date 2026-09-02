"""
joint_sp/scripts/deploy.py — Deployment entry for joint s+p inverse design.

Loads a trained TransformerSP model and generates optimal multilayer structures
for a given 284-dim target spectrum or broadband high-T preset.

Usage:
    python joint_sp/scripts/deploy.py \
        --model joint_sp/models/optogpt_60deg_sp_best.pt \
        --target broadband_high_T --num_candidates 64
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_JOINT_SP = _SCRIPT_DIR.parent
_PKG_ROOT = _JOINT_SP.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from optogpt.core.datasets.sim import load_materials
from joint_sp.constants import (
    SPEC_DIM, BRANCH_DIM, ALLOWED_MATERIALS, BANNED_MATERIALS,
    THETA_DEG, WAVELENGTHS_NM, WAVELENGTHS_UM, SUBSTRATE, SUBSTRATE_THICK_NM,
    MAX_LAYERS,
    validate_joint_spectrum,
)
from joint_sp.model import load_sp_from_pretrained, make_model_SP, load_joint_sp_checkpoint
from joint_sp.decoder import (
    generate_candidates_sp, tmm_rerank_joint, build_joint_logits_mask,
)
from joint_sp.io_utils import atomic_json_dump

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_WL = np.arange(0.4, 1.101, 0.01)


def make_broadband_high_T_target():
    """Create a broadband high-transmittance target: Rs=0, Ts=1, Rp=0, Tp=1."""
    n_pts = BRANCH_DIM // 2  # 71
    Rs = np.zeros(n_pts, dtype=np.float32)
    Ts = np.ones(n_pts, dtype=np.float32)
    Rp = np.zeros(n_pts, dtype=np.float32)
    Tp = np.ones(n_pts, dtype=np.float32)
    return np.concatenate([Rs, Ts, Rp, Tp]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Deploy joint s+p inverse design")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--target", type=str, default="broadband_high_T",
                        help="Target spectrum file (.csv/.npy) or 'broadband_high_T'")
    parser.add_argument("--objective", choices=["auto", "joint_error", "high_transmission"],
                        default="auto")
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--theta", type=float, default=THETA_DEG)
    parser.add_argument("--mean_t_threshold", type=float, default=0.85)
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
    print(f"Joint s+p Deployment: θ={args.theta}°")
    print(f"  Model: {args.model}")
    print(f"  Target: {args.target}")
    print(f"  Candidates: {args.num_candidates}")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading model...")
    model, word_dict, index_dict, sp_configs = load_joint_sp_checkpoint(
        args.model, device=DEVICE,
        architecture_override=args.architecture_override,
    )
    model.eval()

    # Load NK
    print("\n[2/4] Loading NK database...")
    nk_dict = load_materials(
        all_mats=[SUBSTRATE] + ALLOWED_MATERIALS + list(BANNED_MATERIALS),
        wavelengths=DEFAULT_WL,
        DATABASE=str(_PKG_ROOT / "optogpt" / "nk"),
    )

    # Load/create target
    print("\n[3/4] Preparing target spectrum...")
    if args.target == "broadband_high_T":
        target_spec = make_broadband_high_T_target()
        print(f"  Using broadband high-T target: Rs=0, Ts=1, Rp=0, Tp=1")
    elif args.target.endswith('.npy'):
        target_spec = np.load(args.target).astype(np.float32)
    elif args.target.endswith('.csv'):
        target_spec = np.loadtxt(args.target, delimiter=',').astype(np.float32)
    else:
        raise ValueError(f"Unknown target format: {args.target}")

    if target_spec.ndim == 2 and target_spec.shape[0] == 1:
        target_spec = target_spec[0]
    target_spec = validate_joint_spectrum(target_spec, context="deployment target")

    print(f"  Target shape: ({SPEC_DIM},)")

    # Build logits mask
    logits_mask, special_ids = build_joint_logits_mask(word_dict, ALLOWED_MATERIALS)

    # Generate
    print(f"\n[4/4] Generating {args.num_candidates} candidates...")
    candidates = generate_candidates_sp(
        model, target_spec, word_dict, index_dict,
        num_candidates=args.num_candidates,
        max_len=MAX_LAYERS + 2,
        device=DEVICE, logits_mask=logits_mask,
    )
    print(f"  Generated {len(candidates)} unique candidates")

    # TMM re-rank
    objective = args.objective
    if objective == "auto":
        objective = "high_transmission" if args.target == "broadband_high_T" else "joint_error"
    ranked, _failures = tmm_rerank_joint(
        candidates, target_spec, nk_dict,
        wavelengths=DEFAULT_WL, theta=args.theta, objective=objective,
    )

    # Filter by threshold
    passing = [r for r in ranked
               if r['mean_Ts'] > args.mean_t_threshold
               and r['mean_Tp'] > args.mean_t_threshold]

    print(f"\n{'='*70}")
    print(f"Results")
    print(f"{'='*70}")
    print(f"  Total candidates: {len(ranked)}")
    print(f"  Passing (mean_T > {args.mean_t_threshold}): {len(passing)}")

    if len(passing) > 0:
        best = passing[0]
        print(f"\n  Best structure:")
        print(f"    Materials: {best['materials']}")
        print(f"    Thicknesses: {best['thicknesses']} nm")
        print(f"    Layers: {best['n_layers']}")
        print(f"    mean_Ts: {best['mean_Ts']:.4f}")
        print(f"    mean_Tp: {best['mean_Tp']:.4f}")
        print(f"    worst_pol_mean_T: {best['worst_pol_mean_T']:.4f}")
        print(f"    mean_unpolarized_T: {best['mean_unpolarized_T']:.4f}")
        print(f"    E_joint: {best['E_joint']:.4f}")
    elif len(ranked) > 0:
        best = ranked[0]
        print(f"\n  Best (below threshold):")
        print(f"    mean_Ts: {best['mean_Ts']:.4f}, mean_Tp: {best['mean_Tp']:.4f}")

    # Save
    output_path = args.output or f"deploy_result_{Path(args.model).stem}.json"
    output_data = {
        'target': args.target,
        'mean_t_threshold': args.mean_t_threshold,
        'n_candidates': args.num_candidates,
        'n_generated': len(candidates),
        'n_passing': len(passing),
        'top_results': [
            {
                'materials': r['materials'],
                'thicknesses': r['thicknesses'],
                'n_layers': r['n_layers'],
                'mean_Ts': r['mean_Ts'],
                'mean_Tp': r['mean_Tp'],
                'worst_pol_mean_T': r['worst_pol_mean_T'],
                'mean_unpolarized_T': r['mean_unpolarized_T'],
                'E_joint': r['E_joint'],
                'ranking_objective': r['ranking_objective'],
                'ranking_score': r['ranking_score'],
            }
            for r in (passing[:5] if passing else ranked[:5])
        ],
    }
    atomic_json_dump(output_data, output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
