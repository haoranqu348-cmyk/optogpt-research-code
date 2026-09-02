"""
joint_sp/scripts/validate.py — Joint s+p TMM validation.

Decodes structures from 284-dim test spectra, computes both s and p TMM,
and evaluates joint performance metrics.

Usage:
    python joint_sp/scripts/validate.py \
        --model joint_sp/models/optogpt_60deg_sp_best.pt \
        --test_spec data_60deg_sp_joint/Spectrum_test.pkl \
        --test_struct data_60deg_sp_joint/Structure_test.pkl \
        --output_dir joint_sp/validation_results
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

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
)
from joint_sp.model import load_sp_from_pretrained, load_joint_sp_checkpoint
from joint_sp.decoder import (
    generate_candidates_sp, tmm_rerank_joint, build_joint_logits_mask,
    parse_structure, is_valid_structure,
)
from joint_sp.io_utils import atomic_json_dump

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_WL = np.arange(0.4, 1.101, 0.01)


def main():
    parser = argparse.ArgumentParser(description="Joint s+p validation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_spec", type=str, required=True)
    parser.add_argument("--test_struct", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="joint_sp/validation_results")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--mean_t_threshold", type=float, default=0.9)
    parser.add_argument("--p05_t_threshold", type=float, default=0.8)
    parser.add_argument("--theta", type=float, default=THETA_DEG)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--architecture_override", type=str, default=None,
                        choices=["joint_sp_legacy_v1", "joint_sp_relu_v0"],
                        help="Only for unversioned legacy joint checkpoints. "
                             "Cannot replace a checkpoint's saved architecture_version.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = output_dir / "VALIDATION_COMPLETE.json"
    if complete_marker.exists():
        complete_marker.unlink()
    atomic_json_dump(
        {"status": "in_progress", "started_at": datetime.now().isoformat()},
        output_dir / "VALIDATION_IN_PROGRESS.json",
    )

    print("=" * 70)
    print(f"Joint s+p Validation: θ={args.theta}°")
    print(f"  Model: {args.model}")
    print(f"  Test data: {args.test_spec}")
    print(f"  Samples: {args.num_samples}, Candidates: {args.num_candidates}")
    print(f"  Thresholds: mean_T > {args.mean_t_threshold}, p05_T > {args.p05_t_threshold}")
    print("=" * 70)

    # Load model
    print("\n[1/5] Loading model...")
    # Use unified loader — handles architecture_version correctly
    model, word_dict, index_dict, sp_configs = load_joint_sp_checkpoint(
        args.model, device=DEVICE,
        architecture_override=args.architecture_override,
    )
    model.eval()

    # Load NK
    print("\n[2/5] Loading NK database...")
    nk_dict = load_materials(
        all_mats=[SUBSTRATE] + ALLOWED_MATERIALS + list(BANNED_MATERIALS),
        wavelengths=DEFAULT_WL,
        DATABASE=str(_PKG_ROOT / "optogpt" / "nk"),
    )

    # Load test data
    print("\n[3/5] Loading test data...")
    with open(args.test_spec, 'rb') as f:
        test_specs = pickle.load(f)
    with open(args.test_struct, 'rb') as f:
        test_structs = pickle.load(f)

    # Ensure 284-dim
    if isinstance(test_specs, list):
        test_specs = np.array(test_specs)
    if test_specs.shape[1] != SPEC_DIM:
        raise ValueError(f"Test spectra shape {test_specs.shape}, expected (N, {SPEC_DIM})")
    if len(test_specs) != len(test_structs):
        raise ValueError(
            f"Test count mismatch: {len(test_specs)} spectra vs {len(test_structs)} structures"
        )

    # Sample
    n_total = len(test_specs)
    n_samples = min(args.num_samples, n_total)
    indices = np.random.choice(n_total, n_samples, replace=False)

    print(f"  Total test: {n_total}, Sampling: {n_samples}")

    # Build logits mask
    logits_mask, special_ids = build_joint_logits_mask(word_dict, ALLOWED_MATERIALS)

    # Run validation
    print(f"\n[4/5] Running validation...")
    results = []
    failures = []
    n_decode_failed = 0
    n_tmm_failed = 0

    for idx in tqdm(indices, desc="Validating"):
        spec_joint = test_specs[idx]
        target_tokens = test_structs[idx]
        target_tokens_clean = [t for t in target_tokens if t not in ('BOS', 'EOS')]

        candidates = generate_candidates_sp(
            model, spec_joint, word_dict, index_dict,
            num_candidates=args.num_candidates,
            max_len=MAX_LAYERS + 2,
            device=DEVICE, logits_mask=logits_mask,
        )

        if len(candidates) == 0:
            n_decode_failed += 1
            failures.append({'index': int(idx), 'reason': 'decode_failed', 'n_candidates': 0})
            continue

        ranked, tmm_failures = tmm_rerank_joint(
            candidates, spec_joint, nk_dict,
            wavelengths=DEFAULT_WL, theta=args.theta, objective="joint_error",
        )

        if len(ranked) == 0:
            n_tmm_failed += 1
            failures.append({'index': int(idx), 'reason': 'tmm_failed',
                             'n_candidates': len(candidates), 'tmm_failures': tmm_failures})
            continue

        best = ranked[0]

        joint_success = (
            best['mean_Ts'] > args.mean_t_threshold and
            best['mean_Tp'] > args.mean_t_threshold and
            best['p05_Ts'] > args.p05_t_threshold and
            best['p05_Tp'] > args.p05_t_threshold
        )

        results.append({
            'index': int(idx),
            'target_tokens': target_tokens_clean,
            'best_tokens': best['tokens'],
            'best_materials': best['materials'],
            'best_thicknesses': best['thicknesses'],
            'n_layers': best['n_layers'],
            'E_s': best['E_s'],
            'E_p': best['E_p'],
            'E_joint': best['E_joint'],
            'mean_Ts': best['mean_Ts'],
            'mean_Tp': best['mean_Tp'],
            'p05_Ts': best['p05_Ts'],
            'p05_Tp': best['p05_Tp'],
            'min_Ts': best['min_Ts'],
            'min_Tp': best['min_Tp'],
            'mean_unpolarized_T': best['mean_unpolarized_T'],
            'worst_pol_mean_T': best['worst_pol_mean_T'],
            'high_T_loss': best['high_T_loss'],
            'joint_success': joint_success,
            'n_candidates': len(ranked),
        })

    # Compute summary
    print(f"\n[5/5] Computing summary...")
    n_valid = len(results)
    n_success = sum(1 for r in results if r['joint_success'])

    def safe_mean(vals, default=None):
        return float(np.mean(vals)) if len(vals) > 0 else default

    summary = {
        'model': str(args.model),
        'test_spec': str(args.test_spec),
        'n_total': n_total,
        'n_sampled': n_samples,
        'n_valid': n_valid,
        'n_decode_failed': n_decode_failed,
        'n_tmm_failed': n_tmm_failed,
        'theta_deg': args.theta,
        'mean_t_threshold': args.mean_t_threshold,
        'p05_t_threshold': args.p05_t_threshold,
        'valid_decode_rate': n_valid / n_samples if n_samples > 0 else 0.0,
        'conditional_joint_success': n_success / n_valid if n_valid > 0 else None,
        'end_to_end_joint_success': n_success / n_samples if n_samples > 0 else 0.0,
        'metrics': {
            'E_s_mean': safe_mean([r['E_s'] for r in results]),
            'E_p_mean': safe_mean([r['E_p'] for r in results]),
            'E_joint_mean': safe_mean([r['E_joint'] for r in results]),
            'mean_Ts_avg': safe_mean([r['mean_Ts'] for r in results]),
            'mean_Tp_avg': safe_mean([r['mean_Tp'] for r in results]),
            'p05_Ts_avg': safe_mean([r['p05_Ts'] for r in results]),
            'p05_Tp_avg': safe_mean([r['p05_Tp'] for r in results]),
            'min_Ts_avg': safe_mean([r['min_Ts'] for r in results]),
            'min_Tp_avg': safe_mean([r['min_Tp'] for r in results]),
            'mean_unpolarized_T_avg': safe_mean([r['mean_unpolarized_T'] for r in results]),
            'worst_pol_mean_T_avg': safe_mean([r['worst_pol_mean_T'] for r in results]),
        },
        'failed_samples': failures[:20],
    }

    print(f"\n{'='*70}")
    print(f"Joint Validation Results")
    print(f"{'='*70}")
    print(f"  Sampled: {n_samples}, Valid: {n_valid}, Decode failed: {n_decode_failed}, TMM failed: {n_tmm_failed}")
    print(f"  End-to-end joint success: {n_success}/{n_samples} = {summary['end_to_end_joint_success']:.1%}")
    print(f"  Conditional joint success: {n_success}/{n_valid}" if n_valid > 0 else "  No valid results")
    if n_valid > 0:
        print(f"  E_joint mean: {summary['metrics']['E_joint_mean']:.4f}")
        print(f"  mean_Ts: {summary['metrics']['mean_Ts_avg']:.4f}")
        print(f"  mean_Tp: {summary['metrics']['mean_Tp_avg']:.4f}")
        print(f"  worst_pol_mean_T: {summary['metrics']['worst_pol_mean_T_avg']:.4f}")

    # Save (sanitize NaN/Inf → null for JSON compliance)
    import math
    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    atomic_json_dump(sanitize(results), output_dir / "validation_results.json")
    atomic_json_dump(sanitize(summary), output_dir / "summary.json")
    atomic_json_dump(
        {"status": "complete", "n_sampled": n_samples, "n_valid": n_valid,
         "created_at": datetime.now().isoformat()},
        complete_marker,
    )
    in_progress = output_dir / "VALIDATION_IN_PROGRESS.json"
    if in_progress.exists():
        in_progress.unlink()

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
