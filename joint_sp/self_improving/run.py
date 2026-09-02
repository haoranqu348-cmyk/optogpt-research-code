"""
joint_sp/self_improving/run.py — Joint s+p Self-Improving Pipeline Main Entry.

Orchestrates the 7-step self-improving pipeline:
  1. Generate OOD targets
  2. Prepare: decode candidate structures
  3. Perturb: GA/PSO optimization (joint s+p fitness)
  4. Filter: keep only improved structures
  5. Dedup: by structure_hash (NOT floating-point error!)
  6. Leakage check: no overlap with dev/test
  7. Save: added_data.pkl

Usage:
    python joint_sp/self_improving/run.py \
        --model_path joint_sp/models/optogpt_60deg_sp_best.pt \
        --train_struct_path data_60deg_sp_joint/Structure_train.pkl \
        --train_spec_path data_60deg_sp_joint/Spectrum_train.pkl \
        --dev_struct_path data_60deg_sp_joint/Structure_dev.pkl \
        --dev_spec_path data_60deg_sp_joint/Spectrum_dev.pkl \
        --output_dir joint_sp/self_improving_output \
        --augment_only --theta 60
"""

import os
import sys
import json
import pickle
import hashlib
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

_SCRIPT_DIR = Path(__file__).resolve().parent
_JOINT_SP = _SCRIPT_DIR.parent
_PKG_ROOT = _JOINT_SP.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from optogpt.core.datasets.sim import load_materials
from joint_sp.constants import (
    SPEC_DIM, BRANCH_DIM, ALLOWED_MATERIALS, BANNED_MATERIALS,
    THETA_DEG, SUBSTRATE, SUBSTRATE_THICK_NM, MAX_LAYERS,
)
from joint_sp.model import make_model_SP, load_sp_from_pretrained, load_joint_sp_checkpoint
from joint_sp.decoder import build_joint_logits_mask
from joint_sp.io_utils import atomic_json_dump, atomic_numpy_save, atomic_pickle_dump

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_WL = np.arange(0.4, 1.101, 0.01)


def structure_hash_from_tokens(tokens):
    return hashlib.sha256("|".join(tokens).encode()).hexdigest()


def file_fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def extract_hashes_from_pkl(struct_path):
    """Extract structure hashes from a Structure .pkl file."""
    with open(struct_path, 'rb') as f:
        structs = pickle.load(f)
    hashes = set()
    for s in structs:
        tokens = [t for t in s if t not in ('BOS', 'EOS')]
        hashes.add(structure_hash_from_tokens(tokens))
    return hashes


def main():
    parser = argparse.ArgumentParser(description="Joint s+p Self-Improving Pipeline")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_struct_path", type=str, required=True)
    parser.add_argument("--dev_struct_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="joint_sp/self_improving_output")
    parser.add_argument("--perturbation_method", type=str, default="GA_PSO")
    parser.add_argument("--target_aug_size", type=int, default=200000,
                        help="Desired augmentation size (actual may be less)")
    parser.add_argument("--theta", type=float, default=THETA_DEG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--architecture_override", type=str, default=None,
                        choices=["joint_sp_legacy_v1", "joint_sp_relu_v0"],
                        help="Only for unversioned legacy joint checkpoints. "
                             "Cannot replace a checkpoint's saved architecture_version.")
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--n_ood_targets", type=int, default=100)
    parser.add_argument("--min_improvement", type=float, default=1e-4)
    parser.add_argument("--max_joint_error", type=float, default=0.15)
    parser.add_argument("--min_high_t_worst_pol", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = output_dir / "SELF_IMPROVING_COMPLETE.json"
    if complete_marker.exists():
        if args.resume:
            print(f"Self-improving output already complete: {complete_marker}")
            return
        if not args.overwrite:
            raise FileExistsError(
                f"Completed output already exists at {output_dir}; pass --overwrite to rebuild"
            )
        complete_marker.unlink()
    atomic_json_dump(
        {"status": "in_progress", "started_at": datetime.now().isoformat()},
        output_dir / "SELF_IMPROVING_IN_PROGRESS.json",
    )
    run_manifest = {
        "model_sha256": file_fingerprint(args.model_path),
        "train_struct_sha256": file_fingerprint(args.train_struct_path),
        "dev_struct_sha256": file_fingerprint(args.dev_struct_path),
        "theta": args.theta,
        "seed": args.seed,
        "num_candidates": args.num_candidates,
        "n_ood_targets": args.n_ood_targets,
        "perturbation_method": args.perturbation_method,
        "min_improvement": args.min_improvement,
        "max_joint_error": args.max_joint_error,
        "min_high_t_worst_pol": args.min_high_t_worst_pol,
    }
    manifest_path = output_dir / "run_manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Resume manifest not found: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as f:
            saved_manifest = json.load(f)
        if saved_manifest != run_manifest:
            raise RuntimeError("Self-improving inputs or parameters changed since resume point")
    else:
        atomic_json_dump(run_manifest, manifest_path)

    print("=" * 70)
    print(f"Joint s+p Self-Improving Pipeline")
    print(f"  Model: {args.model_path}")
    print(f"  Method: {args.perturbation_method}")
    print(f"  Target aug size: {args.target_aug_size}")
    print(f"  OOD targets: {args.n_ood_targets}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # ---- Load model ----
    print("\n[Step 1/7] Loading model...")
    model, word_dict, index_dict, sp_configs = load_joint_sp_checkpoint(
        args.model_path, device=DEVICE,
        architecture_override=args.architecture_override,
    )
    model.eval()

    # ---- Load NK ----
    nk_dict = load_materials(
        all_mats=[SUBSTRATE] + ALLOWED_MATERIALS + list(BANNED_MATERIALS),
        wavelengths=DEFAULT_WL,
        DATABASE=str(_PKG_ROOT / "optogpt" / "nk"),
    )

    # ---- Load train/dev data for hashes ----
    print("\n[Step 2/7] Loading data hashes...")
    dev_hashes = extract_hashes_from_pkl(args.dev_struct_path)

    # Also load test hashes if available
    test_struct_path = Path(args.dev_struct_path).parent / "Structure_test.pkl"
    test_hashes = set()
    if test_struct_path.exists():
        test_hashes = extract_hashes_from_pkl(str(test_struct_path))

    train_hashes = extract_hashes_from_pkl(args.train_struct_path)
    print(f"  Train: {len(train_hashes)} hashes")
    print(f"  Dev: {len(dev_hashes)} hashes")
    print(f"  Test: {len(test_hashes)} hashes")

    # ---- Generate OOD targets ----
    print(f"\n[Step 3/7] Generating {args.n_ood_targets} OOD targets...")
    from joint_sp.self_improving.ood_targets import generate_all_ood_targets
    targets_path = output_dir / "ood_targets.npy"
    labels_path = output_dir / "ood_labels.json"
    if args.resume and targets_path.exists() and labels_path.exists():
        ood_targets = np.load(targets_path)
        with open(labels_path, encoding="utf-8") as f:
            ood_labels = json.load(f)
    else:
        ood_targets, ood_labels = generate_all_ood_targets(
            n_broadband=1,
            n_gaussian=args.n_ood_targets // 2,
            n_double_gaussian=args.n_ood_targets // 4,
            n_dbr=max(10, args.n_ood_targets // 4),
            n_random=max(10, args.n_ood_targets // 10),
            seed=args.seed,
            nk_dict=nk_dict,
            theta=args.theta,
        )
    print(f"  Generated {len(ood_targets)} OOD targets")

    # Save OOD targets
    atomic_numpy_save(ood_targets, targets_path)
    atomic_json_dump(ood_labels, labels_path)

    # ---- Build logits mask ----
    logits_mask, special_ids = build_joint_logits_mask(word_dict, ALLOWED_MATERIALS)

    # ---- Prepare candidates ----
    print(f"\n[Step 4/7] Preparing candidates via model decoding...")
    from joint_sp.self_improving.prepare import prepare_candidates
    prepared_path = output_dir / "prepared_candidates.pkl"
    if args.resume and prepared_path.exists():
        with open(prepared_path, "rb") as f:
            prepared = pickle.load(f)
    else:
        prepared = prepare_candidates(
            model, word_dict, index_dict, ood_targets, nk_dict,
            num_candidates=args.num_candidates,
            logits_mask=logits_mask,
            device=DEVICE,
            theta=args.theta,
            target_labels=ood_labels,
        )
    print(f"  Prepared {len(prepared)} candidates")

    # Save intermediate
    atomic_pickle_dump(prepared, prepared_path)

    # ---- Perturb ----
    print(f"\n[Step 5/7] Perturbing via {args.perturbation_method}...")

    # Get valid thicknesses from vocab
    valid_thicknesses = sorted(set(
        int(t.split('_')[1]) for t in word_dict
        if '_' in t and t.split('_')[0] in ALLOWED_MATERIALS
        and t.split('_')[1].isdigit()
    ))
    if not valid_thicknesses:
        valid_thicknesses = list(range(10, 201, 10))
    print(f"  Valid thicknesses: {len(valid_thicknesses)} values")

    from joint_sp.self_improving.perturb import perturb_structures
    perturbed_path = output_dir / "perturbed_results.pkl"
    if args.resume and perturbed_path.exists():
        with open(perturbed_path, "rb") as f:
            perturbed = pickle.load(f)
    else:
        perturbed = perturb_structures(
            prepared, nk_dict,
            method=args.perturbation_method,
            allowed_materials=ALLOWED_MATERIALS,
            valid_thicknesses=valid_thicknesses,
            seed=args.seed,
            theta=args.theta,
            min_improvement=args.min_improvement,
            max_joint_error=args.max_joint_error,
            min_high_t_worst_pol=args.min_high_t_worst_pol,
        )
    print(f"  Perturbed (improved): {len(perturbed)} structures")

    # Save intermediate
    atomic_pickle_dump(perturbed, perturbed_path)

    # ---- Combine & dedup ----
    print(f"\n[Step 6/7] Combining and deduplicating...")
    from joint_sp.self_improving.combine import combine_perturbed
    combined = combine_perturbed(perturbed, dev_hashes, test_hashes)

    print(f"  After combining: {combined['stats']['final_count']} structures")

    # DO NOT copy to hit target — use what we have
    final_count = combined['stats']['final_count']
    print(f"  Final augmented count: {final_count}")
    if final_count < args.target_aug_size:
        print(f"  (Target was {args.target_aug_size}, got {final_count} — "
              f"NOT copying to inflate)")

    # ---- Save final augmented data ----
    print(f"\n[Step 7/7] Saving augmented data...")
    added_data = {
        'perturb_struct': combined['structures'],
        'perturb_spec_joint': combined['spectra'],
        'structure_hashes': combined['hashes'],
        'stats': combined['stats'],
        'config': {
            'model_path': str(args.model_path),
            'perturbation_method': args.perturbation_method,
            'target_aug_size': args.target_aug_size,
            'actual_count': final_count,
            'n_ood_targets': args.n_ood_targets,
            'n_prepared': len(prepared),
            'n_perturbed': len(perturbed),
            'seed': args.seed,
            'theta_deg': args.theta,
            'created_at': datetime.now().isoformat(),
        },
    }

    atomic_pickle_dump(added_data, output_dir / "added_data.pkl")

    # Save summary JSON for readability
    summary = {
        'final_count': final_count,
        'n_ood_targets': len(ood_targets),
        'n_prepared': len(prepared),
        'n_perturbed_improved': len(perturbed),
        'stats': combined['stats'],
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    atomic_json_dump(
        {"status": "complete", "final_count": final_count,
         "created_at": datetime.now().isoformat()},
        complete_marker,
    )
    in_progress = output_dir / "SELF_IMPROVING_IN_PROGRESS.json"
    if in_progress.exists():
        in_progress.unlink()

    print(f"\n{'='*70}")
    print(f"Self-Improving Complete!")
    print(f"  Augmented data: {final_count} structures")
    print(f"  Output: {output_dir / 'added_data.pkl'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
