"""
joint_sp/self_improving/combine.py — Combine perturbed structures with deduplication.

Key rules (ABSOLUTELY ENFORCED):
  - Dedup by structure_hash (SHA-256), NOT by floating-point error!
  - Check no hash leakage with dev/test sets
  - DO NOT copy data to hit target_aug_size (use what we have)

Usage (standalone):
    python joint_sp/self_improving/combine.py \
        --perturbed perturbed_results.pkl \
        --dev_hashes dev_hashes.txt \
        --test_hashes test_hashes.txt \
        --output added_data.pkl
"""

import os
import sys
import json
import pickle
import hashlib
import argparse
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_JOINT_SP = _SCRIPT_DIR.parent
_PKG_ROOT = _JOINT_SP.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from joint_sp.constants import SPEC_DIM, validate_disk_structure_tokens
from joint_sp.io_utils import atomic_pickle_dump


def structure_hash_from_tokens(tokens):
    return hashlib.sha256("|".join(tokens).encode()).hexdigest()


def combine_perturbed(perturbed_results, dev_hashes, test_hashes):
    """
    Combine perturbed results with deduplication and leakage checks.

    Args:
        perturbed_results: list of dicts from perturb.py
        dev_hashes: set of structure hashes in dev set
        test_hashes: set of structure hashes in test set

    Returns:
        dict: {
            'structures': list of token lists (without BOS/EOS),
            'spectra': (N, 284) array,
            'hashes': list of hash strings,
            'stats': dict with counts
        }
    """
    seen = set()
    structures = []
    spectra = []
    hash_list = []
    n_leaks = 0
    n_dups = 0

    for item in perturbed_results:
        try:
            tokens = list(validate_disk_structure_tokens(item['perturb_struct']))
            spec = np.asarray(item['perturb_spec_joint'], dtype=np.float32)
            if spec.shape != (SPEC_DIM,) or not np.all(np.isfinite(spec)):
                raise ValueError("invalid joint spectrum")
        except (KeyError, TypeError, ValueError):
            n_dups += 1
            continue
        h = item.get('struct_hash', structure_hash_from_tokens(tokens))

        # Check leakage
        if h in dev_hashes or h in test_hashes:
            n_leaks += 1
            continue

        # Check dedup
        if h in seen:
            n_dups += 1
            continue

        seen.add(h)
        structures.append(tokens)
        spectra.append(spec)
        hash_list.append(h)

    if len(spectra) > 0:
        spectra = np.array(spectra, dtype=np.float32)
    else:
        spectra = np.empty((0, 284), dtype=np.float32)

    stats = {
        'total_input': len(perturbed_results),
        'after_dedup': len(structures),
        'duplicates_removed': n_dups,
        'leaks_removed': n_leaks,
        'final_count': len(structures),
    }

    return {
        'structures': structures,
        'spectra': spectra,
        'hashes': hash_list,
        'stats': stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Combine perturbed structures")
    parser.add_argument("--perturbed", type=str, required=True,
                        help="Perturbed results .pkl from perturb.py")
    parser.add_argument("--dev_hashes", type=str, default=None,
                        help="File with dev hashes (one per line)")
    parser.add_argument("--test_hashes", type=str, default=None,
                        help="File with test hashes (one per line)")
    parser.add_argument("--dev_struct", type=str, default=None,
                        help="Dev Structure.pkl to extract hashes")
    parser.add_argument("--test_struct", type=str, default=None,
                        help="Test Structure.pkl to extract hashes")
    parser.add_argument("--output", type=str, default="added_data.pkl")
    args = parser.parse_args()

    # Load dev/test hashes
    dev_hashes = set()
    test_hashes = set()

    if args.dev_hashes and Path(args.dev_hashes).exists():
        with open(args.dev_hashes) as f:
            dev_hashes = set(line.strip() for line in f if line.strip())

    if args.test_hashes and Path(args.test_hashes).exists():
        with open(args.test_hashes) as f:
            test_hashes = set(line.strip() for line in f if line.strip())

    # Extract hashes from structure files
    def extract_hashes(struct_file):
        hashes = set()
        with open(struct_file, 'rb') as f:
            structs = pickle.load(f)
        for s in structs:
            tokens = [t for t in s if t not in ('BOS', 'EOS')]
            hashes.add(hashlib.sha256("|".join(tokens).encode()).hexdigest())
        return hashes

    if args.dev_struct and Path(args.dev_struct).exists():
        dev_hashes.update(extract_hashes(args.dev_struct))
    if args.test_struct and Path(args.test_struct).exists():
        test_hashes.update(extract_hashes(args.test_struct))

    print(f"  Dev hashes: {len(dev_hashes)}, Test hashes: {len(test_hashes)}")

    # Load perturbed data
    with open(args.perturbed, 'rb') as f:
        perturbed = pickle.load(f)
    print(f"  Perturbed input: {len(perturbed)} structures")

    # Combine
    result = combine_perturbed(perturbed, dev_hashes, test_hashes)

    print(f"\n  Combination stats:")
    for k, v in result['stats'].items():
        print(f"    {k}: {v}")

    # Save
    output_data = {
        'perturb_struct': result['structures'],
        'perturb_spec_joint': result['spectra'],
        'structure_hashes': result['hashes'],
        'stats': result['stats'],
    }
    atomic_pickle_dump(output_data, args.output)
    print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()
