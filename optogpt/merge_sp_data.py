"""
Merge s-pol and p-pol datasets into a unified training set.

Usage:
    python merge_sp_data.py --s_dir ../data_60deg_s_500k_dielectric --p_dir ../data_60deg_p_500k_dielectric --out_dir ../data_60deg_sp_1M_dielectric
"""

import os
import sys
import argparse
import pickle as pkl
import numpy as np
from pathlib import Path


def merge_and_save(s_dir, p_dir, out_dir, name):
    """Load, merge, save train/dev/test split files."""
    os.makedirs(out_dir, exist_ok=True)

    # Load s
    with open(os.path.join(s_dir, f"Structure_{name}.pkl"), "rb") as f:
        s_struct = pkl.load(f)
    with open(os.path.join(s_dir, f"Spectrum_{name}.pkl"), "rb") as f:
        s_spec = pkl.load(f)

    # Load p
    with open(os.path.join(p_dir, f"Structure_{name}.pkl"), "rb") as f:
        p_struct = pkl.load(f)
    with open(os.path.join(p_dir, f"Spectrum_{name}.pkl"), "rb") as f:
        p_spec = pkl.load(f)

    # Merge
    merged_struct = s_struct + p_struct
    merged_spec = np.concatenate([s_spec, p_spec], axis=0)

    print(f"  {name}: s={len(s_struct)} + p={len(p_struct)} = {len(merged_struct)}, "
          f"spec shape {merged_spec.shape}")

    # Shuffle together (same seed for reproducibility)
    rng = np.random.default_rng(42)
    n = len(merged_struct)
    idx = rng.permutation(n)
    merged_struct = [merged_struct[i] for i in idx]
    merged_spec = merged_spec[idx]

    # Save
    with open(os.path.join(out_dir, f"Structure_{name}.pkl"), "wb") as f:
        pkl.dump(merged_struct, f)
    with open(os.path.join(out_dir, f"Spectrum_{name}.pkl"), "wb") as f:
        pkl.dump(merged_spec.astype(np.float32), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s_dir", required=True)
    parser.add_argument("--p_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    print(f"s_dir:  {args.s_dir}")
    print(f"p_dir:  {args.p_dir}")
    print(f"out_dir: {args.out_dir}")
    print()

    for name in ["train", "dev"]:
        merge_and_save(args.s_dir, args.p_dir, args.out_dir, name)

    print("\nTest sets kept separate for independent evaluation:")
    print(f"  s-test: {args.s_dir}/Spectrum_test.pkl")
    print(f"  p-test: {args.p_dir}/Spectrum_test.pkl")


if __name__ == "__main__":
    main()
