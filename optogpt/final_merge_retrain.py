"""
Merge self-improving enhanced data with original training data for final retraining.

After running self-improving with --augment_only on s-pol and p-pol separately,
this script combines the enhanced structures from both runs with the original training data.

Usage:
    python final_merge_retrain.py \
        --original_dir ../data_60deg_sp_1M_dielectric \
        --si_s_dir ./self_improving_60s_dielectric \
        --si_p_dir ./self_improving_60p_dielectric \
        --out_dir ../data_60deg_sp_ultimate
"""

import os
import sys
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", required=True,
                        help="Path to original s+p merged dataset")
    parser.add_argument("--si_s_dir", required=True,
                        help="Path to s-pol self-improving output")
    parser.add_argument("--si_p_dir", required=True,
                        help="Path to p-pol self-improving output")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for ultimate training set")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load original merged data ----
    print("Loading original merged data...")
    with open(os.path.join(args.original_dir, "Structure_train.pkl"), "rb") as f:
        orig_train_struct = pkl.load(f)
    with open(os.path.join(args.original_dir, "Spectrum_train.pkl"), "rb") as f:
        orig_train_spec = pkl.load(f)
    with open(os.path.join(args.original_dir, "Structure_dev.pkl"), "rb") as f:
        orig_dev_struct = pkl.load(f)
    with open(os.path.join(args.original_dir, "Spectrum_dev.pkl"), "rb") as f:
        orig_dev_spec = pkl.load(f)
    print(f"  Original train: {len(orig_train_struct)}, dev: {len(orig_dev_struct)}")

    # ---- Load s-pol enhanced data ----
    si_s_added = os.path.join(args.si_s_dir, "augmented_data", "added_data.pkl")
    if os.path.exists(si_s_added):
        df_s = pd.read_pickle(si_s_added)
        s_structs = df_s["perturb_struct"].tolist()
        s_specs = df_s["perturb_spec"].tolist()
        print(f"  s-enhanced: {len(s_structs)} structures")
    else:
        s_structs, s_specs = [], []
        print(f"  WARNING: s-enhanced data not found at {si_s_added}")

    # ---- Load p-pol enhanced data ----
    si_p_added = os.path.join(args.si_p_dir, "augmented_data", "added_data.pkl")
    if os.path.exists(si_p_added):
        df_p = pd.read_pickle(si_p_added)
        p_structs = df_p["perturb_struct"].tolist()
        p_specs = df_p["perturb_spec"].tolist()
        print(f"  p-enhanced: {len(p_structs)} structures")
    else:
        p_structs, p_specs = [], []
        print(f"  WARNING: p-enhanced data not found at {si_p_added}")

    # ---- Merge ultimate train set ----
    ultimate_train_struct = orig_train_struct + s_structs + p_structs
    ultimate_train_spec = np.concatenate([
        orig_train_spec,
        np.array(s_specs, dtype=np.float32) if s_specs else np.array([]).reshape(0, 142),
        np.array(p_specs, dtype=np.float32) if p_specs else np.array([]).reshape(0, 142),
    ], axis=0)

    # Shuffle
    rng = np.random.default_rng(42)
    n = len(ultimate_train_struct)
    idx = rng.permutation(n)
    ultimate_train_struct = [ultimate_train_struct[i] for i in idx]
    ultimate_train_spec = ultimate_train_spec[idx]

    print(f"\n  Ultimate train: {len(ultimate_train_struct)} structures, "
          f"spec shape {ultimate_train_spec.shape}")
    print(f"  Dev (unchanged): {len(orig_dev_struct)}")

    # ---- Save ----
    with open(os.path.join(args.out_dir, "Structure_train.pkl"), "wb") as f:
        pkl.dump(ultimate_train_struct, f)
    with open(os.path.join(args.out_dir, "Spectrum_train.pkl"), "wb") as f:
        pkl.dump(ultimate_train_spec.astype(np.float32), f)

    # Dev: copy from original (unchanged, since SI didn't touch it)
    with open(os.path.join(args.out_dir, "Structure_dev.pkl"), "wb") as f:
        pkl.dump(orig_dev_struct, f)
    with open(os.path.join(args.out_dir, "Spectrum_dev.pkl"), "wb") as f:
        pkl.dump(orig_dev_spec, f)

    print(f"\nUltimate training set saved to: {args.out_dir}")
    print(f"  Structure_train.pkl + Spectrum_train.pkl")
    print(f"  Structure_dev.pkl   + Spectrum_dev.pkl")


if __name__ == "__main__":
    main()
