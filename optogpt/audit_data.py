"""
Data Leakage Audit for OptoGPT 60deg s-pol dataset.

Checks for:
  1. Intra-set exact/duplicate structures
  2. Inter-set duplicate structures (train↔dev, train↔test, dev↔test)
  3. Near-duplicate spectra (configurable threshold)
  4. Special check: TiN_240 presence across sets

Outputs:
  - Console report
  - JSON audit report saved to output_dir
  - Optionally produces cleaned splits (no de-duplication by default)
"""

import os
import sys
import json
import argparse
import pickle as pkl
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent  # optogpt/optogpt/
sys.path.insert(0, str(PROJECT_ROOT))


def load_pkl(path):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return data


def normalize_structure(struc):
    """Remove BOS/EOS/PAD tokens, keep material_thickness tokens in order."""
    if isinstance(struc, list):
        return tuple(
            tok for tok in struc
            if tok not in ("BOS", "EOS", "PAD", "UNK", "")
            and "_" in tok
        )
    return tuple()


def normalize_structure_from_ids(ids, index_dict):
    """Convert token IDs to normalized structure tuple."""
    tokens = []
    for tid in ids:
        sym = index_dict.get(int(tid), "UNK")
        if sym in ("BOS", "EOS", "PAD", "UNK", ""):
            continue
        tokens.append(sym)
    return tuple(tokens)


def find_duplicates(structures, normalize_fn=None):
    """
    Find duplicate structures in a list.
    Returns (unique_count, duplicate_count, duplicate_examples).
    """
    if normalize_fn is None:
        normalize_fn = normalize_structure
    normalized = [normalize_fn(s) for s in structures]
    counter = Counter(normalized)
    duplicates = {k: v for k, v in counter.items() if v > 1}
    unique = len(counter)
    dup_count = sum(v - 1 for v in counter.values() if v > 1)
    return unique, dup_count, duplicates


def find_cross_duplicates(structs_a, structs_b, name_a, name_b, normalize_fn=None):
    """Find structures that appear in both set A and set B."""
    if normalize_fn is None:
        normalize_fn = normalize_structure
    norm_a = set(normalize_fn(s) for s in structs_a)
    norm_b = set(normalize_fn(s) for s in structs_b)
    inter = norm_a & norm_b
    return inter


def find_near_duplicate_spectra(specs, threshold=1e-4, max_comparisons=None):
    """
    Find near-identical spectra using vectorized approach.
    threshold: max MAE to consider two spectra as near-duplicates.
    """
    specs = np.array(specs, dtype=np.float64)
    n = len(specs)
    if max_comparisons and n > max_comparisons:
        # Use random sampling for very large sets
        rng = np.random.RandomState(42)
        indices = rng.choice(n, max_comparisons, replace=False)
        specs_sample = specs[indices]
    else:
        specs_sample = specs
        indices = np.arange(n)

    near_dups = []
    ns = len(specs_sample)
    for i in range(ns):
        # Compare spec i against all j > i
        diffs = np.mean(np.abs(specs_sample[i+1:] - specs_sample[i]), axis=1)
        dup_indices = np.where(diffs < threshold)[0]
        for dj in dup_indices:
            j = i + 1 + dj
            near_dups.append((int(indices[i]), int(indices[j]), float(diffs[dj])))

    return near_dups


def check_special_token(structures, token="TiN_240"):
    """Check where a specific token appears across datasets."""
    results = {}
    for set_name, structs in structures.items():
        count = 0
        examples = []
        for idx, s in enumerate(structs):
            norm = normalize_structure(s)
            if token in norm:
                count += 1
                if len(examples) < 5:
                    examples.append((idx, list(norm)))
        results[set_name] = {"count": count, "examples": examples}
    return results


def audit_data(data_dir, output_dir, spec_threshold=1e-4, index_dict=None, seed=42):
    """Run full data leakage audit."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sets = {}
    for split in ["train", "dev", "test"]:
        struct_path = data_dir / f"Structure_{split}.pkl"
        spec_path = data_dir / f"Spectrum_{split}.pkl"
        if struct_path.exists() and spec_path.exists():
            sets[split] = {
                "structs": load_pkl(struct_path),
                "specs": load_pkl(spec_path),
            }
            print(f"Loaded {split}: {len(sets[split]['structs'])} structures, "
                  f"{len(sets[split]['specs'])} spectra")

    report = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir.resolve()),
        "spec_threshold": spec_threshold,
        "splits": {k: len(v["structs"]) for k, v in sets.items()},
    }

    # 1. Intra-set duplicates
    print("\n=== Intra-set Structure Duplicates ===")
    report["intra_duplicates"] = {}
    for split_name, data in sets.items():
        unique, dup_count, dup_examples = find_duplicates(data["structs"])
        report["intra_duplicates"][split_name] = {
            "total": len(data["structs"]),
            "unique": unique,
            "duplicate_count": dup_count,
            "duplicate_groups": len(dup_examples),
        }
        print(f"  {split_name}: {len(data['structs'])} total, {unique} unique, "
              f"{dup_count} duplicates in {len(dup_examples)} groups")
        if dup_examples:
            for struct, cnt in list(dup_examples.items())[:3]:
                print(f"    Example: {list(struct)} appears {cnt}x")

    # 2. Cross-set duplicates
    print("\n=== Cross-set Structure Duplicates ===")
    report["cross_duplicates"] = {}
    split_names = list(sets.keys())
    for i in range(len(split_names)):
        for j in range(i+1, len(split_names)):
            a, b = split_names[i], split_names[j]
            inter = find_cross_duplicates(
                sets[a]["structs"], sets[b]["structs"], a, b)
            report["cross_duplicates"][f"{a}_vs_{b}"] = {
                "overlap_count": len(inter),
                "examples": [list(s) for s in list(inter)[:5]],
            }
            print(f"  {a} ∩ {b}: {len(inter)} overlapping structures")
            if inter:
                for s in list(inter)[:3]:
                    print(f"    {list(s)}")

    # 3. Near-duplicate spectra
    print("\n=== Near-duplicate Spectra (MAE < {:.2e}) ===")
    report["spectral_near_duplicates"] = {}
    for split_name, data in sets.items():
        specs = np.array(data["specs"])
        # Sample-based check for large sets
        near = find_near_duplicate_spectra(specs, threshold=spec_threshold, max_comparisons=5000)
        report["spectral_near_duplicates"][split_name] = {
            "total_spectra": len(specs),
            "near_duplicate_pairs": len(near),
            "example_pairs": [
                {"idx_a": a, "idx_b": b, "mae": mae}
                for a, b, mae in near[:5]
            ],
        }
        print(f"  {split_name}: {len(near)} near-duplicate spectral pairs")

        # Cross-set spectral near duplicates
        if split_name != list(sets.keys())[-1]:
            for other in split_names:
                if other <= split_name:
                    continue
                # Sample and compare
                specs_a = np.array(data["specs"])
                specs_b = np.array(sets[other]["specs"])
                # Use random sampling for efficiency
                n_sample = min(1000, len(specs_a), len(specs_b))
                rng = np.random.RandomState(seed)
                idx_a = rng.choice(len(specs_a), n_sample, replace=False)
                idx_b = rng.choice(len(specs_b), n_sample, replace=False)
                cross_near = []
                for ia in idx_a:
                    diffs = np.mean(np.abs(specs_b[idx_b] - specs_a[ia]), axis=1)
                    dup_idx = np.where(diffs < spec_threshold)[0]
                    for dj in dup_idx:
                        cross_near.append((int(ia), int(idx_b[dj]), float(diffs[dj])))
                key = f"{split_name}_vs_{other}_spectra"
                report["spectral_near_duplicates"][key] = {
                    "sampled_comparisons": n_sample * n_sample,
                    "near_duplicate_pairs": len(cross_near),
                }
                print(f"  {split_name} vs {other} spectra: {len(cross_near)} near-duplicate pairs (sampled)")

    # 4. Special token check
    print("\n=== TiN_240 Presence Check ===")
    structs_map = {k: v["structs"] for k, v in sets.items()}
    tiN_report = check_special_token(structs_map, "TiN_240")
    report["special_token_TiN_240"] = tiN_report
    for set_name, info in tiN_report.items():
        print(f"  {set_name}: {info['count']} occurrences")
        for idx, struct in info["examples"]:
            print(f"    idx={idx}: {struct}")

    # Also check other special tokens
    for token in ["Ag_40", "Al_30", "TiN_310"]:
        tok_report = check_special_token(structs_map, token)
        report[f"special_token_{token}"] = tok_report
        counts = {k: v["count"] for k, v in tok_report.items()}
        print(f"  {token}: {counts}")

    # Save report
    report_path = output_dir / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nAudit report saved: {report_path}")

    # Summary
    print("\n=== AUDIT SUMMARY ===")
    total_cross = sum(
        v["overlap_count"] for v in report["cross_duplicates"].values())
    has_leakage = total_cross > 0
    print(f"  Cross-set structure leakage: {'DETECTED' if has_leakage else 'NONE'} ({total_cross} total)")
    for split_name, info in report["intra_duplicates"].items():
        print(f"  {split_name}: {info['unique']}/{info['total']} unique "
              f"({info['duplicate_count']} duplicates)")
    return report


def create_clean_splits(data_dir, output_dir, report, seed=42):
    """If leakage is detected, create cleaned splits preserving original data."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    # Load all data
    all_structs = []
    all_specs = []
    splits_map = {}
    offset = 0
    for split in ["train", "dev", "test"]:
        sp = data_dir / f"Structure_{split}.pkl"
        spp = data_dir / f"Spectrum_{split}.pkl"
        if sp.exists():
            structs = load_pkl(sp)
            specs = load_pkl(spp)
            all_structs.extend(structs)
            all_specs.extend(specs)
            n = len(structs)
            splits_map[split] = list(range(offset, offset + n))
            offset += n

    # Remove cross-set duplicates by keeping only in the first set
    seen = set()
    clean_indices = {k: [] for k in splits_map}
    removed = {k: 0 for k in splits_map}

    for split in ["train", "dev", "test"]:
        for idx in splits_map.get(split, []):
            norm = normalize_structure(all_structs[idx])
            if norm in seen:
                removed[split] += 1
            else:
                seen.add(norm)
                clean_indices[split].append(idx)

    print("\n=== Clean Split Creation ===")
    for split in ["train", "dev", "test"]:
        n_orig = len(splits_map.get(split, []))
        n_clean = len(clean_indices.get(split, []))
        print(f"  {split}: {n_orig} → {n_clean} (removed {removed[split]})")

    # Save clean splits
    clean_dir = output_dir / "clean_splits"
    clean_dir.mkdir(exist_ok=True)
    for split in ["train", "dev", "test"]:
        idxs = clean_indices.get(split, [])
        clean_structs = [all_structs[i] for i in idxs]
        clean_specs = [all_specs[i] for i in idxs]
        with open(clean_dir / f"Structure_{split}.pkl", "wb") as f:
            pkl.dump(clean_structs, f)
        with open(clean_dir / f"Spectrum_{split}.pkl", "wb") as f:
            pkl.dump(np.array(clean_specs, dtype=np.float32), f)

    # Save mapping
    mapping = {
        "original_data_dir": str(data_dir.resolve()),
        "clean_data_dir": str(clean_dir.resolve()),
        "seed": seed,
        "removed_counts": removed,
        "clean_counts": {k: len(v) for k, v in clean_indices.items()},
    }
    with open(output_dir / "clean_split_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2, default=str)

    print(f"  Clean splits saved: {clean_dir}")
    print(f"  Mapping saved: {output_dir / 'clean_split_mapping.json'}")
    return clean_dir


def main():
    parser = argparse.ArgumentParser(description="Data leakage audit for OptoGPT 60° s-pol")
    parser.add_argument("--data_dir", type=str, default="../data_60deg_s",
                        help="Directory with train/dev/test pickle files")
    parser.add_argument("--output_dir", type=str, default="audit_results",
                        help="Directory for audit reports")
    parser.add_argument("--spec_threshold", type=float, default=1e-4,
                        help="MAE threshold for near-duplicate spectra")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--create_clean", action="store_true", default=False,
                        help="Create clean splits if leakage detected")
    args = parser.parse_args()

    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Load pretrained index_dict for optional ID-based normalization
    index_dict = None
    pretrained_path = PROJECT_ROOT.parent / "model" / "optogpt.pt"
    if pretrained_path.exists():
        import torch
        ckpt = torch.load(str(pretrained_path), map_location="cpu", weights_only=False)
        cfg = ckpt["configs"]
        if hasattr(cfg, "struc_index_dict"):
            index_dict = cfg.struc_index_dict

    report = audit_data(data_dir, output_dir, args.spec_threshold, index_dict, args.seed)

    # Create clean splits if needed
    total_cross = sum(v["overlap_count"] for v in report["cross_duplicates"].values())
    if total_cross > 0 and args.create_clean:
        print("\n>>> Creating clean splits (original data preserved)...")
        create_clean_splits(data_dir, output_dir, report, args.seed)

    return report


if __name__ == "__main__":
    main()
