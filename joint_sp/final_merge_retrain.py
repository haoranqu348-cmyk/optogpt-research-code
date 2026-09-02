"""Merge verified joint data with TMM-revalidated self-improvement samples."""

import argparse
import hashlib
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from optogpt.core.datasets.sim import load_materials
from joint_sp.constants import (
    ALLOWED_MATERIALS,
    BANNED_MATERIALS,
    SPEC_DIM,
    SUBSTRATE,
    THETA_DEG,
    WAVELENGTHS_UM,
    validate_disk_structure_tokens,
)
from joint_sp.decoder import tmm_rerank_joint
from joint_sp.io_utils import atomic_json_dump, atomic_pickle_dump


def structure_hash_from_tokens(tokens):
    clean = validate_disk_structure_tokens(tokens, allowed_materials=ALLOWED_MATERIALS)
    return hashlib.sha256("|".join(clean).encode()).hexdigest()


def _load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _load_split(data_dir, name):
    structures = _load_pickle(data_dir / f"Structure_{name}.pkl")
    spectra = np.asarray(_load_pickle(data_dir / f"Spectrum_{name}.pkl"), dtype=np.float32)
    if spectra.ndim != 2 or spectra.shape[1] != SPEC_DIM or len(structures) != len(spectra):
        raise ValueError(f"Invalid {name} split: structs={len(structures)}, spectra={spectra.shape}")
    hashes = [structure_hash_from_tokens(tokens) for tokens in structures]
    return structures, spectra, hashes


def _extract_augmented(aug_data):
    if isinstance(aug_data, dict):
        structures = aug_data.get("perturb_struct", aug_data.get("structures", []))
        spectra = aug_data.get("perturb_spec_joint", aug_data.get("spectra", []))
    elif isinstance(aug_data, (list, tuple)) and len(aug_data) == 2:
        structures, spectra = aug_data
    else:
        raise ValueError("Unsupported augmented data format")
    if len(structures) != len(spectra):
        raise ValueError("Augmented structure/spectrum count mismatch")

    seen, valid_structures, valid_spectra, hashes = set(), [], [], []
    for index, (tokens, spec) in enumerate(zip(structures, spectra)):
        clean = list(validate_disk_structure_tokens(tokens, allowed_materials=ALLOWED_MATERIALS))
        value = np.asarray(spec, dtype=np.float32)
        if value.shape != (SPEC_DIM,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid augmented spectrum at index {index}: {value.shape}")
        digest = structure_hash_from_tokens(clean)
        if digest in seen:
            continue
        seen.add(digest)
        valid_structures.append(clean)
        valid_spectra.append(value)
        hashes.append(digest)
    return valid_structures, valid_spectra, hashes


def main():
    parser = argparse.ArgumentParser(description="Merge original and verified joint augmentation")
    parser.add_argument("--original_dir", required=True)
    parser.add_argument("--si_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--aug_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tmm_tolerance", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.aug_ratio <= 1 or args.num_workers <= 0:
        raise ValueError("aug_ratio must be in [0,1] and num_workers must be positive")

    original_dir = Path(args.original_dir)
    si_dir = Path(args.si_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = out_dir / "MERGE_COMPLETE.json"
    if complete_marker.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Completed merge already exists at {out_dir}; pass --overwrite to rebuild"
            )
        complete_marker.unlink()
    atomic_json_dump(
        {"status": "in_progress", "started_at": datetime.now().isoformat()},
        out_dir / "MERGE_IN_PROGRESS.json",
    )
    if not (original_dir / "BUILD_COMPLETE.json").exists():
        raise RuntimeError("Original dataset has no BUILD_COMPLETE.json")
    if not (si_dir / "SELF_IMPROVING_COMPLETE.json").exists():
        raise RuntimeError("Self-improving output has no completion marker")

    train_structs, train_specs, train_hash_list = _load_split(original_dir, "train")
    dev_structs, dev_specs, dev_hash_list = _load_split(original_dir, "dev")
    test_structs, test_specs, test_hash_list = _load_split(original_dir, "test")
    train_hashes, dev_hashes, test_hashes = map(set, (
        train_hash_list, dev_hash_list, test_hash_list
    ))
    if train_hashes & dev_hashes or train_hashes & test_hashes or dev_hashes & test_hashes:
        raise RuntimeError("Original dataset contains structure-level leakage")

    aug_path = si_dir / "added_data.pkl"
    if not aug_path.exists():
        raise FileNotFoundError(f"Missing augmented data: {aug_path}")
    aug_structs, aug_specs, aug_hash_list = _extract_augmented(_load_pickle(aug_path))
    forbidden = train_hashes | dev_hashes | test_hashes
    safe_indices = [i for i, digest in enumerate(aug_hash_list) if digest not in forbidden]
    removed_leaks = len(aug_hash_list) - len(safe_indices)
    aug_structs = [aug_structs[i] for i in safe_indices]
    aug_specs = [aug_specs[i] for i in safe_indices]
    aug_hash_list = [aug_hash_list[i] for i in safe_indices]

    nk_dict = load_materials(
        all_mats=[SUBSTRATE] + ALLOWED_MATERIALS + list(BANNED_MATERIALS),
        wavelengths=WAVELENGTHS_UM,
        DATABASE=str(_PKG_ROOT / "optogpt" / "nk"),
    )

    def verify(index):
        tokens, target = aug_structs[index], aug_specs[index]
        materials, thicknesses = zip(*(token.rsplit("_", 1) for token in tokens))
        candidate = {
            "tokens": tokens,
            "materials": list(materials),
            "thicknesses": [int(value) for value in thicknesses],
            "n_layers": len(tokens),
        }
        ranked, failures = tmm_rerank_joint(
            [candidate], target, nk_dict, wavelengths=WAVELENGTHS_UM,
            theta=THETA_DEG, objective="joint_error",
        )
        if not ranked or ranked[0]["E_joint"] > args.tmm_tolerance:
            return index, None, failures
        simulated = np.concatenate([
            ranked[0]["sim_Rs"], ranked[0]["sim_Ts"],
            ranked[0]["sim_Rp"], ranked[0]["sim_Tp"],
        ]).astype(np.float32)
        return index, simulated, failures

    verified_structs, verified_specs, verified_hashes = [], [], []
    indices = range(len(aug_structs))
    if args.num_workers == 1:
        verification_results = map(verify, indices)
    else:
        executor = ThreadPoolExecutor(max_workers=args.num_workers)
        verification_results = executor.map(verify, indices)
    try:
        for index, simulated, _failures in tqdm(
                verification_results, total=len(aug_structs), desc="Verifying augmentation"):
            if simulated is None:
                continue
            verified_structs.append(aug_structs[index])
            verified_specs.append(simulated)
            verified_hashes.append(aug_hash_list[index])
    finally:
        if args.num_workers != 1:
            executor.shutdown(wait=True)

    max_aug = int(len(train_structs) * args.aug_ratio)
    if len(verified_structs) > max_aug:
        rng = np.random.RandomState(args.seed)
        chosen = rng.choice(len(verified_structs), max_aug, replace=False)
        verified_structs = [verified_structs[i] for i in chosen]
        verified_specs = [verified_specs[i] for i in chosen]
        verified_hashes = [verified_hashes[i] for i in chosen]

    if verified_specs:
        augmented_array = np.asarray(verified_specs, dtype=np.float32)
        merged_specs = np.concatenate([train_specs, augmented_array], axis=0)
    else:
        merged_specs = train_specs.copy()
    merged_structs = train_structs + verified_structs
    rng = np.random.RandomState(args.seed)
    permutation = rng.permutation(len(merged_structs))
    merged_structs = [merged_structs[i] for i in permutation]
    merged_specs = merged_specs[permutation]

    for name, structures, spectra, hashes in (
        ("train", merged_structs, merged_specs,
         [structure_hash_from_tokens(tokens) for tokens in merged_structs]),
        ("dev", dev_structs, dev_specs, dev_hash_list),
        ("test", test_structs, test_specs, test_hash_list),
    ):
        atomic_pickle_dump(structures, out_dir / f"Structure_{name}.pkl")
        atomic_pickle_dump(spectra, out_dir / f"Spectrum_{name}.pkl")
        atomic_pickle_dump(hashes, out_dir / f"SampleID_{name}.pkl")

    manifest = {
        "original_train_count": len(train_structs),
        "augmented_input": len(aug_hash_list),
        "augmented_removed_leaks_or_overlap": removed_leaks,
        "augmented_tmm_verified": len(verified_structs),
        "final_train_count": len(merged_structs),
        "aug_ratio": args.aug_ratio,
        "tmm_tolerance": args.tmm_tolerance,
        "seed": args.seed,
        "created_at": datetime.now().isoformat(),
    }
    atomic_json_dump(manifest, out_dir / "merge_manifest.json")
    with open(original_dir / "generation_config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    config["merged_augmentation"] = manifest
    atomic_json_dump(config, out_dir / "generation_config.json")
    atomic_json_dump(
        {"status": "complete", "final_train_count": len(merged_structs),
         "created_at": datetime.now().isoformat()},
        complete_marker,
    )
    atomic_json_dump(
        {"status": "complete", "n_samples": len(merged_structs),
         "created_at": datetime.now().isoformat()},
        out_dir / "BUILD_COMPLETE.json",
    )
    in_progress = out_dir / "MERGE_IN_PROGRESS.json"
    if in_progress.exists():
        in_progress.unlink()
    print(f"Merged dataset complete: {len(merged_structs)} train samples")


if __name__ == "__main__":
    main()
