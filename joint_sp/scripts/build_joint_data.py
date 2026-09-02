"""Build verified 284-dimensional joint s+p data from single-pol sources."""

import argparse
import hashlib
import hmac
import json
import pickle
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_JOINT_SP = _SCRIPT_DIR.parent
_PKG_ROOT = _JOINT_SP.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from optogpt.core.datasets.sim import load_materials, spectrum
from joint_sp.constants import (
    ALLOWED_MATERIALS,
    BANNED_MATERIALS,
    BRANCH_DIM,
    SPEC_DIM,
    SUBSTRATE,
    SUBSTRATE_THICK_NM,
    THETA_DEG,
    WAVELENGTHS_NM,
    WAVELENGTHS_UM,
    validate_disk_structure_tokens,
)
from joint_sp.io_utils import atomic_json_dump, atomic_pickle_dump

DEFAULT_WL = WAVELENGTHS_UM
SOURCE_SPLITS = ("train", "dev", "test")


def structure_hash(tokens):
    return hashlib.sha256("|".join(tokens).encode()).hexdigest()


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _source_fingerprint(data_dir):
    data_dir = Path(data_dir)
    files = [data_dir / "generation_config.json"]
    for split in SOURCE_SPLITS:
        files.extend([
            data_dir / f"Structure_{split}.pkl",
            data_dir / f"Spectrum_{split}.pkl",
        ])
    present = [path for path in files if path.exists()]
    return {
        str(path.resolve()): {"size": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in present
    }


def validate_source_metadata(data_dir, expected_pol, theta):
    path = Path(data_dir) / "generation_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing source metadata: {path}. Refusing to infer polarization from a directory name."
        )
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)

    actual_pol = str(config.get("polarization", "")).lower()
    if actual_pol != expected_pol:
        raise ValueError(f"{path}: polarization={actual_pol!r}, expected {expected_pol!r}")
    if float(config.get("theta_deg", float("nan"))) != float(theta):
        raise ValueError(f"{path}: theta_deg does not match requested {theta}")
    if int(config.get("spectrum_dim", -1)) != BRANCH_DIM:
        raise ValueError(f"{path}: spectrum_dim must be {BRANCH_DIM}")
    if int(config.get("n_wavelengths", -1)) != len(WAVELENGTHS_NM):
        raise ValueError(f"{path}: n_wavelengths must be {len(WAVELENGTHS_NM)}")
    if not np.array_equal(np.asarray(config.get("wavelengths_nm")), WAVELENGTHS_NM):
        raise ValueError(f"{path}: wavelength grid must be 400..1100 nm in 10 nm steps")
    if config.get("substrate") != SUBSTRATE:
        raise ValueError(f"{path}: substrate must be {SUBSTRATE}")
    if int(config.get("substrate_thick_nm", -1)) != SUBSTRATE_THICK_NM:
        raise ValueError(f"{path}: substrate thickness must be {SUBSTRATE_THICK_NM} nm")
    if set(config.get("materials", [])) != set(ALLOWED_MATERIALS):
        raise ValueError(f"{path}: material set does not match joint_sp constants")
    return config


def _validate_single_pol_spectrum(value, context):
    spec = np.asarray(value, dtype=np.float32)
    if spec.shape != (BRANCH_DIM,):
        raise ValueError(f"{context}: spectrum shape {spec.shape}, expected ({BRANCH_DIM},)")
    if not np.all(np.isfinite(spec)):
        raise ValueError(f"{context}: spectrum contains NaN/Inf")
    r, t = spec[: len(WAVELENGTHS_NM)], spec[len(WAVELENGTHS_NM):]
    if np.min(r) < -1e-5 or np.min(t) < -1e-5:
        raise ValueError(f"{context}: negative R/T value")
    if np.max(r) > 1.0001 or np.max(t) > 1.0001 or np.max(r + t) > 1.0005:
        raise ValueError(f"{context}: non-physical R/T energy bound")
    return spec


def load_data_dir(data_dir):
    data_dir = Path(data_dir)
    structures, spectra, sample_ids = [], [], []
    found = 0
    for split in SOURCE_SPLITS:
        struct_file = data_dir / f"Structure_{split}.pkl"
        spec_file = data_dir / f"Spectrum_{split}.pkl"
        if struct_file.exists() != spec_file.exists():
            raise FileNotFoundError(f"Incomplete {split} pair in {data_dir}")
        if not struct_file.exists():
            continue
        found += 1
        with open(struct_file, "rb") as handle:
            split_structs = pickle.load(handle)
        with open(spec_file, "rb") as handle:
            split_specs = np.asarray(pickle.load(handle))
        if split_specs.ndim != 2 or split_specs.shape[1] != BRANCH_DIM:
            raise ValueError(f"{spec_file}: shape {split_specs.shape}, expected (N, {BRANCH_DIM})")
        if len(split_structs) != len(split_specs):
            raise ValueError(
                f"{split}: {len(split_structs)} structures != {len(split_specs)} spectra"
            )
        for index, (tokens, spec) in enumerate(zip(split_structs, split_specs)):
            clean = list(validate_disk_structure_tokens(tokens, allowed_materials=ALLOWED_MATERIALS))
            structures.append(clean)
            spectra.append(_validate_single_pol_spectrum(spec, f"{spec_file}[{index}]"))
            sample_ids.append((split, index))
    if found == 0:
        raise FileNotFoundError(f"No Structure/Spectrum split pairs found in {data_dir}")
    return structures, np.asarray(spectra, dtype=np.float32), sample_ids


def parse_structure(tokens):
    clean = validate_disk_structure_tokens(tokens, allowed_materials=ALLOWED_MATERIALS)
    materials, thicknesses = [], []
    for token in clean:
        material, thickness = token.rsplit("_", 1)
        materials.append(material)
        thicknesses.append(int(thickness))
    return materials, thicknesses


def tmm_compute_polarization(tokens, pol, nk_dict, theta):
    materials, thicknesses = parse_structure(tokens)
    result = spectrum(
        materials,
        thicknesses,
        pol=pol,
        theta=theta,
        wavelengths=DEFAULT_WL,
        nk_dict=nk_dict,
        substrate=SUBSTRATE,
        substrate_thick=SUBSTRATE_THICK_NM,
    )
    return _validate_single_pol_spectrum(result, f"TMM pol={pol} tokens={tokens}")


def _register(registry, tokens, spec, pol, sample_id, tolerance):
    key = structure_hash(tokens)
    entry = registry.setdefault(key, {"tokens": tokens, "s_spec": None, "p_spec": None})
    field = f"{pol}_spec"
    previous = entry[field]
    if previous is not None and not np.allclose(previous, spec, atol=tolerance, rtol=0):
        diff = float(np.max(np.abs(previous - spec)))
        raise ValueError(
            f"Conflicting duplicate {pol}-spectrum for structure {key}: max_diff={diff}, "
            f"latest_sample={sample_id}"
        )
    entry[field] = spec


def _checkpoint_payload(work_digest, work_count, next_index, fingerprint, args):
    return {
        "format_version": 1,
        "work_digest": work_digest,
        "work_count": work_count,
        "next_index": next_index,
        "source_fingerprint": fingerprint,
        "theta": args.theta,
        "seed": args.seed,
        "chunk_size": args.chunk_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Build verified joint s+p 284-dim data")
    parser.add_argument("--s_dir", required=True)
    parser.add_argument("--p_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--theta", type=float, default=THETA_DEG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument("--chunk_size", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify_samples", type=int, default=100)
    parser.add_argument("--verify_tolerance", type=float, default=1e-3)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ratios = np.asarray(args.split, dtype=float)
    if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Split ratios must be positive and sum to 1")
    if args.chunk_size <= 0 or args.num_workers <= 0 or args.verify_samples <= 0:
        raise ValueError("chunk_size, num_workers and verify_samples must be positive")
    if float(args.theta) != float(THETA_DEG):
        raise ValueError(f"Joint contract requires theta={THETA_DEG} degrees")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = out_dir / "BUILD_COMPLETE.json"
    if complete_marker.exists():
        if args.resume:
            print(f"Dataset already complete: {complete_marker}")
            return
        if not args.overwrite:
            raise FileExistsError(
                f"Completed dataset already exists at {out_dir}; pass --overwrite to rebuild"
            )
        complete_marker.unlink()
    atomic_json_dump(
        {"status": "in_progress", "started_at": datetime.now().isoformat()},
        out_dir / "BUILD_IN_PROGRESS.json",
    )
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / ".build_checkpoint.pkl"
    chunk_dir = Path(f"{checkpoint_path}.chunks")

    source_configs = {
        "s": validate_source_metadata(args.s_dir, "s", args.theta),
        "p": validate_source_metadata(args.p_dir, "p", args.theta),
    }
    fingerprint = {"s": _source_fingerprint(args.s_dir), "p": _source_fingerprint(args.p_dir)}
    nk_dict = load_materials(
        all_mats=[SUBSTRATE] + ALLOWED_MATERIALS + list(BANNED_MATERIALS),
        wavelengths=DEFAULT_WL,
        DATABASE=str(_PKG_ROOT / "optogpt" / "nk"),
    )

    s_structs, s_specs, s_ids = load_data_dir(args.s_dir)
    p_structs, p_specs, p_ids = load_data_dir(args.p_dir)
    registry = {}
    for tokens, spec, sample_id in zip(s_structs, s_specs, s_ids):
        _register(registry, tokens, spec, "s", sample_id, args.verify_tolerance)
    for tokens, spec, sample_id in zip(p_structs, p_specs, p_ids):
        _register(registry, tokens, spec, "p", sample_id, args.verify_tolerance)

    hashes = sorted(registry)
    rng = np.random.RandomState(args.seed)
    verify_hashes = [hashes[i] for i in rng.choice(
        len(hashes), min(args.verify_samples, len(hashes)), replace=False
    )]
    verification = {"s_checked": 0, "p_checked": 0, "s_mismatches": 0,
                    "p_mismatches": 0, "tmm_failures": 0}
    for key in tqdm(verify_hashes, desc="Verifying source spectra"):
        entry = registry[key]
        for pol in ("s", "p"):
            known = entry[f"{pol}_spec"]
            if known is None:
                continue
            verification[f"{pol}_checked"] += 1
            try:
                computed = tmm_compute_polarization(entry["tokens"], pol, nk_dict, args.theta)
            except Exception:
                verification["tmm_failures"] += 1
                continue
            if float(np.max(np.abs(computed - known))) > args.verify_tolerance:
                verification[f"{pol}_mismatches"] += 1
    if verification["tmm_failures"] or verification["s_mismatches"] or verification["p_mismatches"]:
        atomic_json_dump(verification, out_dir / "audit_report.json")
        raise RuntimeError(f"Source TMM verification failed: {verification}")

    missing = [key for key in hashes if registry[key]["s_spec"] is None or registry[key]["p_spec"] is None]
    work_digest = hashlib.sha256("|".join(missing).encode()).hexdigest()
    start_index = 0
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        with open(checkpoint_path, "rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint.get("source_fingerprint") != fingerprint:
            raise RuntimeError("Source data changed since build checkpoint")
        if checkpoint.get("theta") != args.theta or checkpoint.get("seed") != args.seed:
            raise RuntimeError("Build arguments changed since checkpoint")
        if checkpoint.get("chunk_size") != args.chunk_size:
            raise RuntimeError("chunk_size changed since build checkpoint")
        if (checkpoint.get("work_digest") != work_digest or
                checkpoint.get("work_count") != len(missing)):
            raise RuntimeError("Missing-polarization work queue changed since checkpoint")
        start_index = int(checkpoint["next_index"])
        for chunk_start in range(0, start_index, args.chunk_size):
            chunk_path = chunk_dir / f"chunk_{chunk_start:012d}.pkl"
            if not chunk_path.exists():
                raise RuntimeError(f"Missing completed chunk file: {chunk_path}")
            with open(chunk_path, "rb") as handle:
                chunk_results = pickle.load(handle)
            for key, values in chunk_results.items():
                for pol, value in values.items():
                    registry[key][f"{pol}_spec"] = value

    def compute_missing(key):
        entry = registry[key]
        result = {}
        for pol in ("s", "p"):
            if entry[f"{pol}_spec"] is None:
                result[pol] = tmm_compute_polarization(entry["tokens"], pol, nk_dict, args.theta)
        return key, result

    for chunk_start in tqdm(range(start_index, len(missing), args.chunk_size), desc="TMM chunks"):
        chunk = missing[chunk_start:chunk_start + args.chunk_size]
        if args.num_workers == 1:
            computed = map(compute_missing, chunk)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                computed = executor.map(compute_missing, chunk)
        chunk_results = {}
        for key, values in computed:
            chunk_results[key] = values
            for pol, value in values.items():
                registry[key][f"{pol}_spec"] = value
        atomic_pickle_dump(chunk_results, chunk_dir / f"chunk_{chunk_start:012d}.pkl")
        atomic_pickle_dump(
            _checkpoint_payload(work_digest, len(missing), chunk_start + len(chunk), fingerprint, args),
            checkpoint_path,
        )

    all_hashes, all_structures, all_specs = [], [], []
    for key in sorted(registry):
        entry = registry[key]
        if entry["s_spec"] is None or entry["p_spec"] is None:
            raise RuntimeError(f"Missing polarization after build for {key}")
        joint = np.concatenate([entry["s_spec"], entry["p_spec"]]).astype(np.float32)
        if joint.shape != (SPEC_DIM,) or not np.all(np.isfinite(joint)):
            raise ValueError(f"Invalid joint spectrum for {key}: shape={joint.shape}")
        all_hashes.append(key)
        all_structures.append(entry["tokens"])
        all_specs.append(joint)
    all_hashes = np.asarray(all_hashes)
    all_specs = np.asarray(all_specs, dtype=np.float32)
    if not len(all_hashes):
        raise RuntimeError("No valid joint samples were built")

    seed_bytes = args.seed.to_bytes(8, "big", signed=True)
    buckets = np.asarray([
        int(hmac.new(seed_bytes, key.encode(), "sha256").hexdigest(), 16) % 1_000_000
        for key in all_hashes
    ])
    train_end = int(ratios[0] * 1_000_000)
    dev_end = int((ratios[0] + ratios[1]) * 1_000_000)
    masks = {
        "train": buckets < train_end,
        "dev": (buckets >= train_end) & (buckets < dev_end),
        "test": buckets >= dev_end,
    }
    split_hashes = {name: set(all_hashes[mask]) for name, mask in masks.items()}
    if any(not values for values in split_hashes.values()):
        raise RuntimeError("At least one output split is empty")
    if (split_hashes["train"] & split_hashes["dev"] or
            split_hashes["train"] & split_hashes["test"] or
            split_hashes["dev"] & split_hashes["test"]):
        raise RuntimeError("Structure hash leakage detected between output splits")

    for name, mask in masks.items():
        atomic_pickle_dump([all_structures[i] for i in np.flatnonzero(mask)],
                           out_dir / f"Structure_{name}.pkl")
        atomic_pickle_dump(all_specs[mask], out_dir / f"Spectrum_{name}.pkl")
        atomic_pickle_dump(all_hashes[mask].tolist(), out_dir / f"SampleID_{name}.pkl")

    config = {
        "format_version": 1,
        "spec_layout": ["Rs", "Ts", "Rp", "Tp"],
        "branch_dim": BRANCH_DIM,
        "spec_dim": SPEC_DIM,
        "theta_deg": args.theta,
        "wavelengths_nm": WAVELENGTHS_NM.tolist(),
        "wavelengths_um": WAVELENGTHS_UM.tolist(),
        "substrate": SUBSTRATE,
        "substrate_thick_nm": SUBSTRATE_THICK_NM,
        "allowed_materials": ALLOWED_MATERIALS,
        "source_metadata": source_configs,
        "source_fingerprint": fingerprint,
        "joint_sample_count": len(all_hashes),
        "split_seed": args.seed,
        "split_ratios": ratios.tolist(),
        "created_at": datetime.now().isoformat(),
    }
    manifest = {
        name: {"hashes": all_hashes[mask].tolist(), "count": int(mask.sum())}
        for name, mask in masks.items()
    }
    audit = {
        **verification,
        "invalid_structures": 0,
        "nonfinite_spectra": 0,
        "leakage_count": 0,
        "joint_sample_count": len(all_hashes),
    }
    atomic_json_dump(config, out_dir / "generation_config.json")
    atomic_json_dump(manifest, out_dir / "split_manifest.json")
    atomic_json_dump(audit, out_dir / "audit_report.json")
    atomic_json_dump(
        {"status": "complete", "n_samples": len(all_hashes),
         "created_at": datetime.now().isoformat()},
        complete_marker,
    )
    in_progress = out_dir / "BUILD_IN_PROGRESS.json"
    if in_progress.exists():
        in_progress.unlink()
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    print(f"Joint data complete: train={masks['train'].sum()}, dev={masks['dev'].sum()}, test={masks['test'].sum()}")


if __name__ == "__main__":
    main()
