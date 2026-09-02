param(
    [string]$ProjectRoot = "D:\hrqu\optogpt_project\optogpt",
    [string]$SourceDir = "D:\hrqu\optogpt_project\optogpt\data_60deg_sp_joint_500k_v1",
    [string]$OutDir = "D:\hrqu\optogpt_project\optogpt\data_60deg_sp_joint_500k_v2",
    [int]$TargetUnique = 500000,
    [int]$Workers = 8,
    [int]$ChunkSize = 5000,
    [int]$Seed = 20260724,
    [int]$VerifySamples = 200,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$Python = if ($env:CONDA_PREFIX) {
    Join-Path $env:CONDA_PREFIX "python.exe"
} else {
    $null
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python)) {
    throw "Activate optogpt-joint-sp before running this script."
}
if (-not (Test-Path -LiteralPath $SourceDir)) {
    throw "Missing verified v1 joint dataset: $SourceDir"
}
if ($TargetUnique -ne 500000) {
    throw "This production script requires TargetUnique=500000."
}
if ($Workers -le 0 -or $ChunkSize -le 0 -or $VerifySamples -le 0) {
    throw "Workers, ChunkSize and VerifySamples must be positive."
}

$DriveName = [IO.Path]::GetPathRoot($OutDir).Substring(0, 1)
$Drive = Get-PSDrive -Name $DriveName
if ($Drive.Free -lt 20GB) {
    throw "At least 20 GB free space is required on the output drive."
}

if ($Resume) {
    if (-not (Test-Path -LiteralPath $OutDir)) {
        throw "Resume requested, but output directory does not exist: $OutDir"
    }
} elseif (Test-Path -LiteralPath $OutDir) {
    throw "Refusing to overwrite existing output directory: $OutDir"
}

$env:EXTEND_PROJECT_ROOT = $ProjectRoot
$env:EXTEND_SOURCE_DIR = $SourceDir
$env:EXTEND_OUT_DIR = $OutDir
$env:EXTEND_TARGET_UNIQUE = [string]$TargetUnique
$env:EXTEND_WORKERS = [string]$Workers
$env:EXTEND_CHUNK_SIZE = [string]$ChunkSize
$env:EXTEND_SEED = [string]$Seed
$env:EXTEND_VERIFY_SAMPLES = [string]$VerifySamples
$env:EXTEND_RESUME = if ($Resume) { "1" } else { "0" }

Write-Host "Extending verified joint data to 500000 unique structures"
Write-Host "Source:     $SourceDir"
Write-Host "Output:     $OutDir"
Write-Host "Workers:    $Workers"
Write-Host "Chunk size: $ChunkSize"
Write-Host "Resume:     $Resume"

@'
import hashlib
import hmac
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from joint_sp.constants import (
    ALLOWED_MATERIALS,
    BRANCH_DIM,
    SPEC_DIM,
    SUBSTRATE,
    SUBSTRATE_THICK_NM,
    THETA_DEG,
    WAVELENGTHS_NM,
    WAVELENGTHS_UM,
    validate_disk_structure_tokens,
    validate_joint_spectrum,
)
from joint_sp.io_utils import atomic_json_dump, atomic_pickle_dump
from optogpt.core.datasets.sim import load_materials, spectrum


project_root = Path(os.environ["EXTEND_PROJECT_ROOT"])
source_dir = Path(os.environ["EXTEND_SOURCE_DIR"])
out_dir = Path(os.environ["EXTEND_OUT_DIR"])
target_unique = int(os.environ["EXTEND_TARGET_UNIQUE"])
workers = int(os.environ["EXTEND_WORKERS"])
chunk_size = int(os.environ["EXTEND_CHUNK_SIZE"])
seed = int(os.environ["EXTEND_SEED"])
verify_samples = int(os.environ["EXTEND_VERIFY_SAMPLES"])
resume = os.environ["EXTEND_RESUME"] == "1"
progress_dir = out_dir / ".build_progress"


def structure_hash(tokens):
    return hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def source_fingerprint(data_dir):
    paths = [data_dir / "BUILD_COMPLETE.json", data_dir / "generation_config.json"]
    for split in ("train", "dev", "test"):
        paths.extend([
            data_dir / f"Structure_{split}.pkl",
            data_dir / f"Spectrum_{split}.pkl",
        ])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete v1 source dataset: {missing}")
    values = {
        str(path.resolve()): {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }
    digest = hashlib.sha256(
        json.dumps(values, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return values, digest


def validate_spectrum_array(values, context):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != SPEC_DIM:
        raise ValueError(f"{context}: expected (N, {SPEC_DIM}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{context}: contains NaN or Inf")
    for offset, pol in ((0, "s"), (BRANCH_DIM, "p")):
        reflectance = values[:, offset:offset + 71]
        transmittance = values[:, offset + 71:offset + BRANCH_DIM]
        if reflectance.min() < -1e-5 or transmittance.min() < -1e-5:
            raise ValueError(f"{context}: negative {pol}-polarization R/T")
        if (reflectance.max() > 1.0001 or transmittance.max() > 1.0001 or
                np.max(reflectance + transmittance) > 1.0005):
            raise ValueError(f"{context}: invalid {pol}-polarization energy bound")
    return values


def load_source_hashes():
    with (source_dir / "BUILD_COMPLETE.json").open(encoding="utf-8") as handle:
        complete = json.load(handle)
    with (source_dir / "generation_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if complete.get("status") != "complete":
        raise RuntimeError("v1 BUILD_COMPLETE.json does not report complete")
    if config.get("spec_layout") != ["Rs", "Ts", "Rp", "Tp"]:
        raise RuntimeError("v1 spectrum layout is not [Rs, Ts, Rp, Tp]")
    if config.get("spec_dim") != SPEC_DIM or config.get("theta_deg") != THETA_DEG:
        raise RuntimeError("v1 dimension or angle contract mismatch")

    split_hashes = {}
    seen = set()
    total = 0
    for split in ("train", "dev", "test"):
        with (source_dir / f"Structure_{split}.pkl").open("rb") as handle:
            structures = pickle.load(handle)
        with (source_dir / f"Spectrum_{split}.pkl").open("rb") as handle:
            spectra = validate_spectrum_array(
                pickle.load(handle), f"v1 Spectrum_{split}.pkl"
            )
        if len(structures) != len(spectra):
            raise RuntimeError(f"v1 {split} structure/spectrum count mismatch")
        hashes = set()
        for tokens in structures:
            clean = list(validate_disk_structure_tokens(
                tokens, allowed_materials=ALLOWED_MATERIALS
            ))
            key = structure_hash(clean)
            if key in hashes:
                raise RuntimeError(f"Duplicate structure inside v1 {split}")
            hashes.add(key)
        split_hashes[split] = hashes
        if seen & hashes:
            raise RuntimeError(f"Structure leakage detected in v1 {split}")
        seen.update(hashes)
        total += len(hashes)
        print(f"v1 {split}: spectra={spectra.shape}, unique_structures={len(hashes)}")

    if total != 252007:
        raise RuntimeError(f"Expected the verified v1 source to contain 252007 samples, got {total}")
    if complete.get("n_samples") != total or config.get("joint_sample_count") != total:
        raise RuntimeError("v1 metadata sample count does not match data")
    return config, seen, total


def checkpoint_word_dict():
    checkpoint_path = project_root / "model" / "optogpt.pt"
    # optogpt.pt was serialized when the package was imported as `core`.
    legacy_alias_needed = "core" not in sys.modules
    if legacy_alias_needed:
        import optogpt.core as core_package
        sys.modules["core"] = core_package
        for subpackage in ("models", "datasets", "trains"):
            legacy_name = f"core.{subpackage}"
            try:
                sys.modules[legacy_name] = __import__(
                    f"optogpt.core.{subpackage}", fromlist=[subpackage]
                )
            except ImportError:
                pass
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    finally:
        if legacy_alias_needed:
            for module_name in list(sys.modules):
                if module_name == "core" or module_name.startswith("core."):
                    del sys.modules[module_name]
    configs = checkpoint["configs"]
    word_dict = configs.get("struc_word_dict") if isinstance(configs, dict) else configs.struc_word_dict
    if not word_dict:
        raise RuntimeError("Base checkpoint is missing struc_word_dict")
    choices = {material: [] for material in ALLOWED_MATERIALS}
    for token in word_dict:
        if token in ("UNK", "PAD", "BOS", "EOS") or "_" not in token:
            continue
        material, thickness_text = token.rsplit("_", 1)
        if material in choices and thickness_text.isdigit() and int(thickness_text) > 0:
            choices[material].append(int(thickness_text))
    for material in ALLOWED_MATERIALS:
        choices[material] = sorted(set(choices[material]))
        if not choices[material]:
            raise RuntimeError(f"No valid checkpoint thickness tokens for {material}")
    return word_dict, choices


source_config, source_hashes, source_count = load_source_hashes()
fingerprint, fingerprint_digest = source_fingerprint(source_dir)
word_dict, thickness_choices = checkpoint_word_dict()
needed = target_unique - source_count
if needed <= 0:
    raise RuntimeError(f"Target {target_unique} must exceed source count {source_count}")
print(f"v1_unique={source_count}, new_unique_required={needed}")

if resume:
    if not progress_dir.is_dir():
        raise FileNotFoundError(f"Missing resume progress directory: {progress_dir}")
    if (out_dir / "BUILD_COMPLETE.json").exists():
        raise RuntimeError("v2 is already complete; refusing to resume or overwrite it")
else:
    out_dir.mkdir(parents=True, exist_ok=False)
    progress_dir.mkdir(parents=False, exist_ok=False)
    atomic_json_dump({
        "status": "in_progress",
        "source_dir": str(source_dir.resolve()),
        "source_count": source_count,
        "target_unique": target_unique,
        "new_unique_required": needed,
        "seed": seed,
        "chunk_size": chunk_size,
        "source_fingerprint_digest": fingerprint_digest,
        "started_at": datetime.now().isoformat(),
    }, out_dir / "BUILD_STATUS.json")


seen = set(source_hashes)
generated_count = 0
rng = np.random.default_rng(seed)
chunk_files = sorted(progress_dir.glob("chunk_*.pkl"))
for chunk_path in chunk_files:
    with chunk_path.open("rb") as handle:
        payload = pickle.load(handle)
    expected_name = f"chunk_{generated_count:012d}.pkl"
    if chunk_path.name != expected_name:
        raise RuntimeError(
            f"Non-contiguous resume journal: expected {expected_name}, got {chunk_path.name}"
        )
    if payload.get("format_version") != 1:
        raise RuntimeError(f"Unsupported progress format in {chunk_path}")
    if payload.get("start_index") != generated_count:
        raise RuntimeError(f"Progress start index mismatch in {chunk_path}")
    if payload.get("source_fingerprint_digest") != fingerprint_digest:
        raise RuntimeError("v1 source changed since v2 generation started")
    if (payload.get("target_unique") != target_unique or
            payload.get("seed") != seed or payload.get("chunk_size") != chunk_size):
        raise RuntimeError("Resume arguments differ from the original v2 build")
    results = payload.get("results", [])
    if not results:
        raise RuntimeError(f"Empty progress chunk: {chunk_path}")
    for key, tokens, joint in results:
        clean = list(validate_disk_structure_tokens(
            tokens, word_dict=word_dict, allowed_materials=ALLOWED_MATERIALS
        ))
        if structure_hash(clean) != key or key in seen:
            raise RuntimeError(f"Duplicate or corrupt structure in {chunk_path}")
        validate_joint_spectrum(joint, context=str(chunk_path))
        seen.add(key)
    generated_count += len(results)
    rng.bit_generator.state = payload["rng_state_after"]

if generated_count > needed:
    raise RuntimeError(f"Resume journal has too many samples: {generated_count} > {needed}")
if generated_count:
    print(f"RESUME_OK: completed_new_unique={generated_count}, remaining={needed - generated_count}")


nk_dict = load_materials(
    all_mats=[SUBSTRATE] + ALLOWED_MATERIALS,
    wavelengths=WAVELENGTHS_UM,
    DATABASE=str(project_root / "optogpt" / "nk"),
)
materials_array = np.asarray(ALLOWED_MATERIALS)


def random_unique_candidates(count):
    candidates = []
    pending = set()
    while len(candidates) < count:
        # One-layer space is already exhausted in v1; use 2..20 layers.
        layer_count = int(rng.integers(2, 21))
        tokens = []
        for _ in range(layer_count):
            material = str(rng.choice(materials_array))
            thickness = int(rng.choice(thickness_choices[material]))
            tokens.append(f"{material}_{thickness}")
        key = structure_hash(tokens)
        if key in seen or key in pending:
            continue
        pending.add(key)
        candidates.append((key, tokens))
    return candidates


def compute_joint(candidate):
    key, tokens = candidate
    clean = list(validate_disk_structure_tokens(
        tokens, word_dict=word_dict, allowed_materials=ALLOWED_MATERIALS
    ))
    layer_materials, thicknesses = [], []
    for token in clean:
        material, thickness_text = token.rsplit("_", 1)
        layer_materials.append(material)
        thicknesses.append(int(thickness_text))
    branches = []
    for pol in ("s", "p"):
        branch = np.asarray(spectrum(
            layer_materials,
            thicknesses,
            pol=pol,
            theta=THETA_DEG,
            wavelengths=WAVELENGTHS_UM,
            nk_dict=nk_dict,
            substrate=SUBSTRATE,
            substrate_thick=SUBSTRATE_THICK_NM,
        ), dtype=np.float32)
        if branch.shape != (BRANCH_DIM,):
            raise RuntimeError(f"TMM returned {branch.shape} for pol={pol}, structure={key}")
        branches.append(branch)
    joint = np.concatenate(branches).astype(np.float32)
    validate_joint_spectrum(joint, context=f"generated structure {key}")
    return key, clean, joint


with ThreadPoolExecutor(max_workers=workers) as executor:
    progress = tqdm(total=needed, initial=generated_count, desc="New joint structures")
    while generated_count < needed:
        current_size = min(chunk_size, needed - generated_count)
        candidates = random_unique_candidates(current_size)
        results = list(executor.map(compute_joint, candidates))
        if len(results) != current_size:
            raise RuntimeError("TMM worker result count mismatch")
        payload = {
            "format_version": 1,
            "start_index": generated_count,
            "results": results,
            "rng_state_after": rng.bit_generator.state,
            "source_fingerprint_digest": fingerprint_digest,
            "target_unique": target_unique,
            "seed": seed,
            "chunk_size": chunk_size,
            "saved_at": datetime.now().isoformat(),
        }
        chunk_path = progress_dir / f"chunk_{generated_count:012d}.pkl"
        if chunk_path.exists():
            raise RuntimeError(f"Refusing to overwrite progress chunk: {chunk_path}")
        atomic_pickle_dump(payload, chunk_path)
        for key, _tokens, _joint in results:
            if key in seen:
                raise RuntimeError(f"Generated duplicate structure after TMM: {key}")
            seen.add(key)
        generated_count += len(results)
        progress.update(len(results))
    progress.close()

if generated_count != needed or len(seen) != target_unique:
    raise RuntimeError(
        f"Generation count mismatch: generated={generated_count}, total_unique={len(seen)}"
    )


all_structures = [None] * target_unique
all_hashes = [None] * target_unique
all_specs = np.empty((target_unique, SPEC_DIM), dtype=np.float32)
offset = 0
for split in ("train", "dev", "test"):
    with (source_dir / f"Structure_{split}.pkl").open("rb") as handle:
        structures = pickle.load(handle)
    with (source_dir / f"Spectrum_{split}.pkl").open("rb") as handle:
        spectra = np.asarray(pickle.load(handle), dtype=np.float32)
    stop = offset + len(structures)
    all_structures[offset:stop] = structures
    all_specs[offset:stop] = spectra
    all_hashes[offset:stop] = [structure_hash(tokens) for tokens in structures]
    offset = stop

for chunk_path in sorted(progress_dir.glob("chunk_*.pkl")):
    with chunk_path.open("rb") as handle:
        results = pickle.load(handle)["results"]
    stop = offset + len(results)
    for index, (key, tokens, joint) in enumerate(results, start=offset):
        all_hashes[index] = key
        all_structures[index] = tokens
        all_specs[index] = joint
    offset = stop

if offset != target_unique or len(set(all_hashes)) != target_unique:
    raise RuntimeError(f"Final assembly is not exactly {target_unique} unique structures")
validate_spectrum_array(all_specs, "assembled v2 spectra")


def recompute_joint(index):
    _key, _tokens, computed = compute_joint((all_hashes[index], all_structures[index]))
    error = float(np.max(np.abs(computed - all_specs[index])))
    return index, error


verify_rng = np.random.default_rng(seed ^ 0x5A17)
verify_indices = verify_rng.choice(
    target_unique, size=min(verify_samples, target_unique), replace=False
).tolist()
with ThreadPoolExecutor(max_workers=workers) as executor:
    verification_results = list(tqdm(
        executor.map(recompute_joint, verify_indices),
        total=len(verify_indices),
        desc="Full s+p TMM verification",
    ))
max_tmm_error = max(error for _index, error in verification_results)
if max_tmm_error > 1e-3:
    raise RuntimeError(f"Full TMM verification failed: max error={max_tmm_error}")


seed_bytes = seed.to_bytes(8, "big", signed=True)
buckets = np.asarray([
    int(hmac.new(seed_bytes, key.encode("ascii"), "sha256").hexdigest(), 16) % 1_000_000
    for key in all_hashes
])
masks = {
    "train": buckets < 800000,
    "dev": (buckets >= 800000) & (buckets < 900000),
    "test": buckets >= 900000,
}
split_hash_sets = {
    split: {all_hashes[index] for index in np.flatnonzero(mask)}
    for split, mask in masks.items()
}
if any(not values for values in split_hash_sets.values()):
    raise RuntimeError("At least one v2 split is empty")
if (split_hash_sets["train"] & split_hash_sets["dev"] or
        split_hash_sets["train"] & split_hash_sets["test"] or
        split_hash_sets["dev"] & split_hash_sets["test"]):
    raise RuntimeError("Structure-level leakage detected in v2 split")

manifest = {}
for split, mask in masks.items():
    indices = np.flatnonzero(mask)
    split_structures = [all_structures[index] for index in indices]
    split_specs = all_specs[indices]
    split_hashes = [all_hashes[index] for index in indices]
    atomic_pickle_dump(split_structures, out_dir / f"Structure_{split}.pkl")
    atomic_pickle_dump(split_specs, out_dir / f"Spectrum_{split}.pkl")
    atomic_pickle_dump(split_hashes, out_dir / f"SampleID_{split}.pkl")
    manifest[split] = {"hashes": split_hashes, "count": len(split_hashes)}
    print(f"v2 {split}: spectra={split_specs.shape}, structures={len(split_structures)}")

config = {
    "format_version": 2,
    "description": "500k unique dielectric joint s+p spectra at 60 degrees",
    "spec_layout": ["Rs", "Ts", "Rp", "Tp"],
    "branch_dim": BRANCH_DIM,
    "spec_dim": SPEC_DIM,
    "theta_deg": THETA_DEG,
    "wavelengths_nm": WAVELENGTHS_NM.tolist(),
    "wavelengths_um": WAVELENGTHS_UM.tolist(),
    "substrate": SUBSTRATE,
    "substrate_thick_nm": SUBSTRATE_THICK_NM,
    "allowed_materials": ALLOWED_MATERIALS,
    "joint_sample_count": target_unique,
    "source_directory": str(source_dir.resolve()),
    "source_unique_count": source_count,
    "new_unique_count": needed,
    "new_structure_layer_range": [2, 20],
    "source_fingerprint": fingerprint,
    "source_fingerprint_digest": fingerprint_digest,
    "generator_seed": seed,
    "split_seed": seed,
    "split_ratios": [0.8, 0.1, 0.1],
    "tmm_verify_samples": len(verification_results),
    "max_full_tmm_error": max_tmm_error,
    "created_at": datetime.now().isoformat(),
}
audit = {
    "joint_sample_count": target_unique,
    "unique_structure_count": len(set(all_hashes)),
    "invalid_structures": 0,
    "nonfinite_spectra": 0,
    "leakage_count": 0,
    "tmm_verify_samples": len(verification_results),
    "max_full_tmm_error": max_tmm_error,
}
atomic_json_dump(config, out_dir / "generation_config.json")
atomic_json_dump(manifest, out_dir / "split_manifest.json")
atomic_json_dump(audit, out_dir / "audit_report.json")
atomic_json_dump({
    "status": "complete",
    "source_count": source_count,
    "new_unique_count": needed,
    "target_unique": target_unique,
    "completed_at": datetime.now().isoformat(),
}, out_dir / "BUILD_STATUS.json")
atomic_json_dump({
    "status": "complete",
    "n_samples": target_unique,
    "created_at": datetime.now().isoformat(),
}, out_dir / "BUILD_COMPLETE.json")

print(f"max_full_TMM_error={max_tmm_error:.9g}")
print("materials=dielectric-only")
print("structure_leakage=CLEAN")
print("EXTENDED_FORMAL_JOINT_DATA_OK")
'@ | & $Python -B -

if ($LASTEXITCODE -ne 0) {
    if (Test-Path -LiteralPath $OutDir) {
        Write-Host "Build stopped. Preserve the output directory and resume with:"
        Write-Host "  & .\extend_formal_joint_data.ps1 -Resume"
    }
    throw "Formal v2 joint data build failed. Do not train."
}

& $Python -B .\joint_sp\scripts\windows_preflight.py --data_dir $OutDir
if ($LASTEXITCODE -ne 0) {
    throw "Formal v2 preflight failed. Do not train."
}

Write-Host "FORMAL_V2_PREFLIGHT_AND_TMM_OK"
