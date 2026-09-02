param(
    [string]$ProjectRoot = "D:\hrqu\optogpt_project\optogpt",
    [string]$SDir = "D:\hrqu\optogpt_project\optogpt\data_60deg_s_500k_dielectric",
    [string]$PAnchorDir = "D:\hrqu\optogpt_project\optogpt\joint_sp\build_sources\p_anchor_500k_v1",
    [string]$OutDir = "D:\hrqu\optogpt_project\optogpt\data_60deg_sp_joint_500k_v1",
    [int]$Workers = 8,
    [int]$ChunkSize = 5000,
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
if (-not (Test-Path -LiteralPath $SDir)) {
    throw "Missing s-polarization source: $SDir"
}
if ($Workers -le 0 -or $ChunkSize -le 0) {
    throw "Workers and ChunkSize must be positive."
}

$Drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($OutDir).Substring(0, 1))
if ($Drive.Free -lt 20GB) {
    throw "At least 20 GB free space is required on the output drive."
}

if (-not $Resume) {
    foreach ($Path in @($PAnchorDir, $OutDir)) {
        if (Test-Path -LiteralPath $Path) {
            throw "Refusing to overwrite existing path: $Path"
        }
    }

    $env:FORMAL_S_DIR = $SDir
    $env:FORMAL_P_ANCHOR_DIR = $PAnchorDir

    @'
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from joint_sp.constants import (
    ALLOWED_MATERIALS,
    BRANCH_DIM,
    SUBSTRATE,
    SUBSTRATE_THICK_NM,
    THETA_DEG,
    WAVELENGTHS_NM,
    WAVELENGTHS_UM,
    validate_disk_structure_tokens,
)
from joint_sp.io_utils import atomic_json_dump, atomic_pickle_dump
from optogpt.core.datasets.sim import load_materials, spectrum

s_dir = Path(os.environ["FORMAL_S_DIR"])
p_dir = Path(os.environ["FORMAL_P_ANCHOR_DIR"])
config = json.loads((s_dir / "generation_config.json").read_text(encoding="utf-8"))

assert config.get("polarization") == "s"
assert config.get("num_samples_target") == 500000
assert config.get("theta_deg") == THETA_DEG == 60
assert config.get("spectrum_dim") == BRANCH_DIM == 142
assert config.get("n_wavelengths") == len(WAVELENGTHS_NM) == 71
assert config.get("substrate") == SUBSTRATE
assert config.get("substrate_thick_nm") == SUBSTRATE_THICK_NM
assert set(config.get("materials", [])) == set(ALLOWED_MATERIALS)

with (s_dir / "Structure_train.pkl").open("rb") as handle:
    source_structures = pickle.load(handle)

tokens = list(validate_disk_structure_tokens(
    source_structures[0], allowed_materials=ALLOWED_MATERIALS
))
materials, thicknesses = [], []
for token in tokens:
    material, thickness = token.rsplit("_", 1)
    materials.append(material)
    thicknesses.append(int(thickness))

nk_dict = load_materials(
    all_mats=[SUBSTRATE] + ALLOWED_MATERIALS,
    wavelengths=WAVELENGTHS_UM,
    DATABASE=str(Path.cwd() / "optogpt" / "nk"),
)
p_spec = np.asarray(spectrum(
    materials,
    thicknesses,
    pol="p",
    theta=THETA_DEG,
    wavelengths=WAVELENGTHS_UM,
    nk_dict=nk_dict,
    substrate=SUBSTRATE,
    substrate_thick=SUBSTRATE_THICK_NM,
), dtype=np.float32)

assert p_spec.shape == (BRANCH_DIM,)
assert np.isfinite(p_spec).all()
assert p_spec.min() >= -1e-5
reflectance, transmittance = p_spec[:71], p_spec[71:]
assert np.max(reflectance + transmittance) <= 1.0005

p_config = dict(config)
p_config.update({
    "description": "p-polarization anchor for verified 500k joint build",
    "polarization": "p",
    "num_samples_target": 1,
    "generated_at": datetime.now().isoformat(),
    "source_s_directory": str(s_dir.resolve()),
})

atomic_pickle_dump([tokens], p_dir / "Structure_train.pkl")
atomic_pickle_dump(p_spec.reshape(1, -1), p_dir / "Spectrum_train.pkl")
atomic_json_dump(p_config, p_dir / "generation_config.json")
print(f"P_ANCHOR_OK: {p_dir}")
'@ | & $Python -B -

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create p-polarization anchor."
    }
}

$BuildArguments = @(
    "-B"
    ".\joint_sp\scripts\build_joint_data.py"
    "--s_dir", $SDir
    "--p_dir", $PAnchorDir
    "--out_dir", $OutDir
    "--theta", "60"
    "--seed", "42"
    "--split", "0.8", "0.1", "0.1"
    "--chunk_size", "$ChunkSize"
    "--num_workers", "$Workers"
    "--verify_samples", "100"
    "--verify_tolerance", "0.001"
)
if ($Resume) {
    $BuildArguments += "--resume"
}

Write-Host "Starting verified 500k joint build"
Write-Host "S source: $SDir"
Write-Host "P anchor: $PAnchorDir"
Write-Host "Output:   $OutDir"
Write-Host "Workers:  $Workers"

& $Python @BuildArguments
if ($LASTEXITCODE -ne 0) {
    throw "Formal joint data build failed. Keep the directory and rerun with -Resume."
}

& $Python -B .\joint_sp\scripts\windows_preflight.py --data_dir $OutDir
if ($LASTEXITCODE -ne 0) {
    throw "Formal data preflight failed. Do not train."
}

$env:FORMAL_JOINT_DIR = $OutDir
@'
import json
import os
import pickle
from pathlib import Path

import numpy as np

data_dir = Path(os.environ["FORMAL_JOINT_DIR"])
config = json.loads((data_dir / "generation_config.json").read_text(encoding="utf-8"))
complete = json.loads((data_dir / "BUILD_COMPLETE.json").read_text(encoding="utf-8"))

assert complete.get("status") == "complete"
assert config.get("spec_layout") == ["Rs", "Ts", "Rp", "Tp"]
assert config.get("spec_dim") == 284
assert config.get("theta_deg") == 60

sets = {}
total = 0
for split in ("train", "dev", "test"):
    with (data_dir / f"Spectrum_{split}.pkl").open("rb") as handle:
        spectra = np.asarray(pickle.load(handle))
    with (data_dir / f"Structure_{split}.pkl").open("rb") as handle:
        structures = pickle.load(handle)
    assert spectra.ndim == 2 and spectra.shape[1] == 284
    assert len(structures) == len(spectra)
    assert np.isfinite(spectra).all()
    sets[split] = {tuple(tokens) for tokens in structures}
    total += len(structures)
    print(f"{split}: spectra={spectra.shape}, structures={len(structures)}")

assert not sets["train"] & sets["dev"]
assert not sets["train"] & sets["test"]
assert not sets["dev"] & sets["test"]
assert total == 500000, f"Expected exactly 500000 unique structures, got {total}"
print(f"total_samples={total}")
print("FORMAL_JOINT_DATA_OK")
'@ | & $Python -B -

if ($LASTEXITCODE -ne 0) {
    throw "Formal data contract check failed. Do not train."
}
