"""Build exact-grid film n/k CSV files from approved raw RII YAML sources."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tabulated_nk(document):
    blocks = [item for item in document["DATA"] if item["type"] == "tabulated nk"]
    if len(blocks) != 1:
        raise ValueError("Approved sources must contain exactly one tabulated nk block")
    rows = np.asarray([
        [float(value) for value in line.split()]
        for line in blocks[0]["data"].strip().splitlines()
    ])
    if rows.ndim != 2 or rows.shape[1] != 3 or np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError("Expected increasing wavelength_um,n,k rows")
    return rows


def interpolate_source(source_path, output_path, wavelengths_nm):
    with Path(source_path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    rows = tabulated_nk(document)
    wavelengths_um = np.asarray(wavelengths_nm, dtype=float) / 1000.0
    if wavelengths_um[0] < rows[0, 0] or wavelengths_um[-1] > rows[-1, 0]:
        raise ValueError(
            f"Refusing extrapolation: source covers {rows[0,0]}..{rows[-1,0]} um"
        )
    n = np.interp(wavelengths_um, rows[:, 0], rows[:, 1])
    k = np.interp(wavelengths_um, rows[:, 0], rows[:, 2])
    if np.any(n <= 0) or np.any(k < 0) or not np.all(np.isfinite(n + k)):
        raise ValueError("Interpolated optical constants are non-physical")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(["nm", "n", "k", "wl"])
        for nm, nr, ki in zip(wavelengths_nm, n, k):
            writer.writerow([f"{nm:.0f}", f"{nr:.12g}", f"{ki:.12g}", f"{nm/1000:.3f}"])
    return {
        "source": str(Path(source_path)), "source_sha256": sha256(source_path),
        "output": str(output_path), "output_sha256": sha256(output_path),
        "n_range": [float(n.min()), float(n.max())],
        "k_range": [float(k.min()), float(k.max())],
        "max_k_400_1100": float(k.max()), "interpolation": "linear_no_extrapolation",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--materials-dir", default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    materials_dir = Path(args.materials_dir)
    with (materials_dir / "material_audit.yaml").open(encoding="utf-8") as handle:
        audit = yaml.safe_load(handle)
    wavelengths = np.arange(400.0, 1101.0, 10.0)
    manifest = {"wavelengths_nm": wavelengths.tolist(), "materials": {}}
    for material, record in audit["materials"].items():
        if record["gate"] != "approved_physics_search":
            continue
        raw_file = record.get("raw_file")
        if not raw_file:
            raise ValueError(f"Approved material {material} has no raw_file")
        manifest["materials"][material] = interpolate_source(
            materials_dir / raw_file, Path(args.output_dir) / f"{material}.csv", wavelengths
        )
    output = Path(args.output_dir) / "nk_manifest.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(output)


if __name__ == "__main__":
    main()
