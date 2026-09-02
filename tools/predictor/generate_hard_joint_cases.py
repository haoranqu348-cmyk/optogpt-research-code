"""Generate reproducible difficult 60-degree joint s/p benchmark cases."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from interactive_predictor import WAVELENGTHS_NM, tmm_simulate


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "inputs" / "hard_joint_sp_60cases"
MATERIALS = [
    "Al2O3", "AlN", "HfO2", "MgF2", "MgO",
    "Si3N4", "SiO2", "Ta2O5", "TiO2", "ZnO",
]


def build_structure(rng, index):
    n_layers = 8 + (index % 7)
    materials = []
    while len(materials) < n_layers:
        material = rng.choice(MATERIALS)
        if not materials or material != materials[-1]:
            materials.append(material)

    # Mix chirped, alternating, and perturbed thickness profiles.
    family = index % 4
    if family == 0:
        base = np.linspace(25, 460, n_layers)
        thicknesses = base + rng.choices([-15, 0, 15], k=n_layers)
    elif family == 1:
        thicknesses = [35 + 25 * ((index + j * 3) % 15) for j in range(n_layers)]
    elif family == 2:
        thicknesses = [450 - 25 * ((index + j * 2) % 15) for j in range(n_layers)]
    else:
        thicknesses = [rng.randrange(20, 481, 10) for _ in range(n_layers)]
    thicknesses = [int(min(480, max(20, round(value / 10) * 10))) for value in thicknesses]
    return materials, thicknesses


def write_spectrum(path, columns):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength_nm", "Rs", "Ts", "Rp", "Tp"])
        for row in zip(WAVELENGTHS_NM, *columns):
            writer.writerow([int(row[0]), *[f"{value:.9f}" for value in row[1:]]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.count < 50:
        raise ValueError("困难样本数量至少为 50")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    seen = set()
    for index in range(args.count):
        materials, thicknesses = build_structure(rng, index)
        key = (tuple(materials), tuple(thicknesses))
        if key in seen:
            index -= 1
            continue
        seen.add(key)
        Rs, Ts, _ = tmm_simulate(materials, thicknesses, theta_deg=60, pol="s")
        Rp, Tp, _ = tmm_simulate(materials, thicknesses, theta_deg=60, pol="p")
        case = f"hard_{index + 1:03d}"
        path = output_dir / f"{case}_60deg_sp.csv"
        write_spectrum(path, [Rs, Ts, Rp, Tp])
        rows.append({
            "case": case,
            "file": path.name,
            "materials": "|".join(materials),
            "thicknesses_nm": "|".join(str(value) for value in thicknesses),
            "n_layers": len(materials),
        })

    with (output_dir / "answers.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "file", "materials", "thicknesses_nm", "n_layers"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} difficult joint s+p cases in {output_dir}")


if __name__ == "__main__":
    main()
