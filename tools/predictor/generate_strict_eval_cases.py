"""Generate out-of-grid and noisy joint s/p evaluation cases."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from interactive_predictor import WAVELENGTHS_NM, tmm_simulate


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "inputs" / "strict_joint_sp_eval_20260726"
MATERIALS = [
    "Al2O3", "AlN", "HfO2", "MgF2", "MgO",
    "Si3N4", "SiO2", "Ta2O5", "TiO2", "ZnO",
]


def _continuous_thicknesses(rng, n_layers, family):
    if family == 0:
        values = np.linspace(17.0, 493.0, n_layers)
        values += np.array([rng.uniform(-8.0, 8.0) for _ in range(n_layers)])
    elif family == 1:
        values = [rng.uniform(11.0, 499.0) for _ in range(n_layers)]
    elif family == 2:
        values = [
            (14.0 + 474.0 * j / max(n_layers - 1, 1))
            if j % 2 == 0 else (490.0 - 460.0 * j / max(n_layers - 1, 1))
            for j in range(n_layers)
        ]
    else:
        values = [
            rng.uniform(12.0, 85.0) if j % 2 == 0 else rng.uniform(410.0, 498.0)
            for j in range(n_layers)
        ]
    return [round(float(np.clip(v, 11.0, 499.0)), 3) for v in values]


def _materials(rng, n_layers, family):
    if family == 2:
        low = ["MgF2", "SiO2", "Al2O3", "MgO"]
        high = ["TiO2", "Ta2O5", "HfO2", "ZnO"]
        return [rng.choice(low if i % 2 == 0 else high) for i in range(n_layers)]
    result = []
    for _ in range(n_layers):
        value = rng.choice(MATERIALS)
        if len(result) > 1 and value == result[-1]:
            value = rng.choice([m for m in MATERIALS if m != result[-1]])
        result.append(value)
    return result


def _write_csv(path, columns):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength_nm", "Rs", "Ts", "Rp", "Tp"])
        for row in zip(WAVELENGTHS_NM, *columns):
            writer.writerow([int(row[0]), *[f"{float(v):.9f}" for v in row[1:]]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.count < 30:
        raise ValueError("strict evaluation requires at least 30 cases")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    seen = set()
    for index in range(args.count):
        family = index % 4
        n_layers = 15 + (index % 6)  # deliberately near the 20-layer limit
        materials = _materials(rng, n_layers, family)
        thicknesses = _continuous_thicknesses(rng, n_layers, family)
        key = (tuple(materials), tuple(thicknesses))
        if key in seen:
            continue
        seen.add(key)
        Rs, Ts, _ = tmm_simulate(materials, thicknesses, theta_deg=60, pol="s")
        Rp, Tp, _ = tmm_simulate(materials, thicknesses, theta_deg=60, pol="p")
        noise_level = 0.0
        if index >= int(args.count * 0.8):
            noise_level = 0.0025
            noise = np.random.default_rng(args.seed + index).normal(0, noise_level, (4, len(WAVELENGTHS_NM)))
            Rs, Ts, Rp, Tp = [np.clip(v + noise[i], 0.0, 1.0) for i, v in enumerate((Rs, Ts, Rp, Tp))]
            Ts = np.minimum(Ts, 1.0 - Rs)
            Tp = np.minimum(Tp, 1.0 - Rp)
        case = f"strict_{index + 1:03d}"
        _write_csv(out / f"{case}_60deg_sp.csv", [Rs, Ts, Rp, Tp])
        rows.append({
            "case": case,
            "file": f"{case}_60deg_sp.csv",
            "family": ["chirped_continuous", "random_continuous", "alternating_continuous", "bimodal_continuous"][family],
            "noise_level": noise_level,
            "materials": "|".join(materials),
            "thicknesses_nm": "|".join(f"{v:.3f}" for v in thicknesses),
            "n_layers": n_layers,
        })
    with (out / "answers.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "generation_config.json").write_text(json.dumps({
        "seed": args.seed, "count": len(rows), "theta_deg": 60,
        "materials": MATERIALS, "thickness_domain_nm": [11, 499],
        "thicknesses_are_continuous": True,
        "noise_cases_fraction": 0.2,
        "description": "OOD continuous-thickness joint s+p cases; model vocabulary remains 10 nm-grid tokens",
    }, indent=2))
    print(f"Generated {len(rows)} strict cases in {out}")


if __name__ == "__main__":
    main()
