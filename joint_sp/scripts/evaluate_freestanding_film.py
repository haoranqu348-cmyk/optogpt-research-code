"""Evaluate a freestanding coherent film stack with air on both sides."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tmm import coh_tmm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from joint_sp.constants import ALLOWED_MATERIALS, THETA_DEG, WAVELENGTHS_NM  # noqa: E402
from optogpt.core.datasets.sim import load_materials  # noqa: E402

NK_DATABASE = _PROJECT_ROOT / "optogpt" / "nk"


def parse_layers(text):
    """Parse air-side-to-exit-side layers written as Material:thickness_nm."""
    layers = []
    for token in text.split(","):
        try:
            material, thickness = token.strip().rsplit(":", 1)
            thickness = float(thickness)
        except ValueError as exc:
            raise ValueError(
                "Layers must use Material:thickness_nm, separated by commas"
            ) from exc
        if material not in ALLOWED_MATERIALS:
            raise ValueError(f"Material {material!r} is not in the allowed dielectric set")
        if not np.isfinite(thickness) or thickness <= 0:
            raise ValueError("Every layer thickness must be finite and positive")
        layers.append((material, thickness))
    if not layers:
        raise ValueError("At least one coating layer is required")
    return layers


def simulate_freestanding(layers, angle_deg, wavelengths_nm, nk_dict):
    """Return complex amplitudes and power R/T/A for air -> layers -> air."""
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
    theta_rad = np.deg2rad(angle_deg)
    thicknesses = [np.inf, *[value for _, value in layers], np.inf]
    results = {pol: {key: [] for key in ("r", "t", "R", "T", "A")} for pol in ("s", "p")}

    for index, wavelength_nm in enumerate(wavelengths_nm):
        n_list = [1.0, *[nk_dict[material][index] for material, _ in layers], 1.0]
        for pol in ("s", "p"):
            value = coh_tmm(pol, n_list, thicknesses, theta_rad, wavelength_nm)
            reflectance = float(value["R"])
            transmittance = float(value["T"])
            results[pol]["r"].append(complex(value["r"]))
            results[pol]["t"].append(complex(value["t"]))
            results[pol]["R"].append(reflectance)
            results[pol]["T"].append(transmittance)
            results[pol]["A"].append(1.0 - reflectance - transmittance)

    return {
        pol: {key: np.asarray(values) for key, values in pol_result.items()}
        for pol, pol_result in results.items()
    }


def summarize(results):
    metrics = {}
    for pol in ("s", "p"):
        for key in ("R", "T", "A"):
            values = results[pol][key]
            metrics[f"mean_{key}{pol}"] = float(np.mean(values))
            metrics[f"min_{key}{pol}"] = float(np.min(values))
            metrics[f"max_{key}{pol}"] = float(np.max(values))
        metrics[f"max_energy_error_{pol}"] = float(
            np.max(np.abs(results[pol]["R"] + results[pol]["T"] + results[pol]["A"] - 1.0))
        )
    return metrics


def write_spectrum_csv(path, wavelengths_nm, results):
    fields = ["wavelength_nm"]
    for pol in ("s", "p"):
        fields.extend(
            [f"r{pol}_real", f"r{pol}_imag", f"t{pol}_real", f"t{pol}_imag", f"R{pol}", f"T{pol}", f"A{pol}"]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, wavelength_nm in enumerate(wavelengths_nm):
            row = {"wavelength_nm": float(wavelength_nm)}
            for pol in ("s", "p"):
                row.update(
                    {
                        f"r{pol}_real": float(results[pol]["r"][index].real),
                        f"r{pol}_imag": float(results[pol]["r"][index].imag),
                        f"t{pol}_real": float(results[pol]["t"][index].real),
                        f"t{pol}_imag": float(results[pol]["t"][index].imag),
                        f"R{pol}": float(results[pol]["R"][index]),
                        f"T{pol}": float(results[pol]["T"][index]),
                        f"A{pol}": float(results[pol]["A"][index]),
                    }
                )
            writer.writerow(row)


def plot_spectrum(path, wavelengths_nm, results, title):
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for axis, pol in zip(axes, ("s", "p")):
        axis.plot(wavelengths_nm, results[pol]["R"], label=f"R{pol}", linewidth=2)
        axis.plot(wavelengths_nm, results[pol]["T"], label=f"T{pol}", linewidth=2)
        axis.plot(wavelengths_nm, results[pol]["A"], label=f"A{pol}", linewidth=1.5)
        axis.set_ylabel("Power fraction")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(ncol=3)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Wavelength (nm)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Coherent TMM for a freestanding film stack: air -> layers -> air"
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Air-side order, for example MgF2:100,Al2O3:100",
    )
    parser.add_argument("--angle", type=float, default=THETA_DEG)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0 <= args.angle < 90:
        raise ValueError("Incidence angle must be in [0, 90) degrees")
    layers = parse_layers(args.layers)
    wavelengths_nm = np.asarray(WAVELENGTHS_NM, dtype=np.float64)
    wavelengths_um = wavelengths_nm / 1000.0
    nk_dict = load_materials(
        all_mats=sorted({material for material, _ in layers}),
        wavelengths=wavelengths_um,
        DATABASE=str(NK_DATABASE),
    )
    results = simulate_freestanding(layers, args.angle, wavelengths_nm, nk_dict)
    metrics = summarize(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (_PROJECT_ROOT / "results" / "freestanding_film" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_spectrum_csv(output_dir / "spectrum.csv", wavelengths_nm, results)
    layer_label = " / ".join(f"{material} {thickness:g} nm" for material, thickness in layers)
    plot_spectrum(
        output_dir / "spectrum.png",
        wavelengths_nm,
        results,
        f"Air / {layer_label} / Air at {args.angle:g} deg",
    )
    report = {
        "simulation_contract": "air -> coherent coating layers -> air",
        "amplitude_note": "lowercase r/t are complex field amplitudes; uppercase R/T/A are power fractions",
        "layer_order": "air side to exit side",
        "layers": [
            {"material": material, "thickness_nm": thickness}
            for material, thickness in layers
        ],
        "angle_deg": args.angle,
        "wavelengths_nm": {
            "start": float(wavelengths_nm[0]),
            "stop": float(wavelengths_nm[-1]),
            "step": float(wavelengths_nm[1] - wavelengths_nm[0]),
            "count": int(len(wavelengths_nm)),
        },
        "tmm": "tmm.coh_tmm",
        "incident_index": 1.0,
        "exit_index": 1.0,
        "metrics": metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output_dir": str(output_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()
