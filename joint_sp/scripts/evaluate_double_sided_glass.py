"""Evaluate identical coatings on both sides of a finite glass substrate."""

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
from tmm import inc_tmm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from joint_sp.constants import (  # noqa: E402
    SUBSTRATE,
    SUBSTRATE_THICK_NM,
    THETA_DEG,
    WAVELENGTHS_NM,
)
from joint_sp.scripts.evaluate_freestanding_film import (  # noqa: E402
    NK_DATABASE,
    parse_layers,
)
from optogpt.core.datasets.sim import load_materials  # noqa: E402


def build_double_sided_stack(front_layers, substrate_nk, substrate_thickness_nm):
    """Build air/front/glass/reversed-front/air for identical physical coatings."""
    materials = [material for material, _ in front_layers]
    thicknesses = [thickness for _, thickness in front_layers]
    n_list = [1.0, *materials, substrate_nk, *reversed(materials), 1.0]
    d_list = [
        np.inf,
        *thicknesses,
        substrate_thickness_nm,
        *reversed(thicknesses),
        np.inf,
    ]
    c_list = ["i", *(["c"] * len(front_layers)), "i", *(["c"] * len(front_layers)), "i"]
    return n_list, d_list, c_list


def simulate_double_sided(front_layers, angle_deg, wavelengths_nm, nk_dict, substrate_thickness_nm):
    """Return whole-sample power R/T/A with an incoherent glass substrate."""
    results = {pol: {key: [] for key in ("R", "T", "A")} for pol in ("s", "p")}
    theta_rad = np.deg2rad(angle_deg)
    for index, wavelength_nm in enumerate(wavelengths_nm):
        indexed_layers = [
            (nk_dict[material][index], thickness) for material, thickness in front_layers
        ]
        n_list, d_list, c_list = build_double_sided_stack(
            indexed_layers, nk_dict[SUBSTRATE][index], substrate_thickness_nm
        )
        for pol in ("s", "p"):
            value = inc_tmm(pol, n_list, d_list, c_list, theta_rad, wavelength_nm)
            reflectance = float(value["R"])
            transmittance = float(value["T"])
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
        metrics[f"p05_T{pol}"] = float(np.percentile(results[pol]["T"], 5))
        metrics[f"p95_R{pol}"] = float(np.percentile(results[pol]["R"], 95))
        metrics[f"max_energy_error_{pol}"] = float(
            np.max(np.abs(results[pol]["R"] + results[pol]["T"] + results[pol]["A"] - 1.0))
        )
    return metrics


def write_spectrum_csv(path, wavelengths_nm, results):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength_nm", "Rs", "Ts", "As", "Rp", "Tp", "Ap"])
        for index, wavelength_nm in enumerate(wavelengths_nm):
            writer.writerow(
                [
                    float(wavelength_nm),
                    *[float(results["s"][key][index]) for key in ("R", "T", "A")],
                    *[float(results["p"][key][index]) for key in ("R", "T", "A")],
                ]
            )


def plot_spectrum(path, wavelengths_nm, results, title):
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for axis, pol in zip(axes, ("s", "p")):
        for key in ("R", "T", "A"):
            axis.plot(wavelengths_nm, results[pol][key], label=f"{key}{pol}", linewidth=2)
        axis.set_ylabel("Power fraction")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(ncol=3)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Wavelength (nm)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="TMM for identical coatings on the front and back of finite glass"
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="One-side air-to-glass order, for example MgF2:130,MgO:210",
    )
    parser.add_argument("--angle", type=float, default=THETA_DEG)
    parser.add_argument("--substrate-thickness", type=float, default=SUBSTRATE_THICK_NM)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.angle < 90:
        raise ValueError("Incidence angle must be in [0, 90) degrees")
    if not np.isfinite(args.substrate_thickness) or args.substrate_thickness <= 0:
        raise ValueError("Substrate thickness must be finite and positive")

    front_layers = parse_layers(args.layers)
    wavelengths_nm = np.asarray(WAVELENGTHS_NM, dtype=np.float64)
    nk_dict = load_materials(
        all_mats=sorted({SUBSTRATE, *[material for material, _ in front_layers]}),
        wavelengths=wavelengths_nm / 1000.0,
        DATABASE=str(NK_DATABASE),
    )
    results = simulate_double_sided(
        front_layers, args.angle, wavelengths_nm, nk_dict, args.substrate_thickness
    )
    metrics = summarize(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (_PROJECT_ROOT / "results" / "double_sided_glass" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_spectrum_csv(output_dir / "spectrum.csv", wavelengths_nm, results)
    label = " / ".join(f"{material} {thickness:g} nm" for material, thickness in front_layers)
    plot_spectrum(
        output_dir / "spectrum.png",
        wavelengths_nm,
        results,
        f"Double-sided {label} on {args.substrate_thickness / 1e6:g} mm glass at {args.angle:g} deg",
    )
    report = {
        "simulation_contract": "air -> front coating -> incoherent finite glass -> mirrored back coating -> air",
        "coherence_note": "coating layers are coherent; glass substrate is incoherent",
        "amplitude_note": "whole-sample complex r/t are not reported for an incoherent substrate; R/T/A are power fractions",
        "one_side_layers_air_to_glass": [
            {"material": material, "thickness_nm": thickness}
            for material, thickness in front_layers
        ],
        "full_stack_left_to_right": [
            "air",
            *[f"{material}_{thickness:g}" for material, thickness in front_layers],
            f"{SUBSTRATE}_{args.substrate_thickness:g}",
            *[f"{material}_{thickness:g}" for material, thickness in reversed(front_layers)],
            "air",
        ],
        "angle_deg": args.angle,
        "substrate": SUBSTRATE,
        "substrate_thickness_nm": args.substrate_thickness,
        "wavelengths_nm": {"start": 400, "stop": 1100, "step": 10, "count": 71},
        "tmm": "tmm.inc_tmm",
        "metrics": metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output_dir": str(output_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()
