"""Independent mixed-coherence TMM truth labels and objective metrics."""

from typing import Dict

import numpy as np
from tmm import coh_tmm, inc_tmm

from .config import DoubleSidedConfig
from .contract import DoubleSidedStructure


METRIC_KEYS = (
    "mean_Rs", "mean_Rp", "p95_Rs", "p95_Rp", "max_Rs", "max_Rp",
    "mean_Ts", "mean_Tp", "mean_As", "mean_Ap",
)
OBJECTIVE_WEIGHTS = {
    "mean_Rs": 0.30, "mean_Rp": 0.25, "p95_Rs": 0.15,
    "p95_Rp": 0.15, "max_Rs": 0.10, "max_Rp": 0.05,
}


def _complex_at(nk_dict, material, index):
    values = np.asarray(nk_dict[material], dtype=np.complex128)
    return complex(values[index])


def simulate_abc(structure: DoubleSidedStructure, nk_dict, config=None, require_truth_grid=True):
    """Return A/B/C labels, with C always computed by tmm.inc_tmm."""
    config = (config or DoubleSidedConfig()).validate(require_truth_grid=require_truth_grid)
    wavelengths = np.asarray(config.wavelengths_nm, dtype=float)
    theta = np.deg2rad(config.angle_deg)
    front, back = structure.front, structure.back
    output = {
        definition: {pol: {key: [] for key in ("R", "T", "A")} for pol in ("s", "p")}
        for definition in ("A", "B", "C")
    }

    for index, wavelength in enumerate(wavelengths):
        substrate_n = _complex_at(nk_dict, config.substrate, index)
        front_n = [_complex_at(nk_dict, layer.material, index) for layer in front]
        back_n = [_complex_at(nk_dict, layer.material, index) for layer in back]

        stacks = {
            "A": (
                [1.0, *front_n, substrate_n],
                [np.inf, *[layer.thickness_nm for layer in front], np.inf],
                None,
            ),
            "B": (
                [1.0, *front_n, substrate_n, 1.0],
                [np.inf, *[layer.thickness_nm for layer in front],
                 config.substrate_thickness_nm, np.inf],
                ["i", *(["c"] * len(front)), "i", "i"],
            ),
            "C": (
                [1.0, *front_n, substrate_n, *back_n, 1.0],
                [np.inf, *[layer.thickness_nm for layer in front],
                 config.substrate_thickness_nm,
                 *[layer.thickness_nm for layer in back], np.inf],
                ["i", *(["c"] * len(front)), "i", *(["c"] * len(back)), "i"],
            ),
        }
        for definition, (n_list, d_list, coherence) in stacks.items():
            for pol in ("s", "p"):
                result = (coh_tmm(pol, n_list, d_list, theta, wavelength)
                          if definition == "A"
                          else inc_tmm(pol, n_list, d_list, coherence, theta, wavelength))
                reflectance, transmittance = float(result["R"]), float(result["T"])
                output[definition][pol]["R"].append(reflectance)
                output[definition][pol]["T"].append(transmittance)
                output[definition][pol]["A"].append(1.0 - reflectance - transmittance)

    for definition in output:
        for pol in output[definition]:
            for key in output[definition][pol]:
                output[definition][pol][key] = np.asarray(
                    output[definition][pol][key], dtype=np.float64
                )
    return output


def simulate_c(structure: DoubleSidedStructure, nk_dict, config=None, require_truth_grid=True):
    """Compute only the final finite-glass two-sided truth with inc_tmm."""
    config = (config or DoubleSidedConfig()).validate(require_truth_grid=require_truth_grid)
    wavelengths = np.asarray(config.wavelengths_nm, dtype=float)
    theta = np.deg2rad(config.angle_deg)
    output = {pol: {key: [] for key in ("R", "T", "A")} for pol in ("s", "p")}
    for index, wavelength in enumerate(wavelengths):
        front_n = [_complex_at(nk_dict, layer.material, index) for layer in structure.front]
        back_n = [_complex_at(nk_dict, layer.material, index) for layer in structure.back]
        n_list = [
            1.0, *front_n, _complex_at(nk_dict, config.substrate, index), *back_n, 1.0
        ]
        d_list = [
            np.inf, *[layer.thickness_nm for layer in structure.front],
            config.substrate_thickness_nm,
            *[layer.thickness_nm for layer in structure.back], np.inf,
        ]
        coherence = [
            "i", *(["c"] * len(structure.front)), "i",
            *(["c"] * len(structure.back)), "i",
        ]
        for pol in ("s", "p"):
            result = inc_tmm(pol, n_list, d_list, coherence, theta, wavelength)
            reflectance, transmittance = float(result["R"]), float(result["T"])
            output[pol]["R"].append(reflectance)
            output[pol]["T"].append(transmittance)
            output[pol]["A"].append(1.0 - reflectance - transmittance)
    return {
        pol: {key: np.asarray(values, dtype=np.float64) for key, values in data.items()}
        for pol, data in output.items()
    }


def spectrum_vector(result):
    return np.concatenate([
        result["s"]["R"], result["s"]["T"],
        result["p"]["R"], result["p"]["T"],
    ]).astype(np.float32)


def summarize(result):
    metrics: Dict[str, float] = {}
    for pol in ("s", "p"):
        reflectance = np.asarray(result[pol]["R"])
        transmittance = np.asarray(result[pol]["T"])
        absorption = np.asarray(result[pol]["A"])
        metrics.update({
            f"mean_R{pol}": float(np.mean(reflectance)),
            f"p95_R{pol}": float(np.percentile(reflectance, 95)),
            f"max_R{pol}": float(np.max(reflectance)),
            f"mean_T{pol}": float(np.mean(transmittance)),
            f"mean_A{pol}": float(np.mean(absorption)),
        })
    metrics["objective"] = sum(
        OBJECTIVE_WEIGHTS[key] * metrics[key] for key in OBJECTIVE_WEIGHTS
    )
    metrics["objective_as_written"] = (
        0.30 * metrics["mean_Rs"] - 0.25 * metrics["mean_Rp"]
        - 0.15 * metrics["p95_Rs"] - 0.15 * metrics["p95_Rp"]
        - 0.10 * metrics["max_Rs"] - 0.05 * metrics["max_Rp"]
    )
    metrics["passes_strict"] = bool(
        metrics["mean_Rs"] <= 0.02 and metrics["mean_Rp"] <= 0.02
        and metrics["p95_Rs"] <= 0.05 and metrics["p95_Rp"] <= 0.05
        and metrics["mean_Ts"] >= 0.90 and metrics["mean_Tp"] >= 0.90
        and metrics["mean_As"] <= 0.02 and metrics["mean_Ap"] <= 0.02
    )
    return metrics


def verify_merge_equivalence(structure, nk_dict, config=None, atol=2e-10):
    before = simulate_abc(structure, nk_dict, config)["C"]
    after = simulate_abc(structure.merged(), nk_dict, config)["C"]
    differences = {
        f"{key}{pol}": float(np.max(np.abs(before[pol][key] - after[pol][key])))
        for pol in ("s", "p") for key in ("R", "T", "A")
    }
    maximum = max(differences.values())
    if maximum > atol:
        raise RuntimeError(f"Adjacent-layer merge changed the spectrum: max_diff={maximum}")
    return {"equivalent": True, "atol": atol, "max_abs_difference": maximum,
            "components": differences}
