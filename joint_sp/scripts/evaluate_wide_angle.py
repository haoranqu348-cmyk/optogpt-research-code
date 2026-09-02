"""Evaluate broadband s+p transmission over an incidence-angle range.

The trained joint model is conditioned on a 60-degree spectrum, not on angle.
This script therefore treats it as a candidate generator and uses TMM over the
full angle/wavelength grid as the source of truth.
"""

import argparse
import copy
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_JOINT_SP = _SCRIPT_DIR.parent
_PKG_ROOT = _JOINT_SP.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from joint_sp.constants import (  # noqa: E402
    ALLOWED_MATERIALS,
    BANNED_MATERIALS,
    BRANCH_DIM,
    MAX_LAYERS,
    SUBSTRATE,
    SUBSTRATE_THICK_NM,
    WAVELENGTHS_NM,
    normalize_structure_tokens,
    structure_hash_from_tokens,
)
from joint_sp.decoder import build_joint_logits_mask, generate_candidates_sp  # noqa: E402
from joint_sp.io_utils import atomic_json_dump  # noqa: E402
from joint_sp.model import load_joint_sp_checkpoint  # noqa: E402
DEFAULT_WAVELENGTHS_UM = np.arange(0.4, 1.101, 0.01)


def _get_sim_backend():
    """Load the TMM-backed simulator only when a physical scan is requested."""
    from optogpt.core.datasets.sim import load_materials, spectrum

    return load_materials, spectrum


def parse_angle_grid(text):
    """Parse either start:stop:step or a comma-separated angle list."""
    if ":" in text:
        parts = [float(value) for value in text.split(":")]
        if len(parts) != 3:
            raise ValueError("Angle grid must use start:stop:step")
        start, stop, step = parts
        if step <= 0 or start < 0 or stop >= 90 or start > stop:
            raise ValueError("Angle grid must satisfy 0 <= start <= stop < 90 and step > 0")
        count = int(math.floor((stop - start) / step + 1e-9)) + 1
        values = start + step * np.arange(count, dtype=np.float64)
        if values[-1] < stop - 1e-8:
            values = np.append(values, stop)
    else:
        values = np.asarray([float(value) for value in text.split(",")], dtype=np.float64)
        if values.size == 0 or np.any(values < 0) or np.any(values >= 90):
            raise ValueError("Angles must be in [0, 90)")
        values = np.unique(values)
    return values


def make_broadband_high_t_target():
    reflectance = np.zeros(BRANCH_DIM // 2, dtype=np.float32)
    transmittance = np.ones(BRANCH_DIM // 2, dtype=np.float32)
    return np.concatenate(
        [reflectance, transmittance, reflectance, transmittance]
    ).astype(np.float32)


def _candidate_from_mapping(item):
    materials = list(item.get("materials", item.get("best_materials", [])))
    thicknesses = list(item.get("thicknesses", item.get("best_thicknesses", [])))
    if not materials and item.get("tokens"):
        tokens = normalize_structure_tokens(item["tokens"])
        materials = [token.rsplit("_", 1)[0] for token in tokens]
        thicknesses = [int(token.rsplit("_", 1)[1]) for token in tokens]
    if not materials or len(materials) != len(thicknesses):
        raise ValueError("Each structure needs equal-length materials and thicknesses")
    if len(materials) > MAX_LAYERS or any(mat not in ALLOWED_MATERIALS for mat in materials):
        raise ValueError("Structure violates the pure-dielectric material/layer contract")
    if any(
        not np.isfinite(value)
        or float(value) <= 0
        or not float(value).is_integer()
        for value in thicknesses
    ):
        raise ValueError("Thicknesses must be finite positive integers matching legal tokens")
    tokens = [f"{mat}_{int(value)}" for mat, value in zip(materials, thicknesses)]
    return {
        "tokens": tokens,
        "materials": materials,
        "thicknesses": [int(value) for value in thicknesses],
        "n_layers": len(materials),
        "structure_hash": structure_hash_from_tokens(tokens),
    }


def load_structure_candidates(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ("top_results", "results", "candidates", "structures"):
            if key in payload:
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Structure JSON must contain a structure or a list of structures")
    return deduplicate_candidates([_candidate_from_mapping(item) for item in payload])


def deduplicate_candidates(candidates):
    unique = {}
    for candidate in candidates:
        normalized = _candidate_from_mapping(candidate)
        unique.setdefault(normalized["structure_hash"], normalized)
    return list(unique.values())


def generate_model_candidates(model_path, num_candidates, seeds, device):
    model, word_dict, index_dict, configs = load_joint_sp_checkpoint(
        model_path, device=device
    )
    model.eval()
    mask, _ = build_joint_logits_mask(word_dict, ALLOWED_MATERIALS)
    target = make_broadband_high_t_target()
    generated = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        generated.extend(
            generate_candidates_sp(
                model,
                target,
                word_dict,
                index_dict,
                num_candidates=num_candidates,
                max_len=MAX_LAYERS + 2,
                device=device,
                logits_mask=mask,
            )
        )
    config_keys = (
        "architecture_version", "model_type", "N", "d_model", "d_ff",
        "head_num", "dropout", "spec_dim", "spec_layout", "theta_deg",
        "polarizations", "allowed_materials", "pretrained_sha256",
    )
    report_configs = {key: configs.get(key) for key in config_keys if key in configs}
    return deduplicate_candidates(generated), report_configs


def simulate_angle_grid(candidate, angles, nk_dict, wavelengths=DEFAULT_WAVELENGTHS_UM):
    _load_materials, spectrum_fn = _get_sim_backend()
    matrices = {"Ts": [], "Tp": []}
    for angle in angles:
        for pol, key in (("s", "Ts"), ("p", "Tp")):
            result = spectrum_fn(
                candidate["materials"],
                list(candidate["thicknesses"]),
                pol=pol,
                theta=float(angle),
                wavelengths=wavelengths,
                nk_dict=nk_dict,
                substrate=SUBSTRATE,
                substrate_thick=SUBSTRATE_THICK_NM,
            )
            n_points = len(result) // 2
            transmission = np.asarray(result[n_points:], dtype=np.float64)
            if transmission.shape != (len(wavelengths),) or not np.all(np.isfinite(transmission)):
                raise ValueError(f"Invalid {pol}-polarization TMM result at {angle} degrees")
            matrices[key].append(np.clip(transmission, 0.0, 1.0))
    return {key: np.asarray(value) for key, value in matrices.items()}


def summarize_grid(matrices, angles, mean_threshold, p05_threshold, min_threshold):
    ts = matrices["Ts"]
    tp = matrices["Tp"]
    angle_mean_ts = np.mean(ts, axis=1)
    angle_mean_tp = np.mean(tp, axis=1)
    angle_p05_ts = np.percentile(ts, 5, axis=1)
    angle_p05_tp = np.percentile(tp, 5, axis=1)
    angle_min_ts = np.min(ts, axis=1)
    angle_min_tp = np.min(tp, axis=1)
    angle_pass = (
        (angle_mean_ts >= mean_threshold)
        & (angle_mean_tp >= mean_threshold)
        & (angle_p05_ts >= p05_threshold)
        & (angle_p05_tp >= p05_threshold)
        & (angle_min_ts >= min_threshold)
        & (angle_min_tp >= min_threshold)
    )
    worst_mean_by_angle = np.minimum(angle_mean_ts, angle_mean_tp)
    worst_p05_by_angle = np.minimum(angle_p05_ts, angle_p05_tp)
    worst_min_by_angle = np.minimum(angle_min_ts, angle_min_tp)
    score = (
        0.50 * (1.0 - float(np.min(worst_mean_by_angle)))
        + 0.30 * (1.0 - float(min(np.percentile(ts, 5), np.percentile(tp, 5))))
        + 0.20 * (1.0 - float(min(np.min(ts), np.min(tp))))
    )
    return {
        "mean_Ts": float(np.mean(ts)),
        "mean_Tp": float(np.mean(tp)),
        "p05_Ts": float(np.percentile(ts, 5)),
        "p05_Tp": float(np.percentile(tp, 5)),
        "min_Ts": float(np.min(ts)),
        "min_Tp": float(np.min(tp)),
        "worst_angle_mean_Ts": float(np.min(angle_mean_ts)),
        "worst_angle_mean_Tp": float(np.min(angle_mean_tp)),
        "worst_pol_angle_mean_T": float(np.min(worst_mean_by_angle)),
        "worst_pol_grid_p05_T": float(min(np.percentile(ts, 5), np.percentile(tp, 5))),
        "worst_pol_grid_min_T": float(min(np.min(ts), np.min(tp))),
        "worst_mean_angle_deg": float(angles[int(np.argmin(worst_mean_by_angle))]),
        "worst_p05_angle_deg": float(angles[int(np.argmin(worst_p05_by_angle))]),
        "worst_min_angle_deg": float(angles[int(np.argmin(worst_min_by_angle))]),
        "angle_pass_rate": float(np.mean(angle_pass)),
        "all_angles_pass": bool(np.all(angle_pass)),
        "wide_angle_loss": float(score),
        "per_angle": [
            {
                "angle_deg": float(angle),
                "mean_Ts": float(angle_mean_ts[index]),
                "mean_Tp": float(angle_mean_tp[index]),
                "p05_Ts": float(angle_p05_ts[index]),
                "p05_Tp": float(angle_p05_tp[index]),
                "min_Ts": float(angle_min_ts[index]),
                "min_Tp": float(angle_min_tp[index]),
                "passes": bool(angle_pass[index]),
            }
            for index, angle in enumerate(angles)
        ],
    }


def evaluate_candidate(candidate, angles, nk_dict, thresholds):
    matrices = simulate_angle_grid(candidate, angles, nk_dict)
    metrics = summarize_grid(matrices, angles, **thresholds)
    return {**copy.deepcopy(candidate), **metrics}, matrices


def perturb_candidate_and_nk(candidate, nk_dict, rng, thickness_sigma, bias_sigma, n_sigma):
    perturbed = copy.deepcopy(candidate)
    common_bias = rng.normal(0.0, bias_sigma)
    layer_noise = rng.normal(0.0, thickness_sigma, size=len(candidate["thicknesses"]))
    perturbed["thicknesses"] = (
        np.asarray(candidate["thicknesses"]) * (1.0 + common_bias + layer_noise)
    ).clip(0.1).tolist()

    perturbed_nk = dict(nk_dict)
    for material in set(candidate["materials"]):
        contrast_error = rng.normal(0.0, n_sigma)
        values = np.asarray(nk_dict[material], dtype=np.complex128)
        real = 1.0 + (values.real - 1.0) * (1.0 + contrast_error)
        perturbed_nk[material] = real + 1j * values.imag
    return perturbed, perturbed_nk


def monte_carlo_reliability(
    candidate,
    angles,
    nk_dict,
    thresholds,
    trials,
    seed,
    thickness_sigma,
    bias_sigma,
    n_sigma,
    angle_sigma,
):
    rng = np.random.default_rng(seed)
    trial_rows = []
    failures = 0
    for trial in range(trials):
        perturbed, perturbed_nk = perturb_candidate_and_nk(
            candidate, nk_dict, rng, thickness_sigma, bias_sigma, n_sigma
        )
        angle_offset = rng.normal(0.0, angle_sigma)
        perturbed_angles = np.clip(angles + angle_offset, 0.0, 89.999)
        try:
            matrices = simulate_angle_grid(perturbed, perturbed_angles, perturbed_nk)
            metrics = summarize_grid(matrices, perturbed_angles, **thresholds)
        except Exception:
            failures += 1
            continue
        trial_rows.append(
            {
                "trial": trial,
                "all_angles_pass": metrics["all_angles_pass"],
                "worst_pol_angle_mean_T": metrics["worst_pol_angle_mean_T"],
                "worst_pol_grid_p05_T": metrics["worst_pol_grid_p05_T"],
                "worst_pol_grid_min_T": metrics["worst_pol_grid_min_T"],
            }
        )

    def percentile(key, value):
        return float(np.percentile([row[key] for row in trial_rows], value)) if trial_rows else None

    passing = sum(row["all_angles_pass"] for row in trial_rows)
    return {
        "trials_requested": trials,
        "trials_valid": len(trial_rows),
        "tmm_failures": failures,
        "estimated_pass_probability": passing / trials if trials else None,
        "conditional_pass_probability": passing / len(trial_rows) if trial_rows else None,
        "worst_pol_angle_mean_T_p05": percentile("worst_pol_angle_mean_T", 5),
        "worst_pol_grid_p05_T_p05": percentile("worst_pol_grid_p05_T", 5),
        "worst_pol_grid_min_T_p05": percentile("worst_pol_grid_min_T", 5),
        "uncertainty_model": {
            "independent_layer_thickness_sigma_rel": thickness_sigma,
            "common_thickness_bias_sigma_rel": bias_sigma,
            "material_index_contrast_sigma_rel": n_sigma,
            "incidence_angle_sigma_deg": angle_sigma,
            "distribution": "independent Gaussian draws unless noted",
        },
        "trials": trial_rows,
    }


def write_angle_csv(path, result):
    fieldnames = [
        "angle_deg", "mean_Ts", "mean_Tp", "p05_Ts", "p05_Tp",
        "min_Ts", "min_Tp", "passes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["per_angle"])


def main():
    parser = argparse.ArgumentParser(description="Wide-angle broadband transmission reliability")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Joint s+p checkpoint used to generate candidates")
    source.add_argument(
        "--structures", nargs="+", help="One or more JSON files containing candidate structures"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_candidates", type=int, default=256, help="Candidates per seed")
    parser.add_argument("--seeds", default="42,43,44,45")
    parser.add_argument("--coarse_angles", default="0:80:5")
    parser.add_argument("--dense_angles", default="0:80:1")
    parser.add_argument("--coarse_top_k", type=int, default=32)
    parser.add_argument("--reliability_top_k", type=int, default=3)
    parser.add_argument("--mean_t_threshold", type=float, default=0.85)
    parser.add_argument("--p05_t_threshold", type=float, default=0.80)
    parser.add_argument("--min_t_threshold", type=float, default=0.70)
    parser.add_argument("--mc_trials", type=int, default=100)
    parser.add_argument("--mc_angles", default="0:80:5")
    parser.add_argument("--mc_seed", type=int, default=20260727)
    parser.add_argument("--thickness_sigma_rel", type=float, default=0.02)
    parser.add_argument("--thickness_bias_sigma_rel", type=float, default=0.01)
    parser.add_argument("--n_sigma_rel", type=float, default=0.005)
    parser.add_argument("--angle_sigma_deg", type=float, default=0.25)
    args = parser.parse_args()

    if args.num_candidates < 1 or args.coarse_top_k < 1 or args.reliability_top_k < 0:
        raise ValueError("Candidate counts must be positive; reliability_top_k may be zero")
    if args.mc_trials < 0:
        raise ValueError("mc_trials cannot be negative")
    for value, name in (
        (args.mean_t_threshold, "mean_t_threshold"),
        (args.p05_t_threshold, "p05_t_threshold"),
        (args.min_t_threshold, "min_t_threshold"),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    for value, name in (
        (args.thickness_sigma_rel, "thickness_sigma_rel"),
        (args.thickness_bias_sigma_rel, "thickness_bias_sigma_rel"),
        (args.n_sigma_rel, "n_sigma_rel"),
        (args.angle_sigma_deg, "angle_sigma_deg"),
    ):
        if value < 0:
            raise ValueError(f"{name} cannot be negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    in_progress = output_dir / "WIDE_ANGLE_IN_PROGRESS.json"
    complete = output_dir / "WIDE_ANGLE_COMPLETE.json"
    if complete.exists():
        raise FileExistsError(f"Refusing to overwrite completed evaluation: {complete}")
    atomic_json_dump({"status": "in_progress", "started_at": datetime.now().isoformat()}, in_progress)

    coarse_angles = parse_angle_grid(args.coarse_angles)
    dense_angles = parse_angle_grid(args.dense_angles)
    mc_angles = parse_angle_grid(args.mc_angles)
    thresholds = {
        "mean_threshold": args.mean_t_threshold,
        "p05_threshold": args.p05_t_threshold,
        "min_threshold": args.min_t_threshold,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_configs = None
    if args.model:
        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
        candidates, model_configs = generate_model_candidates(
            args.model, args.num_candidates, seeds, device
        )
    else:
        seeds = []
        candidates = deduplicate_candidates(
            [
                candidate
                for path in args.structures
                for candidate in load_structure_candidates(path)
            ]
        )
    if not candidates:
        raise RuntimeError("No valid candidate structures were supplied or generated")

    load_materials, _spectrum = _get_sim_backend()
    nk_dict = load_materials(
        all_mats=[SUBSTRATE] + ALLOWED_MATERIALS + list(BANNED_MATERIALS),
        wavelengths=DEFAULT_WAVELENGTHS_UM,
        DATABASE=str(_PKG_ROOT / "optogpt" / "nk"),
    )

    coarse_results = []
    tmm_failures = []
    for candidate in candidates:
        try:
            result, _ = evaluate_candidate(candidate, coarse_angles, nk_dict, thresholds)
            coarse_results.append(result)
        except Exception as exc:
            tmm_failures.append({"structure_hash": candidate["structure_hash"], "error": str(exc)})
    coarse_results.sort(key=lambda row: (row["wide_angle_loss"], -row["angle_pass_rate"]))

    dense_results = []
    for candidate in coarse_results[: args.coarse_top_k]:
        result, _ = evaluate_candidate(candidate, dense_angles, nk_dict, thresholds)
        dense_results.append(result)
    dense_results.sort(key=lambda row: (row["wide_angle_loss"], -row["angle_pass_rate"]))

    reliability_results = []
    for rank, candidate in enumerate(dense_results[: args.reliability_top_k], start=1):
        reliability = monte_carlo_reliability(
            candidate,
            mc_angles,
            nk_dict,
            thresholds,
            args.mc_trials,
            args.mc_seed + rank - 1,
            args.thickness_sigma_rel,
            args.thickness_bias_sigma_rel,
            args.n_sigma_rel,
            args.angle_sigma_deg,
        )
        reliability_results.append(
            {"rank": rank, "structure_hash": candidate["structure_hash"], **reliability}
        )
        write_angle_csv(output_dir / f"rank_{rank:02d}_per_angle.csv", candidate)

    compact_dense = []
    for row in dense_results:
        compact_dense.append({key: value for key, value in row.items() if key != "per_angle"})
    summary = {
        "status": "complete",
        "created_at": datetime.now().isoformat(),
        "source": {"model": args.model, "structures": args.structures},
        "device": str(device),
        "model_configs": model_configs,
        "candidate_generation": {
            "seeds": seeds,
            "requested_per_seed": args.num_candidates if args.model else None,
            "unique_candidates": len(candidates),
        },
        "evaluation_contract": {
            "wavelengths_nm": [float(WAVELENGTHS_NM[0]), float(WAVELENGTHS_NM[-1]), 10.0],
            "polarizations": ["s", "p"],
            "coarse_angles_deg": coarse_angles.tolist(),
            "dense_angles_deg": dense_angles.tolist(),
            "monte_carlo_angles_deg": mc_angles.tolist(),
            "thresholds": thresholds,
            "nominal_pass_definition": (
                "At every evaluated angle, both polarizations must meet the band-mean, "
                "within-angle p05, and point-minimum thresholds."
            ),
        },
        "counts": {
            "coarse_valid": len(coarse_results),
            "tmm_failed": len(tmm_failures),
            "dense_evaluated": len(dense_results),
            "nominal_dense_pass": sum(row["all_angles_pass"] for row in dense_results),
            "monte_carlo_evaluated": len(reliability_results),
        },
        "best_nominal": dense_results[0] if dense_results else None,
        "dense_ranking": compact_dense,
        "reliability": reliability_results,
        "tmm_failures": tmm_failures,
        "claim_limit": (
            "This is TMM-based computational evidence under the recorded nk database and "
            "uncertainty model; experimental validation is required for a fabrication claim."
        ),
    }
    atomic_json_dump(summary, output_dir / "wide_angle_summary.json")
    atomic_json_dump(
        {"status": "complete", "created_at": datetime.now().isoformat()}, complete
    )
    if in_progress.exists():
        in_progress.unlink()

    best = summary["best_nominal"]
    print(f"Unique candidates: {len(candidates)}")
    print(f"Dense evaluations: {len(dense_results)}")
    if best:
        print(f"Best structure: {best['tokens']}")
        print(f"Worst-polarization angle mean T: {best['worst_pol_angle_mean_T']:.4f}")
        print(f"Worst-polarization grid p05 T: {best['worst_pol_grid_p05_T']:.4f}")
        print(f"Worst-polarization grid min T: {best['worst_pol_grid_min_T']:.4f}")
        print(f"All dense angles pass: {best['all_angles_pass']}")
    print(f"Report: {output_dir / 'wide_angle_summary.json'}")


if __name__ == "__main__":
    main()
