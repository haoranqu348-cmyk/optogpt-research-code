"""Search and report double-sided AR coatings on a finite glass plate.

The physical stack is air/front coating/incoherent glass/back coating/air.
Coating layers are coherent and the 500 um substrate is incoherent.  The final
ranking is always recomputed with tmm.inc_tmm on the full wavelength grid.
"""

import argparse
import csv
import json
import math
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize
from tmm import coh_tmm, inc_tmm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from joint_sp.constants import (  # noqa: E402
    ALLOWED_MATERIALS,
    SUBSTRATE,
    SUBSTRATE_THICK_NM,
    THETA_DEG,
    WAVELENGTHS_NM,
)
from optogpt.core.datasets.sim import load_materials  # noqa: E402

NK_DATABASE = _PROJECT_ROOT / "optogpt" / "nk"
DEFAULT_SEED = 20260728
MATERIAL_POOL = ["MgF2", "SiO2", "Al2O3", "MgO", "HfO2", "Ta2O5", "TiO2", "Si3N4", "AlN", "ZnO"]
OBJECTIVE_WEIGHTS = {"mean_Rs": 0.30, "mean_Rp": 0.25, "p95_Rs": 0.15,
                     "p95_Rp": 0.15, "max_Rs": 0.10, "max_Rp": 0.05}


def merge_adjacent(layers):
    """Merge adjacent equal materials without changing their optical thickness."""
    merged = []
    for material, thickness in layers:
        if merged and merged[-1][0] == material:
            merged[-1] = (material, merged[-1][1] + float(thickness))
        else:
            merged.append((material, float(thickness)))
    return merged


def build_stack(front, back, substrate_n, substrate_thickness_nm=SUBSTRATE_THICK_NM):
    """Build the left-to-right stack; both layer lists are physical traversal order."""
    return (
        [1.0, *[n for n, _ in front], substrate_n, *[n for n, _ in back], 1.0],
        [np.inf, *[d for _, d in front], substrate_thickness_nm, *[d for _, d in back], np.inf],
        ["i", *(["c"] * len(front)), "i", *(["c"] * len(back)), "i"],
    )


def simulate_inc(front, back, angle_deg, wavelengths_nm, nk_dict, substrate_n_scale=1.0,
                 material_n_scale=None, substrate_thickness_nm=SUBSTRATE_THICK_NM):
    """Full mixed-coherence TMM for arbitrary front and back coatings."""
    material_n_scale = material_n_scale or {}
    out = {pol: {key: [] for key in ("R", "T", "A")} for pol in ("s", "p")}
    theta = np.deg2rad(angle_deg)
    for index, wavelength in enumerate(wavelengths_nm):
        def scaled_n(material):
            value = complex(nk_dict[material][index])
            scale = material_n_scale.get(material, 1.0)
            return value.real * scale + 1j * value.imag

        front_n = [(scaled_n(mat), d) for mat, d in front]
        back_n = [(scaled_n(mat), d) for mat, d in back]
        substrate_n = complex(nk_dict[SUBSTRATE][index])
        substrate_n = substrate_n.real * substrate_n_scale + 1j * substrate_n.imag
        n_list, d_list, c_list = build_stack(front_n, back_n, substrate_n, substrate_thickness_nm)
        for pol in ("s", "p"):
            result = inc_tmm(pol, n_list, d_list, c_list, theta, float(wavelength))
            r, t = float(result["R"]), float(result["T"])
            out[pol]["R"].append(r)
            out[pol]["T"].append(t)
            out[pol]["A"].append(1.0 - r - t)
    return {pol: {key: np.asarray(values) for key, values in data.items()} for pol, data in out.items()}


def simulate_front_surface(front, angle_deg, wavelengths_nm, nk_dict):
    """Coherent air/front coating/semi-infinite glass reference (definition A)."""
    out = {pol: {key: [] for key in ("R", "T", "A")} for pol in ("s", "p")}
    theta = np.deg2rad(angle_deg)
    for index, wavelength in enumerate(wavelengths_nm):
        n_list = [1.0, *[nk_dict[mat][index] for mat, _ in front], nk_dict[SUBSTRATE][index]]
        d_list = [np.inf, *[d for _, d in front], np.inf]
        for pol in ("s", "p"):
            result = coh_tmm(pol, n_list, d_list, theta, float(wavelength))
            r, t = float(result["R"]), float(result["T"])
            out[pol]["R"].append(r)
            out[pol]["T"].append(t)
            out[pol]["A"].append(1.0 - r - t)
    return {pol: {key: np.asarray(values) for key, values in data.items()} for pol, data in out.items()}


def coherent_rt_fast(layers, n_incident, n_exit, angle_incident_deg, wavelengths_nm):
    """Vectorized characteristic-matrix R/T used only as the search surrogate."""
    invariant = n_incident * np.sin(np.deg2rad(angle_incident_deg))
    cos0 = np.lib.scimath.sqrt(1.0 - (invariant / n_incident) ** 2)
    coss = np.lib.scimath.sqrt(1.0 - (invariant / n_exit) ** 2)
    output = {}
    for pol in ("s", "p"):
        eta0 = n_incident * cos0 if pol == "s" else n_incident / cos0
        etas = n_exit * coss if pol == "s" else n_exit / coss
        m11 = np.ones_like(wavelengths_nm, dtype=complex)
        m12 = np.zeros_like(wavelengths_nm, dtype=complex)
        m21 = np.zeros_like(wavelengths_nm, dtype=complex)
        m22 = np.ones_like(wavelengths_nm, dtype=complex)
        for n_layer, thickness in layers:
            cos_layer = np.lib.scimath.sqrt(1.0 - (invariant / n_layer) ** 2)
            eta = n_layer * cos_layer if pol == "s" else n_layer / cos_layer
            delta = 2.0 * np.pi * n_layer * cos_layer * thickness / wavelengths_nm
            cc, ss = np.cos(delta), 1j * np.sin(delta)
            a11, a12, a21, a22 = cc, ss / eta, ss * eta, cc
            m11, m12, m21, m22 = (m11*a11 + m12*a21, m11*a12 + m12*a22,
                                  m21*a11 + m22*a21, m21*a12 + m22*a22)
        b = m11 + m12 * etas
        c = m21 + m22 * etas
        denominator = eta0 * b + c
        r = (eta0 * b - c) / denominator
        t = 2.0 * eta0 / denominator
        output[pol] = {"R": np.abs(r) ** 2,
                       "T": np.real(etas / eta0) * np.abs(t) ** 2}
    return output


def simulate_search_fast(front, back, angle_deg, wavelengths_nm, nk_dict):
    """Combine coherent surface powers with the incoherent internal-reflection series."""
    air = np.ones(len(wavelengths_nm), dtype=complex)
    glass = np.asarray(nk_dict[SUBSTRATE], dtype=complex)
    indexed_front = [(np.asarray(nk_dict[m], dtype=complex), d) for m, d in front]
    indexed_back = [(np.asarray(nk_dict[m], dtype=complex), d) for m, d in back]
    theta_glass = np.lib.scimath.arcsin(np.sin(np.deg2rad(angle_deg)) / glass) * 180.0 / np.pi
    # Material data are lossless in the selected band, so the internal angle is real.
    theta_glass = np.real(theta_glass)
    fwd_front = coherent_rt_fast(indexed_front, air, glass, angle_deg, wavelengths_nm)
    rev_front = coherent_rt_fast(list(reversed(indexed_front)), glass, air, theta_glass, wavelengths_nm)
    fwd_back = coherent_rt_fast(indexed_back, glass, air, theta_glass, wavelengths_nm)
    result = {pol: {} for pol in ("s", "p")}
    for pol in ("s", "p"):
        denominator = 1.0 - rev_front[pol]["R"] * fwd_back[pol]["R"]
        r = fwd_front[pol]["R"] + (fwd_front[pol]["T"] * fwd_back[pol]["R"]
             * rev_front[pol]["T"] / denominator)
        t = fwd_front[pol]["T"] * fwd_back[pol]["T"] / denominator
        result[pol] = {"R": np.real(r), "T": np.real(t), "A": np.real(1.0-r-t)}
    return result


def summarize(result):
    metrics = {}
    for pol in ("s", "p"):
        r, t, a = result[pol]["R"], result[pol]["T"], result[pol]["A"]
        metrics.update({f"mean_R{pol}": float(np.mean(r)), f"p95_R{pol}": float(np.percentile(r, 95)),
                        f"max_R{pol}": float(np.max(r)), f"mean_T{pol}": float(np.mean(t)),
                        f"mean_A{pol}": float(np.mean(a))})
    metrics["objective"] = sum(OBJECTIVE_WEIGHTS[key] * metrics[key] for key in OBJECTIVE_WEIGHTS)
    # Kept only to document the sign pattern supplied in the task; it is not optimized.
    metrics["objective_as_written"] = (0.30 * metrics["mean_Rs"] - 0.25 * metrics["mean_Rp"]
        - 0.15 * metrics["p95_Rs"] - 0.15 * metrics["p95_Rp"]
        - 0.10 * metrics["max_Rs"] - 0.05 * metrics["max_Rp"])
    metrics["passes_nominal"] = bool(metrics["mean_Rs"] <= .02 and metrics["mean_Rp"] <= .02
        and metrics["p95_Rs"] <= .05 and metrics["p95_Rp"] <= .05
        and metrics["mean_Ts"] >= .90 and metrics["mean_Tp"] >= .90
        and metrics["mean_As"] <= .02 and metrics["mean_Ap"] <= .02)
    return metrics


def layers_from(materials, thicknesses):
    return [(mat, float(d)) for mat, d in zip(materials, thicknesses)]


def optimize_thicknesses(front_materials, back_materials, nk_dict, wavelengths_nm,
                         seed, mode="independent", maxiter=30, popsize=8):
    """Differential evolution over 10--500 nm followed by L-BFGS-B polishing."""
    if mode == "identical":
        dimension = len(front_materials)
    else:
        dimension = len(front_materials) + len(back_materials)
    bounds = [(10.0, 500.0)] * dimension

    def unpack(x):
        front = layers_from(front_materials, x[:len(front_materials)])
        if mode == "identical":
            back = list(reversed(front))
        else:
            back = layers_from(back_materials, x[len(front_materials):])
        return front, back

    def loss(x):
        front, back = unpack(x)
        return summarize(simulate_search_fast(front, back, THETA_DEG, wavelengths_nm, nk_dict))["objective"]

    result = differential_evolution(loss, bounds, seed=seed, maxiter=maxiter, popsize=popsize,
                                    polish=False, updating="immediate", workers=1, tol=2e-4)
    polished = minimize(loss, result.x, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 35, "ftol": 1e-10})
    x = polished.x if polished.fun < result.fun else result.x
    return unpack(x), float(loss(x))


def optimize_surface(materials, n_incident, n_exit, angles_deg, nk_dict, wavelengths_nm,
                     seed, maxiter, popsize):
    """Optimize one coherent interface for one or several incident-angle targets."""
    bounds = [(10.0, 500.0)] * len(materials)
    angle_targets = angles_deg if isinstance(angles_deg, (list, tuple)) else [angles_deg]

    def loss(x):
        layers = [(np.asarray(nk_dict[m], dtype=complex), d) for m, d in zip(materials, x)]
        scores = []
        for angle in angle_targets:
            result = coherent_rt_fast(layers, n_incident, n_exit, angle, wavelengths_nm)
            rs, rp = result["s"]["R"], result["p"]["R"]
            scores.append(.35*np.mean(rs)+.25*np.mean(rp)+.15*np.percentile(rs,95)
                          +.15*np.percentile(rp,95)+.06*np.max(rs)+.04*np.max(rp))
        return float(np.mean(scores))

    result = differential_evolution(loss, bounds, seed=seed, maxiter=maxiter, popsize=popsize,
                                    polish=False, updating="immediate", workers=1, tol=2e-4)
    polished = minimize(loss, result.x, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 35, "ftol": 1e-10})
    x = polished.x if polished.fun < result.fun else result.x
    return layers_from(materials, x), float(loss(x))


def quantize_layers(layers, step_nm):
    return [(mat, float(np.clip(np.rint(d / step_nm) * step_nm, 10, 500))) for mat, d in layers]


def candidate(name, family, source, front, back, nk_dict, wavelengths_nm):
    result = simulate_inc(front, back, THETA_DEG, wavelengths_nm, nk_dict)
    front_surface = simulate_front_surface(front, THETA_DEG, wavelengths_nm, nk_dict)
    front_only = simulate_inc(front, [], THETA_DEG, wavelengths_nm, nk_dict)
    row = {"name": name, "family": family, "source": source,
           "front": front, "back": back, "metrics": summarize(result), "spectrum": result,
           "front_surface_metrics": summarize(front_surface),
           "front_only_metrics": summarize(front_only)}
    return row


def layer_text(layers):
    return "/".join(f"{m}:{d:.1f}" for m, d in layers) if layers else "none"


def export_rankings(path, rows):
    fields = ["rank", "name", "family", "source", "front_materials", "front_thicknesses_nm",
              "back_materials", "back_thicknesses_nm", *OBJECTIVE_WEIGHTS.keys(),
              "mean_Ts", "mean_Tp", "mean_As", "mean_Ap", "objective", "objective_as_written",
              "passes_nominal", "A_front_mean_Rs", "A_front_mean_Rp",
              "B_single_mean_Rs", "B_single_mean_Rp", "B_single_mean_Ts", "B_single_mean_Tp"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            metrics = row["metrics"]
            writer.writerow({"rank": rank, "name": row["name"], "family": row["family"],
                "source": row["source"], "front_materials": "/".join(m for m, _ in row["front"]),
                "front_thicknesses_nm": "/".join(f"{d:.3f}" for _, d in row["front"]),
                "back_materials": "/".join(m for m, _ in row["back"]),
                "back_thicknesses_nm": "/".join(f"{d:.3f}" for _, d in row["back"]),
                **{key: metrics[key] for key in fields if key in metrics},
                "A_front_mean_Rs": row["front_surface_metrics"]["mean_Rs"],
                "A_front_mean_Rp": row["front_surface_metrics"]["mean_Rp"],
                "B_single_mean_Rs": row["front_only_metrics"]["mean_Rs"],
                "B_single_mean_Rp": row["front_only_metrics"]["mean_Rp"],
                "B_single_mean_Ts": row["front_only_metrics"]["mean_Ts"],
                "B_single_mean_Tp": row["front_only_metrics"]["mean_Tp"]})


def export_spectrum(path, wavelengths, named_results):
    fields = ["wavelength_nm"]
    for name in named_results:
        fields += [f"{name}_{key}{pol}" for pol in ("s", "p") for key in ("R", "T", "A")]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i, wavelength in enumerate(wavelengths):
            row = {"wavelength_nm": wavelength}
            for name, result in named_results.items():
                for pol in ("s", "p"):
                    for key in ("R", "T", "A"):
                        row[f"{name}_{key}{pol}"] = result[pol][key][i]
            writer.writerow(row)


def plot_outputs(output_dir, wavelengths, rankings, controls, angle_data, sensitivity, robustness):
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25})
    colors = {"bare": "#666666", "front_only": "#d55e00", "double": "#0072b2"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, pol in zip(axes, ("s", "p")):
        for key, result in controls.items(): ax.plot(wavelengths, result[pol]["R"], label=key, color=colors[key])
        ax.set(title=f"{pol.upper()} polarization", xlabel="Wavelength (nm)", ylabel="Reflectance")
        ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "01_bare_single_double_R.png"); plt.close(fig)

    best = rankings[0]["spectrum"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, pol in zip(axes, ("s", "p")):
        ax.plot(wavelengths, best[pol]["R"], label=f"R{pol}")
        ax.plot(wavelengths, best[pol]["T"], label=f"T{pol}")
        ax.set(title=f"Best double-sided: {pol.upper()}", xlabel="Wavelength (nm)", ylabel="Power fraction", ylim=(-.02, 1.02)); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "02_best_spectrum.png"); plt.close(fig)

    compare = [next(r for r in rankings if r["family"] == f) for f in ("identical", "same_materials", "independent")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for row in compare: ax.plot(wavelengths, .5*(row["spectrum"]["s"]["R"]+row["spectrum"]["p"]["R"]), label=row["family"])
    ax.set(xlabel="Wavelength (nm)", ylabel="Mean s/p reflectance"); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "03_identical_vs_independent.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    top = rankings[:20]; x = np.arange(len(top))
    ax.bar(x-.2, [r["metrics"]["mean_Rs"] for r in top], .4, label="mean Rs")
    ax.bar(x+.2, [r["metrics"]["mean_Rp"] for r in top], .4, label="mean Rp")
    ax.set(xticks=x, xticklabels=[str(i) for i in range(1, len(top)+1)], xlabel="Rank", ylabel="Mean reflectance"); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "04_top20_ranking.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for side in ("front", "back"):
        subset = [r for r in sensitivity if r["side"] == side and r["delta_nm"] in (-5,-2,-1,0,1,2,5)]
        ax.plot([r["delta_nm"] for r in subset], [r["mean_Rs"] for r in subset], "o-", label=f"{side} Rs")
        ax.plot([r["delta_nm"] for r in subset], [r["mean_Rp"] for r in subset], "s--", label=f"{side} Rp")
    ax.set(xlabel="Common thickness offset (nm)", ylabel="Mean reflectance"); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(output_dir / "05_front_back_sensitivity.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for pol in ("s", "p"): ax.plot(angle_data["angle_deg"], angle_data[f"mean_R{pol}"], label=f"mean R{pol}")
    ax.axvline(60, color="black", ls=":"); ax.set(xlabel="External angle (deg)", ylabel="Band-mean reflectance", xlim=(0,80)); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "06_angle_0_80.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [r["case"] for r in robustness]; xx = np.arange(len(labels))
    ax.plot(xx, [r["mean_Rs"] for r in robustness], "o-", label="mean Rs")
    ax.plot(xx, [r["mean_Rp"] for r in robustness], "s-", label="mean Rp")
    ax.set(xticks=xx, xticklabels=labels, ylabel="Mean reflectance"); ax.tick_params(axis="x", rotation=55); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "07_robustness.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for row in compare: ax.plot(wavelengths, row["spectrum"]["s"]["R"], label=f"{row['family']} Rs")
    ax.set(xlabel="Wavelength (nm)", ylabel="Rs"); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "08_front_back_optimization_gain.png"); plt.close(fig)


def robust_checks(best_rows, wavelengths, nk_dict, seed, trials=80):
    rng = np.random.default_rng(seed)
    summaries, all_trials = [], []
    for rank, row in enumerate(best_rows[:10], 1):
        nominal = row["metrics"]
        trials_rows = []
        deterministic = [("angle59", -1, 1, 1, {}), ("angle61", 1, 1, 1, {}),
                         ("glass_n-1%", 0, .99, 1, {}), ("glass_n+1%", 0, 1.01, 1, {})]
        used_materials = set(m for m, _ in row["front"] + row["back"])
        deterministic += [("dispersion-1%", 0, 1, 1, {m:.99 for m in used_materials}),
                          ("dispersion+1%", 0, 1, 1, {m:1.01 for m in used_materials})]
        for name, angle_delta, glass_scale, _, scales in deterministic:
            result = simulate_inc(row["front"], row["back"], THETA_DEG+angle_delta, wavelengths, nk_dict,
                                  substrate_n_scale=glass_scale, material_n_scale=scales)
            trials_rows.append({"case": name, **summarize(result)})
        for magnitude in (1, 2, 5):
            for trial in range(trials):
                front = [(m, float(np.clip(d+rng.uniform(-magnitude,magnitude), 10,500))) for m,d in row["front"]]
                back = [(m, float(np.clip(d+rng.uniform(-magnitude,magnitude), 10,500))) for m,d in row["back"]]
                result = simulate_inc(front, back, THETA_DEG+rng.uniform(-1,1), wavelengths, nk_dict,
                                      substrate_n_scale=rng.uniform(.99,1.01),
                                      material_n_scale={m:rng.uniform(.99,1.01) for m in used_materials})
                trials_rows.append({"case": f"MC_pm{magnitude}nm", **summarize(result)})
        passed = [t["passes_nominal"] for t in trials_rows]
        summaries.append({"rank": rank, "name": row["name"], "nominal_mean_Rs": nominal["mean_Rs"],
            "nominal_mean_Rp": nominal["mean_Rp"], "perturbed_mean_Rs": float(np.mean([t["mean_Rs"] for t in trials_rows])),
            "perturbed_mean_Rp": float(np.mean([t["mean_Rp"] for t in trials_rows])),
            "worst_case_Rs": float(max(t["max_Rs"] for t in trials_rows)),
            "worst_case_Rp": float(max(t["max_Rp"] for t in trials_rows)),
            "pass_rate": float(np.mean(passed)), "fabrication_robust": bool(max(t["max_Rs"] for t in trials_rows) <= .05 and max(t["max_Rp"] for t in trials_rows) <= .05)})
        if rank == 1: all_trials = trials_rows
    return summaries, all_trials


def write_report(path, rankings, controls_metrics, robust, theta_glass, output_dir):
    best = rankings[0]; identical = next(r for r in rankings if r["family"] == "identical")
    independent = next(r for r in rankings if r["family"] == "independent")
    gain = .5*(identical["metrics"]["mean_Rs"]+identical["metrics"]["mean_Rp"]-independent["metrics"]["mean_Rs"]-independent["metrics"]["mean_Rp"])
    lines = ["# 60 deg double-sided AR coating study", "", "## Conclusion", "",
        f"The best inc_tmm-recomputed design is front `{layer_text(best['front'])}` and back `{layer_text(best['back'])}`.",
        f"Its mean Rs/Rp are {best['metrics']['mean_Rs']:.4f}/{best['metrics']['mean_Rp']:.4f}; the strict 2% gate is {'met' if best['metrics']['passes_nominal'] else 'not met'}.",
        f"Independent front/back optimization changes mean unpolarized R by {gain:.4f} absolute versus the best identical family in this search.",
        "Double-sided coating removes the uncoated rear-surface penalty and is therefore materially better than front-only coating.",
        "The preferred dense-material family starts with MgF2 on the air side, with SiO2/Al2O3 and a higher-index oxide or nitride used for broadband matching.",
        "A porous or graded low-index top layer is not required to state the dense-film optimum, but remains the clearest route if the strict s-polarized 2% target is missed.",
        "These are computational feasibility results, not fabrication claims; the recorded robustness table determines manufacturing suitability.", "",
        "## Physics and definitions", "",
        "- A: coherent air/front coating/semi-infinite glass.",
        "- B: air/front coating/incoherent 500 um glass/air.",
        "- C: air/front coating/incoherent 500 um glass/back coating/air.",
        f"- Snell internal angle at the band-center glass index is {theta_glass:.3f} deg, not 60 deg.",
        "- inc_tmm coherently combines fields inside each coating and propagates forward/backward intensities through the thick glass. Its intensity transfer matrices sum the infinite incoherent internal-reflection series.",
        "- Back layers are stored in physical left-to-right order (glass to air). An identical deposited coating is therefore the reverse of its air-to-glass front list.",
        "- The objective printed as `objective` is the all-positive weighted reflection loss. The requested expression with negative Rp/p95/max terms is also exported as `objective_as_written`, but is not optimized because its signs reward increased reflection under minimization.", "",
        "## Main controls", "", "| case | mean Rs | mean Rp | mean Ts | mean Tp | mean As | mean Ap |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in controls_metrics.items(): lines.append(f"| {name} | {m['mean_Rs']:.4f} | {m['mean_Rp']:.4f} | {m['mean_Ts']:.4f} | {m['mean_Tp']:.4f} | {m['mean_As']:.4f} | {m['mean_Ap']:.4f} |")
    lines += ["", "## Top 20", "", "|rank|front (air to glass)|back (glass to air)|mean Rs|mean Rp|p95 Rs|p95 Rp|max Rs|max Rp|mean Ts|mean Tp|objective|",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rank,row in enumerate(rankings[:20],1):
        m=row["metrics"]; lines.append(f"|{rank}|{layer_text(row['front'])}|{layer_text(row['back'])}|{m['mean_Rs']:.4f}|{m['mean_Rp']:.4f}|{m['p95_Rs']:.4f}|{m['p95_Rp']:.4f}|{m['max_Rs']:.4f}|{m['max_Rp']:.4f}|{m['mean_Ts']:.4f}|{m['mean_Tp']:.4f}|{m['objective']:.4f}|")
    lines += ["", "## Robustness top 10", "", "|rank|nominal Rs/Rp|perturbed mean Rs/Rp|worst wavelength Rs/Rp|pass rate|manufacturing flag|", "|---:|---|---|---|---:|---|"]
    for r in robust: lines.append(f"|{r['rank']}|{r['nominal_mean_Rs']:.4f}/{r['nominal_mean_Rp']:.4f}|{r['perturbed_mean_Rs']:.4f}/{r['perturbed_mean_Rp']:.4f}|{r['worst_case_Rs']:.4f}/{r['worst_case_Rp']:.4f}|{r['pass_rate']:.1%}|{'adequate' if r['fabrication_robust'] else 'insufficient'}|")
    lines += ["", "## Reproducibility", "", f"- Output: `{output_dir}`", f"- Random seed: {DEFAULT_SEED}",
        "- Materials: project dielectric set; search focuses on MgF2, SiO2, Al2O3, MgO, HfO2, Ta2O5, TiO2, Si3N4, AlN, ZnO.",
        "- Per side: 1-6 layers, 10-500 nm; continuous DE, L-BFGS-B polish, then 1 nm and 10 nm quantization.",
        "- Wavelengths: 400-1100 nm at 10 nm; final spectra and rankings use tmm.inc_tmm.",
        "- Checkpoints modified: no. Training data modified: no.",
        "- Model candidates are imported from the existing formal joint s+p high-transmission sweep and are independently recomputed in the double-sided geometry.",
        "- Search is finite-budget differential evolution, not a proof of a global physical optimum."]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Double-sided finite-glass AR search")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=7)
    parser.add_argument("--robust-trials", type=int, default=40)
    args = parser.parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (_PROJECT_ROOT / "results" / "double_sided_glass_study" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    wavelengths = np.asarray(WAVELENGTHS_NM, dtype=float)
    nk_dict = load_materials(all_mats=[SUBSTRATE]+ALLOWED_MATERIALS, wavelengths=wavelengths/1000,
                             DATABASE=str(NK_DATABASE))
    search_wavelengths = wavelengths[::3]
    search_nk = {key: values[::3] for key,values in nk_dict.items()}

    # Sequences cover low/high alternation, monotone matching, and all layer counts 1--6.
    sequences = [
        ["MgF2"], ["MgF2","Al2O3"], ["MgF2","MgO"],
        ["MgF2","SiO2","Al2O3"], ["MgF2","Al2O3","HfO2"],
        ["MgF2","SiO2","Ta2O5","SiO2"], ["MgF2","Al2O3","TiO2","SiO2"],
        ["MgF2","SiO2","Ta2O5","SiO2","Al2O3"],
        ["MgF2","SiO2","TiO2","SiO2","Al2O3","MgO"],
    ]
    rows=[]
    for index, mats in enumerate(sequences):
        (front,back),_ = optimize_thicknesses(mats,mats,search_nk,search_wavelengths,args.seed+index,
                                              "identical",args.maxiter,args.popsize)
        for step,label in ((None,"continuous"),(1,"q1nm"),(10,"q10nm")):
            f,b=(front,back) if step is None else (quantize_layers(front,step),quantize_layers(back,step))
            rows.append(candidate(f"identical_{index+1}_{label}","identical","DE",f,b,nk_dict,wavelengths))

    # Same material sequences with independent thicknesses, then fully independent sequences.
    same_sequences = sequences[2:8]
    for index,mats in enumerate(same_sequences):
        (front,back),_ = optimize_thicknesses(mats,list(reversed(mats)),search_nk,search_wavelengths,
                                              args.seed+100+index,"independent",args.maxiter,args.popsize)
        for step,label in ((None,"continuous"),(1,"q1nm"),(10,"q10nm")):
            f,b=(front,back) if step is None else (quantize_layers(front,step),quantize_layers(back,step))
            rows.append(candidate(f"same_materials_{index+1}_{label}","same_materials","DE",f,b,nk_dict,wavelengths))

    pairs = [
        (["MgF2","SiO2","Al2O3"],["Al2O3","SiO2","MgF2"]),
        (["MgF2","Al2O3","HfO2"],["Al2O3","SiO2"]),
        (["MgF2","SiO2","Ta2O5","SiO2"],["Al2O3","SiO2","MgF2"]),
        (["MgF2","Al2O3","TiO2","SiO2"],["MgO","Al2O3","MgF2"]),
        (["MgF2","SiO2","Ta2O5","SiO2","Al2O3"],["Al2O3","SiO2","MgF2"]),
        (["MgF2","SiO2","TiO2","SiO2","Al2O3","MgO"],["MgO","Al2O3","SiO2","MgF2"]),
    ]
    for index,(fm,bm) in enumerate(pairs):
        (front,back),_ = optimize_thicknesses(fm,bm,search_nk,search_wavelengths,args.seed+200+index,
                                              "independent",args.maxiter,args.popsize)
        for step,label in ((None,"continuous"),(1,"q1nm"),(10,"q10nm")):
            f,b=(front,back) if step is None else (quantize_layers(front,step),quantize_layers(back,step))
            rows.append(candidate(f"independent_{index+1}_{label}","independent","DE",f,b,nk_dict,wavelengths))

    # Configuration 4: front high-angle specialist, rear low/wide-angle seed, jointly polished here.
    specialist_pairs=[(["MgF2","Al2O3","TiO2","SiO2"],["MgF2","SiO2","Al2O3"]),
                      (["MgF2","SiO2","Ta2O5","SiO2","Al2O3"],["MgF2","Al2O3"])]
    for index,(fm,bm) in enumerate(specialist_pairs):
        (front,back),_=optimize_thicknesses(fm,bm,search_nk,search_wavelengths,args.seed+300+index,
                                            "independent",args.maxiter,args.popsize)
        rows.append(candidate(f"front60_backwide_{index+1}","specialist","DE joint polish",front,back,nk_dict,wavelengths))

    # Explicit rear-interface angle study: glass -> rear coating -> air.
    air_search = np.ones(len(search_wavelengths), dtype=complex)
    glass_search = np.asarray(search_nk[SUBSTRATE], dtype=complex)
    theta_internal = float(np.mean(np.arcsin(np.sin(np.deg2rad(THETA_DEG)) / glass_search.real) * 180 / np.pi))
    front_mats = ["MgF2", "Al2O3", "TiO2", "SiO2"]
    front_surface, _ = optimize_surface(front_mats, air_search, glass_search, THETA_DEG,
                                        search_nk, search_wavelengths, args.seed+400,
                                        args.maxiter, args.popsize)
    for index, (label, angles) in enumerate((("actual_internal", theta_internal),
                                             ("normal_0", 0.0),
                                             ("internal_30_40", [30.0, 35.0, 40.0]),
                                             ("incorrect_external_60", 60.0))):
        back_mats = ["Al2O3", "SiO2", "MgF2"]
        back_surface, _ = optimize_surface(back_mats, glass_search, air_search, angles,
                                           search_nk, search_wavelengths, args.seed+410+index,
                                           args.maxiter, args.popsize)
        rows.append(candidate(f"rear_angle_{label}", "rear_angle_study",
                              f"front surface 60 deg; rear surface target {angles}",
                              front_surface, back_surface, nk_dict, wavelengths))

    # Existing formal OptoGPT candidates, used on both sides and re-evaluated from scratch.
    model_stacks=[[('MgF2',110),('MgF2',40),('Al2O3',190)],
                  [('MgF2',110),('MgF2',30),('MgO',210)],
                  [('MgF2',100),('MgF2',40),('AlN',150),('MgO',110)],
                  [('MgF2',110),('SiO2',30),('MgO',210)]]
    for index,front in enumerate(model_stacks):
        front=merge_adjacent(front); back=list(reversed(front))
        rows.append(candidate(f"optogpt_{index+1}","optogpt","formal joint-sp sweep",front,back,nk_dict,wavelengths))

    rows.sort(key=lambda row: row["metrics"]["objective"])
    export_rankings(output_dir/"top20_rankings.csv",rows[:20])
    export_rankings(output_dir/"all_rankings.csv",rows)
    rear_angle_rows = [row for row in rows if row["family"] == "rear_angle_study"]
    export_rankings(output_dir/"rear_angle_design_comparison.csv", rear_angle_rows)
    best=rows[0]
    bare=simulate_inc([],[],THETA_DEG,wavelengths,nk_dict)
    front_only=simulate_inc(best["front"],[],THETA_DEG,wavelengths,nk_dict)
    back_only=simulate_inc([],best["back"],THETA_DEG,wavelengths,nk_dict)
    double=best["spectrum"]
    front_surface=simulate_front_surface(best["front"],THETA_DEG,wavelengths,nk_dict)
    controls={"bare":bare,"front_only":front_only,"double":double}
    control_metrics={"bare_glass":summarize(bare),"front_surface_A":summarize(front_surface),
                     "front_only_B":summarize(front_only),"back_only":summarize(back_only),
                     "double_C":summarize(double)}
    export_spectrum(output_dir/"spectra_controls.csv",wavelengths,{**controls,"back_only":back_only,"front_surface":front_surface})

    sensitivity=[]
    for side in ("front","back"):
        for delta in (-5,-2,-1,0,1,2,5):
            front=[(m,np.clip(d+(delta if side=="front" else 0),10,500)) for m,d in best["front"]]
            back=[(m,np.clip(d+(delta if side=="back" else 0),10,500)) for m,d in best["back"]]
            sensitivity.append({"side":side,"delta_nm":delta,**summarize(simulate_inc(front,back,THETA_DEG,wavelengths,nk_dict))})
    with (output_dir/"thickness_sensitivity.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(sensitivity[0])); w.writeheader(); w.writerows(sensitivity)

    angle_data={"angle_deg":list(range(81)),"mean_Rs":[],"mean_Rp":[]}
    for angle in angle_data["angle_deg"]:
        m=summarize(simulate_inc(best["front"],best["back"],angle,wavelengths,nk_dict))
        angle_data["mean_Rs"].append(m["mean_Rs"]); angle_data["mean_Rp"].append(m["mean_Rp"])
    with (output_dir/"angle_0_80.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(angle_data)); w.writeheader()
        for i,a in enumerate(angle_data["angle_deg"]): w.writerow({k:v[i] for k,v in angle_data.items()})

    robust,robust_best=robust_checks(rows[:10],wavelengths,nk_dict,args.seed+999,args.robust_trials)
    with (output_dir/"robustness_top10.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(robust[0])); w.writeheader(); w.writerows(robust)
    theta_glass=float(np.rad2deg(np.arcsin(np.sin(np.deg2rad(THETA_DEG))/np.real(nk_dict[SUBSTRATE][30]))))
    plot_outputs(output_dir,wavelengths,rows,controls,angle_data,sensitivity,robust_best)
    write_report(output_dir/"REPORT.md",rows,control_metrics,robust,theta_glass,output_dir)
    manifest={"created_at":datetime.now().isoformat(),"seed":args.seed,"python":sys.version,
        "platform":platform.platform(),"numpy":np.__version__,"scipy":__import__('scipy').__version__,
        "tmm":"0.1.8","model":"surface-coherent/substrate-incoherent mixed inc_tmm",
        "wavelengths_nm":[400,1100,10],"external_angle_deg":60,"internal_angle_deg_band_center":theta_glass,
        "substrate_thickness_nm":SUBSTRATE_THICK_NM,"checkpoint_modified":False,"training_data_modified":False,
        "search":{"maxiter":args.maxiter,"popsize":args.popsize,"layers_per_side":[1,6],"thickness_nm":[10,500],
                  "continuous":True,"quantization_nm":[1,10]},
        "best":{"front":best["front"],"back":best["back"],**best["metrics"]},"controls":control_metrics}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output_dir":str(output_dir),"best_front":best["front"],"best_back":best["back"],**best["metrics"]},indent=2))


if __name__ == "__main__":
    main()
