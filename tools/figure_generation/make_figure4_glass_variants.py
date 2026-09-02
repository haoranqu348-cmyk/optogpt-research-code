#!/usr/bin/env python3
"""Recompute and select high-transmission Figure 4 candidates for semi-infinite glass."""

import csv
import json
from pathlib import Path

import numpy as np
from tmm import coh_tmm


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "paper_figures" / "data"
NK_DIR = ROOT / "optogpt_project" / "optogpt" / "optogpt" / "nk"
INPUT = DATA_DIR / "figure4_s_priority_flat95_pool_candidates.json"
OUTPUT = DATA_DIR / "figure4_semi_infinite_high_transmission_selection.json"
CSV_OUTPUT = DATA_DIR / "figure4_semi_infinite_high_transmission_spectra.csv"
WAVELENGTHS_NM = np.arange(400, 1101, 10)
WAVELENGTHS_UM = WAVELENGTHS_NM / 1000.0
THETA_DEG = 60.0


def load_nk(material):
    raw = np.genfromtxt(NK_DIR / f"{material}.csv", delimiter=",", names=True)
    nk = np.interp(WAVELENGTHS_UM, raw["wl"], raw["n"]) + 1j * np.interp(
        WAVELENGTHS_UM, raw["wl"], raw["k"]
    )
    return nk


def material_distance(left, right):
    left_materials = left["materials"]
    right_materials = right["materials"]
    left_set = set(left_materials)
    right_set = set(right_materials)
    union = left_set | right_set
    jaccard = 1.0 - len(left_set & right_set) / max(len(union), 1)
    previous = list(range(len(right_materials) + 1))
    for row_index, left_item in enumerate(left_materials, 1):
        current = [row_index]
        for col_index, right_item in enumerate(right_materials, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[col_index] + 1,
                    previous[col_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    levenshtein = previous[-1] / max(len(left_materials), len(right_materials), 1)
    return 0.65 * levenshtein + 0.35 * jaccard


def simulate(candidate, nk_dict, glass_nk):
    n_list = [1.0, *[nk_dict[material] for material in candidate["materials"]], glass_nk]
    d_list = [np.inf, *candidate["thicknesses"], np.inf]
    result = {pol: {key: [] for key in ("R", "T", "A")} for pol in ("s", "p")}
    for index, wavelength_nm in enumerate(WAVELENGTHS_NM):
        local_n = [values[index] if isinstance(values, np.ndarray) else values for values in n_list]
        for pol in ("s", "p"):
            tmm_result = coh_tmm(pol, local_n, d_list, np.deg2rad(THETA_DEG), float(wavelength_nm))
            result[pol]["R"].append(float(tmm_result["R"]))
            result[pol]["T"].append(float(tmm_result["T"]))
            result[pol]["A"].append(float(1.0 - tmm_result["R"] - tmm_result["T"]))
    return result


def main():
    raw = json.loads(INPUT.read_text())
    materials = sorted({material for candidate in raw["candidates"] for material in candidate["materials"]})
    nk_dict = {material: load_nk(material) for material in materials}
    glass_nk = load_nk("Glass_Substrate")
    target_r = np.full(len(WAVELENGTHS_NM), 0.05, dtype=float)
    scored = []
    for source in raw["candidates"]:
        candidate = {key: source[key] for key in ("tokens", "materials", "thicknesses", "n_layers")}
        result = simulate(candidate, nk_dict, glass_nk)
        rs = np.asarray(result["s"]["R"])
        rp = np.asarray(result["p"]["R"])
        ts = np.asarray(result["s"]["T"])
        tp = np.asarray(result["p"]["T"])
        candidate.update(
            {
                "sim_Rs": rs.tolist(),
                "sim_Ts": ts.tolist(),
                "sim_Rp": rp.tolist(),
                "sim_Tp": tp.tolist(),
                "E_s": float(np.mean(np.abs(rs - target_r))),
                "E_p": float(np.mean(np.abs(rp - target_r))),
                "E_joint": float(np.mean((np.abs(rs - target_r) + np.abs(rp - target_r)) / 2)),
                "mean_Ts": float(np.mean(ts)),
                "mean_Tp": float(np.mean(tp)),
                "p05_Ts": float(np.percentile(ts, 5)),
                "p05_Tp": float(np.percentile(tp, 5)),
                "min_Ts": float(np.min(ts)),
                "min_Tp": float(np.min(tp)),
                "mean_unpolarized_T": float(np.mean((ts + tp) / 2)),
                "worst_pol_mean_T": float(min(np.mean(ts), np.mean(tp))),
                "worst_pol_p05_T": float(min(np.percentile(ts, 5), np.percentile(tp, 5))),
                "high_transmission_score": float(
                    0.55 * min(np.mean(ts), np.mean(tp))
                    + 0.30 * np.mean((ts + tp) / 2)
                    + 0.15 * min(np.percentile(ts, 5), np.percentile(tp, 5))
                ),
                "ranking_objective": "maximize worst-polarization transmission, then mean transmission",
            }
        )
        scored.append(candidate)

    ranked = sorted(
        scored,
        key=lambda candidate: (
            candidate["high_transmission_score"],
            candidate["worst_pol_mean_T"],
            candidate["mean_unpolarized_T"],
            -candidate["E_joint"],
        ),
        reverse=True,
    )
    shortlist = ranked[: max(30, min(100, len(ranked) // 10))]
    selected = [shortlist[0]]
    selection_details = [{"selection_step": 1, "reason": "maximum high-transmission score"}]
    while len(selected) < 3:
        remaining = [candidate for candidate in shortlist if candidate not in selected]
        chosen = max(
            remaining,
            key=lambda candidate: (
                min(material_distance(candidate, prior) for prior in selected),
                candidate["high_transmission_score"],
            ),
        )
        selected.append(chosen)
        selection_details.append(
            {
                "selection_step": len(selected),
                "reason": "material diversity within high-transmission shortlist",
                "minimum_material_distance": min(material_distance(chosen, prior) for prior in selected[:-1]),
            }
        )

    for candidate, detail in zip(selected, selection_details):
        candidate["selection"] = detail
    output = {
        "selection_contract": {
            "model": "coherent air/dielectric multilayer/semi-infinite glass TMM",
            "substrate": "Glass_Substrate",
            "substrate_thickness_nm": None,
            "theta_deg": THETA_DEG,
            "wavelengths_nm": WAVELENGTHS_NM.tolist(),
            "objective": "maximize worst-polarization transmission, then mean unpolarized transmission, then material diversity",
            "source_pool": str(INPUT),
            "source_requested_candidates": raw["run_contract"]["requested_candidates"],
            "source_retained_tmm_candidates": raw["run_contract"]["retained_tmm_candidates"],
            "semi_infinite_shortlist_size": len(shortlist),
            "target_note": "Flat Rs=Rp=0.05 and Ts=Tp=0.95 are shown as a reference target; candidates are selected for highest achievable two-polarization transmission.",
        },
        "shared_target": {
            "target_profile": "flat_high_transmission",
            "target_definition": {"Rs": 0.05, "Ts": 0.95, "Rp": 0.05, "Tp": 0.95},
            "wavelengths_nm": WAVELENGTHS_NM.tolist(),
            "Rs": target_r.tolist(),
            "Ts": (1 - target_r).tolist(),
            "Rp": target_r.tolist(),
            "Tp": (1 - target_r).tolist(),
        },
        "selected_candidates": selected,
        "ranked_summary": [
            {
                "tokens": candidate["tokens"],
                "high_transmission_score": candidate["high_transmission_score"],
                "worst_pol_mean_T": candidate["worst_pol_mean_T"],
                "mean_unpolarized_T": candidate["mean_unpolarized_T"],
            }
            for candidate in ranked[:20]
        ],
    }
    OUTPUT.write_text(json.dumps(output, indent=2))
    with CSV_OUTPUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength_nm", "target_Rs", "target_Ts", "target_Rp", "target_Tp", *[f"candidate_{index}_{value}" for index in range(1, 4) for value in ("Rs", "Ts", "Rp", "Tp")]])
        for index, wavelength in enumerate(WAVELENGTHS_NM):
            row = [wavelength, target_r[index], 1 - target_r[index], target_r[index], 1 - target_r[index]]
            for candidate in selected:
                row.extend([candidate[f"sim_{value}"][index] for value in ("Rs", "Ts", "Rp", "Tp")])
            writer.writerow(row)
    print(json.dumps({"output": str(OUTPUT), "csv": str(CSV_OUTPUT), "selected": [{"tokens": c["tokens"], "mean_Ts": c["mean_Ts"], "mean_Tp": c["mean_Tp"], "worst_pol_mean_T": c["worst_pol_mean_T"]} for c in selected]}, indent=2))


if __name__ == "__main__":
    main()
