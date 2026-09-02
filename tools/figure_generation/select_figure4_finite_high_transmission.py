#!/usr/bin/env python3
"""Select finite-glass Figure 4 candidates by the same two-polarization high-T objective."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "paper_figures" / "data"
INPUT = DATA_DIR / "figure4_s_priority_flat95_pool_candidates.json"
OUTPUT = DATA_DIR / "figure4_finite_high_transmission_selection.json"


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
            current.append(min(current[-1] + 1, previous[col_index] + 1, previous[col_index - 1] + (left_item != right_item)))
        previous = current
    levenshtein = previous[-1] / max(len(left_materials), len(right_materials), 1)
    return 0.65 * levenshtein + 0.35 * jaccard


def main():
    raw = json.loads(INPUT.read_text())
    candidates = []
    for source in raw["candidates"]:
        candidate = dict(source)
        candidate["worst_pol_mean_T"] = min(candidate["mean_Ts"], candidate["mean_Tp"])
        candidate["mean_unpolarized_T"] = (candidate["mean_Ts"] + candidate["mean_Tp"]) / 2
        candidate["worst_pol_p05_T"] = min(candidate["p05_Ts"], candidate["p05_Tp"])
        candidate["high_transmission_score"] = (
            0.55 * candidate["worst_pol_mean_T"]
            + 0.30 * candidate["mean_unpolarized_T"]
            + 0.15 * candidate["worst_pol_p05_T"]
        )
        candidate["ranking_objective"] = "maximize worst-polarization transmission, then mean transmission"
        candidates.append(candidate)
    ranked = sorted(candidates, key=lambda candidate: (candidate["high_transmission_score"], candidate["worst_pol_mean_T"], candidate["mean_unpolarized_T"], -candidate["E_joint"]), reverse=True)
    shortlist = ranked[: max(30, min(100, len(ranked) // 10))]
    selected = [shortlist[0]]
    details = [{"selection_step": 1, "reason": "maximum high-transmission score"}]
    while len(selected) < 3:
        remaining = [candidate for candidate in shortlist if candidate not in selected]
        chosen = max(remaining, key=lambda candidate: (min(material_distance(candidate, prior) for prior in selected), candidate["high_transmission_score"]))
        selected.append(chosen)
        details.append({"selection_step": len(selected), "reason": "material diversity within high-transmission shortlist", "minimum_material_distance": min(material_distance(chosen, prior) for prior in selected[:-1])})
    for candidate, detail in zip(selected, details):
        candidate["selection"] = detail
    output = {
        "selection_contract": {
            "model": "coherent dielectric films with incoherent 500 um finite glass substrate, tmm.inc_tmm",
            "substrate": "Glass_Substrate",
            "substrate_thickness_nm": 500000,
            "theta_deg": raw["run_contract"]["theta_deg"],
            "wavelengths_nm": raw["run_contract"]["wavelengths_nm"],
            "objective": "maximize worst-polarization transmission, then mean unpolarized transmission, then material diversity",
            "source_pool": str(INPUT),
            "source_requested_candidates": raw["run_contract"]["requested_candidates"],
            "source_retained_tmm_candidates": raw["run_contract"]["retained_tmm_candidates"],
            "finite_shortlist_size": len(shortlist),
            "target_note": "Flat Rs=Rp=0.05 and Ts=Tp=0.95 are shown as a reference target; candidates are selected for highest achievable two-polarization transmission.",
        },
        "shared_target": raw["shared_target"],
        "selected_candidates": selected,
        "ranked_summary": [{"tokens": c["tokens"], "high_transmission_score": c["high_transmission_score"], "worst_pol_mean_T": c["worst_pol_mean_T"], "mean_unpolarized_T": c["mean_unpolarized_T"]} for c in ranked[:20]],
    }
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(json.dumps({"output": str(OUTPUT), "selected": [{"tokens": c["tokens"], "mean_Ts": c["mean_Ts"], "mean_Tp": c["mean_Tp"], "worst_pol_mean_T": c["worst_pol_mean_T"]} for c in selected]}, indent=2))


if __name__ == "__main__":
    main()
