#!/usr/bin/env python3
"""Select Figure 4 candidates with s-polarization target agreement first."""

import argparse
import csv
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent


def normalized_levenshtein(left, right):
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, 1):
        current = [row_index]
        for col_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[col_index] + 1,
                    previous[col_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right), 1)


def material_distance(left, right):
    left_sequence = left["materials"]
    right_sequence = right["materials"]
    left_set = set(left_sequence)
    right_set = set(right_sequence)
    union = left_set | right_set
    jaccard = 1.0 - len(left_set & right_set) / max(len(union), 1)
    return 0.65 * normalized_levenshtein(left_sequence, right_sequence) + 0.35 * jaccard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=WORKSPACE / "paper_figures" / "data" / "figure4_s_priority_flat95_pool_candidates.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "paper_figures" / "data" / "figure4_s_priority_diverse_selection.json",
    )
    parser.add_argument("--absolute-es-tolerance", type=float, default=0.015)
    parser.add_argument("--absolute-ep-tolerance", type=float, default=0.005)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text())
    candidates = sorted(raw["candidates"], key=lambda item: (item["E_s"], item["E_p"]))
    best_es = candidates[0]["E_s"]
    threshold = best_es + args.absolute_es_tolerance
    s_shortlist = [candidate for candidate in candidates if candidate["E_s"] <= threshold]
    best_ep_in_s_shortlist = min(candidate["E_p"] for candidate in s_shortlist)
    ep_threshold = best_ep_in_s_shortlist + args.absolute_ep_tolerance
    shortlist = [candidate for candidate in s_shortlist if candidate["E_p"] <= ep_threshold]

    # The first candidate is the global best s-polarization match. Subsequent
    # candidates must pass both s and p error windows before diversity is used.
    selected = [candidates[0]]
    details = [{"selection_step": 1, "reason": "minimum E_s", "minimum_material_distance": None}]
    while len(selected) < args.count:
        remaining = [candidate for candidate in shortlist if candidate not in selected]
        scored = []
        for candidate in remaining:
            distance = min(material_distance(candidate, prior) for prior in selected)
            scored.append((distance, -candidate["E_s"], -candidate["E_p"], candidate))
        distance, _, _, chosen = max(scored, key=lambda item: (item[0], item[1], item[2]))
        selected.append(chosen)
        details.append(
            {
                "selection_step": len(selected),
                "reason": "maximum material distance after both s-priority and p-error filtering",
                "minimum_material_distance": distance,
            }
        )

    output = {
        "selection_contract": {
            "priority_1": "Minimize E_s against flat Rs=0.05 and Ts=0.95 over 400-1100 nm.",
            "priority_2": "Within E_s <= best E_s + 0.015, retain E_p <= best shortlist E_p + 0.005.",
            "priority_3": "Only after s and p filtering, maximize material sequence/set diversity.",
            "best_E_s": best_es,
            "E_s_shortlist_threshold": threshold,
            "absolute_E_s_tolerance": args.absolute_es_tolerance,
            "best_E_p_in_s_shortlist": best_ep_in_s_shortlist,
            "E_p_shortlist_threshold": ep_threshold,
            "absolute_E_p_tolerance": args.absolute_ep_tolerance,
            "s_shortlist_size": len(s_shortlist),
            "shortlist_size": len(shortlist),
            "requested_candidates": raw["run_contract"]["requested_candidates"],
            "retained_tmm_candidates": raw["run_contract"]["retained_tmm_candidates"],
            "source_pool": str(args.input),
            "attainment_note": (
                "The 0.05/0.95 curve is the requested target. The retained model-generated "
                "candidates do not attain mean Ts=0.95; predicted TMM curves are shown separately."
            ),
        },
        "shared_target": raw["shared_target"],
        "selected_candidates": [
            {**candidate, "selection": details[index]}
            for index, candidate in enumerate(selected)
        ],
        "pairwise_material_distance": [
            {
                "left": left + 1,
                "right": right + 1,
                "distance": material_distance(selected[left], selected[right]),
            }
            for left in range(len(selected))
            for right in range(left + 1, len(selected))
        ],
    }
    args.output.write_text(json.dumps(output, indent=2))

    csv_path = args.output.with_name("figure4_s_priority_diverse_spectra.csv")
    wavelengths = raw["run_contract"]["wavelengths_nm"]
    target = raw["shared_target"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "wavelength_nm", "target_Rs", "target_Ts", "target_Rp", "target_Tp",
                *[
                    f"candidate_{rank}_{quantity}"
                    for rank in range(1, args.count + 1)
                    for quantity in ("Rs", "Ts", "Rp", "Tp")
                ],
            ]
        )
        for index, wavelength in enumerate(wavelengths):
            row = [wavelength, target["Rs"][index], target["Ts"][index], target["Rp"][index], target["Tp"][index]]
            for candidate in selected:
                row.extend([candidate["sim_Rs"][index], candidate["sim_Ts"][index], candidate["sim_Rp"][index], candidate["sim_Tp"][index]])
            writer.writerow(row)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "csv": str(csv_path),
                "best_E_s": best_es,
                "threshold": threshold,
                "shortlist_size": len(shortlist),
                "selected": [
                    {
                        "tokens": candidate["tokens"],
                        "E_s": candidate["E_s"],
                        "E_p": candidate["E_p"],
                        "mean_Ts": candidate["mean_Ts"],
                        "mean_Tp": candidate["mean_Tp"],
                    }
                    for candidate in selected
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
