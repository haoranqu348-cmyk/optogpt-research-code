#!/usr/bin/env python3
"""Select high-transmission candidates, then maximize material diversity."""

import argparse
import csv
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent


def material_sequence(candidate):
    return list(candidate["materials"])


def normalized_levenshtein(left, right):
    if not left and not right:
        return 0.0
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
    left_sequence = material_sequence(left)
    right_sequence = material_sequence(right)
    left_set = set(left_sequence)
    right_set = set(right_sequence)
    union = left_set | right_set
    jaccard_distance = 1.0 - len(left_set & right_set) / max(len(union), 1)
    sequence_distance = normalized_levenshtein(left_sequence, right_sequence)
    return 0.65 * sequence_distance + 0.35 * jaccard_distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=WORKSPACE / "paper_figures" / "data" / "figure4_high_transmission_pool_candidates.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "paper_figures" / "data" / "figure4_high_transmission_diverse_selection.json",
    )
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text())
    candidates = raw["candidates"]
    if len(candidates) < args.count:
        raise RuntimeError("Candidate pool is too small")

    candidates = sorted(candidates, key=lambda item: (item["high_T_loss"], item["E_joint"]))
    best_loss = candidates[0]["high_T_loss"]
    threshold = best_loss * (1.0 + args.relative_tolerance)
    shortlist = [candidate for candidate in candidates if candidate["high_T_loss"] <= threshold]
    if len(shortlist) < args.count:
        shortlist = candidates[: max(args.count, len(shortlist))]

    selected = [shortlist[0]]
    selection_details = [
        {
            "selection_step": 1,
            "reason": "lowest high_T_loss in the retained pool",
            "minimum_distance_to_previous": None,
        }
    ]
    while len(selected) < args.count:
        remaining = [candidate for candidate in shortlist if candidate not in selected]
        if not remaining:
            raise RuntimeError("Not enough distinct shortlisted candidates")
        scored = []
        for candidate in remaining:
            minimum_distance = min(material_distance(candidate, prior) for prior in selected)
            scored.append((minimum_distance, -candidate["high_T_loss"], candidate))
        minimum_distance, _, chosen = max(scored, key=lambda item: (item[0], item[1]))
        selected.append(chosen)
        selection_details.append(
            {
                "selection_step": len(selected),
                "reason": "maximum minimum material distance within the high-transmission shortlist",
                "minimum_distance_to_previous": minimum_distance,
            }
        )

    output = {
        "selection_contract": {
            "priority_1": (
                "Minimize high_T_loss, which combines mean(1-Ts), mean(1-Tp), "
                "p95(1-Ts), p95(1-Tp), worst-polarization mean T, and minimum T."
            ),
            "priority_2": (
                "Within 10% of the best high_T_loss, maximize material composition "
                "distance using 0.65 normalized material-sequence Levenshtein distance "
                "+ 0.35 material-set Jaccard distance."
            ),
            "relative_high_T_tolerance": args.relative_tolerance,
            "best_high_T_loss": best_loss,
            "shortlist_threshold": threshold,
            "requested_candidates": raw["run_contract"]["requested_candidates"],
            "retained_tmm_candidates": raw["run_contract"]["retained_tmm_candidates"],
            "shortlist_size": len(shortlist),
            "shared_target_index": raw["shared_target"]["archived_validation_index"],
            "source_pool": str(args.input),
        },
        "shared_target": raw["shared_target"],
        "selected_candidates": [
            {**candidate, "selection": selection_details[index]}
            for index, candidate in enumerate(selected)
        ],
        "pairwise_material_distance": [
            {
                "left": left_index + 1,
                "right": right_index + 1,
                "distance": material_distance(selected[left_index], selected[right_index]),
            }
            for left_index in range(len(selected))
            for right_index in range(left_index + 1, len(selected))
        ],
    }
    args.output.write_text(json.dumps(output, indent=2))

    csv_path = args.output.with_name("figure4_high_transmission_diverse_spectra.csv")
    wavelengths = raw["run_contract"]["wavelengths_nm"]
    target = raw["shared_target"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "wavelength_nm",
                "target_Rs", "target_Ts", "target_Rp", "target_Tp",
                *[
                    f"candidate_{rank}_{quantity}"
                    for rank in range(1, args.count + 1)
                    for quantity in ("Rs", "Ts", "Rp", "Tp")
                ],
            ]
        )
        for index, wavelength in enumerate(wavelengths):
            row = [
                wavelength,
                target["Rs"][index], target["Ts"][index],
                target["Rp"][index], target["Tp"][index],
            ]
            for candidate in selected:
                row.extend(
                    [
                        candidate["sim_Rs"][index], candidate["sim_Ts"][index],
                        candidate["sim_Rp"][index], candidate["sim_Tp"][index],
                    ]
                )
            writer.writerow(row)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "csv": str(csv_path),
                "best_high_T_loss": best_loss,
                "threshold": threshold,
                "shortlist_size": len(shortlist),
                "selected": [
                    {
                        "tokens": candidate["tokens"],
                        "high_T_loss": candidate["high_T_loss"],
                        "mean_Ts": candidate["mean_Ts"],
                        "mean_Tp": candidate["mean_Tp"],
                        "p05_Ts": candidate["p05_Ts"],
                        "p05_Tp": candidate["p05_Tp"],
                    }
                    for candidate in selected
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
