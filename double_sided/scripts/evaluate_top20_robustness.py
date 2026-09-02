"""Run the complete robustness contract for model-evaluation Top-20 structures."""

import argparse
import csv
import json
from pathlib import Path

from double_sided.config import BASE_MATERIALS, DoubleSidedConfig
from double_sided.contract import DoubleSidedStructure
from double_sided.robustness import evaluate_robustness
from optogpt.core.datasets.sim import load_materials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-evaluation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-layers-per-side", type=int, required=True)
    args = parser.parse_args()
    evaluation = Path(args.model_evaluation_dir)
    records = json.loads((evaluation / "top20_ABC.json").read_text())
    if not records:
        raise ValueError("No Top-20 structures to evaluate")
    config = DoubleSidedConfig().validate()
    root = Path(__file__).resolve().parents[2]
    nk_dict = load_materials(
        all_mats=[config.substrate, *BASE_MATERIALS], wavelengths=config.wavelengths_nm / 1000.0,
        DATABASE=str(root / "optogpt" / "nk"),
    )
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    summary_rows, wavelength_rows, scenario_rows = [], [], []
    for record in records:
        rank = int(record["rank"])
        structure = DoubleSidedStructure.from_tokens(
            record["tokens"], BASE_MATERIALS, args.max_layers_per_side
        ).merged()
        scenarios, worst, worst_metrics = evaluate_robustness(
            structure, nk_dict, config, random_trials=args.random_trials,
            seed=args.seed + rank,
        )
        nominal = next(item["metrics"] for item in scenarios if item["scenario"] == "nominal")
        summary_rows.append({
            "rank": rank, "front_layers": len(structure.front), "back_layers": len(structure.back),
            "tokens": " ".join(structure.to_tokens()),
            **{f"nominal_{key}": value for key, value in nominal.items()},
            **{f"worst_envelope_{key}": value for key, value in worst_metrics.items()},
            "scenario_count": len(scenarios),
        })
        for scenario in scenarios:
            scenario_rows.append({"rank": rank, "scenario": scenario["scenario"], **scenario["metrics"]})
        for index, wavelength in enumerate(config.wavelengths_nm):
            wavelength_rows.append({
                "rank": rank, "wavelength_nm": wavelength,
                **{f"worst_{key}{pol}": worst[pol][key][index]
                   for pol in ("s", "p") for key in ("R", "T", "A")},
            })
    for name, rows in (("robustness_top20_summary.csv", summary_rows),
                       ("robustness_top20_scenarios.csv", scenario_rows),
                       ("robustness_top20_wavelength.csv", wavelength_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    manifest = {
        "structures_evaluated": len(records), "random_trials_per_structure": args.random_trials,
        "truth_backend": "tmm.inc_tmm 71 points",
        "component_worst_envelope_warning": "Rmax/Tmin/Amax can come from different scenarios",
        "total_scenarios": len(scenario_rows),
        "tmm_calls": len(scenario_rows) * 2 * len(config.wavelengths_nm),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
