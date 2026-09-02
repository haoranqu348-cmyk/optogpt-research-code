"""Evaluate incumbent robustness with full 71-point mixed-coherence truth."""

import argparse
import csv
import json
from pathlib import Path

from double_sided.config import DoubleSidedConfig
from double_sided.robustness import evaluate_robustness
from double_sided.scripts.run_elite_material_gate import INCUMBENT_BACK, INCUMBENT_FRONT
from double_sided.contract import DoubleSidedStructure
from optogpt.core.datasets.sim import load_materials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = DoubleSidedConfig().validate()
    root = Path(__file__).resolve().parents[2]
    materials = sorted({x.material for x in (*INCUMBENT_FRONT, *INCUMBENT_BACK)})
    nk = load_materials(
        all_mats=[config.substrate, *materials], wavelengths=config.wavelengths_nm / 1000.0,
        DATABASE=str(root / "optogpt" / "nk"),
    )
    scenarios, worst, worst_metrics = evaluate_robustness(
        DoubleSidedStructure(INCUMBENT_FRONT, INCUMBENT_BACK), nk, config,
        args.random_trials, args.seed,
    )
    rows = [{"scenario": item["scenario"], **item["metrics"]} for item in scenarios]
    with (output / "robustness_scenarios.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output / "wavelength_worst_case.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength_nm", "worst_Rs", "worst_Ts", "worst_As",
                         "worst_Rp", "worst_Tp", "worst_Ap"])
        for index, wavelength in enumerate(config.wavelengths_nm):
            writer.writerow([wavelength, *[
                worst[pol][key][index] for pol in ("s", "p") for key in ("R", "T", "A")
            ]])
    manifest = {
        "structure_source": "formal_v2 incumbent independently recomputed",
        "random_trials": args.random_trials, "seed": args.seed,
        "scenario_count": len(scenarios), "truth_backend": "tmm.inc_tmm",
        "worst_case_metrics": worst_metrics,
    }
    with (output / "robustness_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
