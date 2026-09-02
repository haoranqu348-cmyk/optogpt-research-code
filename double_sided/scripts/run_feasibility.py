"""Run material/layer-stage feasibility before any large-scale retraining."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from double_sided.config import BASE_MATERIALS, DoubleSidedConfig
from double_sided.search import TMMBudget, export_feasibility, run_stage
from optogpt.core.datasets.sim import load_materials


def load_all_nk(project_root, generated_nk, material_sets, config):
    names = sorted({name for values in material_sets.values() for name in values})
    base_names = [name for name in names if name in BASE_MATERIALS]
    nk = load_materials(
        all_mats=[config.substrate, *base_names],
        wavelengths=config.wavelengths_nm / 1000.0,
        DATABASE=str(project_root / "optogpt" / "nk"),
    )
    for name in names:
        if name in nk:
            continue
        table = pd.read_csv(Path(generated_nk) / f"{name}.csv")
        if not np.array_equal(table["nm"].to_numpy(dtype=float), config.wavelengths_nm):
            raise ValueError(f"Generated n/k grid mismatch for {name}")
        nk[name] = table["n"].to_numpy() + 1j * table["k"].to_numpy()
    return nk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-nk", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--budget", type=int, default=300000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=3)
    parser.add_argument("--popsize", type=int, default=2)
    parser.add_argument("--stages", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    project_root = Path(__file__).resolve().parents[2]
    config = DoubleSidedConfig().validate()
    material_sets = {
        "base": list(BASE_MATERIALS),
        "base_plus_Nb2O5": [*BASE_MATERIALS, "Nb2O5"],
        "base_plus_Sc2O3": [*BASE_MATERIALS, "Sc2O3"],
        "base_plus_all_approved_new": [*BASE_MATERIALS, "Nb2O5", "Sc2O3"],
    }
    nk = load_all_nk(project_root, args.generated_nk, material_sets, config)
    budget = TMMBudget(args.budget)
    rng = np.random.RandomState(args.seed)
    started = time.time()
    rows, stage_summary = [], []
    for label, materials in material_sets.items():
        best_values = []
        for stage in config.layer_stages[:args.stages]:
            candidates = run_stage(
                materials, stage, nk, config, budget, rng, trials=args.trials,
                maxiter=args.maxiter, popsize=args.popsize,
            )
            for candidate in candidates:
                candidate["material_ablation"] = label
            rows.extend(candidates)
            best_values.append(candidates[0]["metrics"]["objective"])
            stage_summary.append({
                "material_ablation": label, "stage": [stage.minimum, stage.maximum],
                "best_objective": best_values[-1], "cumulative_tmm_calls": budget.calls,
                "stop_after_stage": config.should_stop(best_values),
            })
            if config.should_stop(best_values):
                break
    metadata = {
        "seed": args.seed, "truth_backend": "tmm.inc_tmm 71-point independent final recompute",
        "search_grid": "every fifth wavelength only during DE",
        "tmm_budget": args.budget, "tmm_calls_used": budget.calls,
        "elapsed_seconds": time.time() - started, "stage_summary": stage_summary,
        "material_gate": "only approved_physics_search materials with numeric thin-film n/k",
        "training_started": False,
    }
    export_feasibility(output, rows, metadata, config.wavelengths_nm)
    shutil.copy2(project_root / "double_sided" / "materials" / "material_audit.yaml",
                 output / "material_audit.yaml")
    shutil.copy2(Path(args.generated_nk) / "nk_manifest.json", output / "nk_manifest.json")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
