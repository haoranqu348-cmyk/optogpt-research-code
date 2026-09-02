"""Fair material gate seeded by the known formal_v2 incumbent structure."""

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from double_sided.config import BASE_MATERIALS, DoubleSidedConfig
from double_sided.contract import DoubleSidedStructure, Layer
from double_sided.scripts.run_feasibility import load_all_nk
from double_sided.search import TMMBudget, evaluate, optimize_material_sequence


INCUMBENT_FRONT = (
    Layer("MgF2", 143.85428742561706), Layer("Al2O3", 151.47795684123955),
    Layer("TiO2", 10.0), Layer("SiO2", 32.92028729866185),
)
INCUMBENT_BACK = (
    Layer("MgO", 169.26362712009282), Layer("Al2O3", 108.88118834206364),
    Layer("MgF2", 139.07442407286678),
)


def material_variants(new_materials):
    variants = [("incumbent", [x.material for x in INCUMBENT_FRONT],
                 [x.material for x in INCUMBENT_BACK])]
    for material in new_materials:
        for side_name, source_front, source_back in (
                ("front", INCUMBENT_FRONT, INCUMBENT_BACK),
                ("back", INCUMBENT_BACK, INCUMBENT_FRONT)):
            source = [layer.material for layer in source_front]
            other = [layer.material for layer in source_back]
            for index in range(len(source)):
                changed = list(source); changed[index] = material
                front, back = (changed, other) if side_name == "front" else (other, changed)
                variants.append((f"replace_{side_name}_{index}_{material}", front, back))
            for index in range(len(source) + 1):
                changed = list(source); changed.insert(index, material)
                front, back = (changed, other) if side_name == "front" else (other, changed)
                variants.append((f"insert_{side_name}_{index}_{material}", front, back))
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-nk", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=180000)
    parser.add_argument("--maxiter", type=int, default=4)
    parser.add_argument("--popsize", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    project_root = Path(__file__).resolve().parents[2]
    config = DoubleSidedConfig().validate()
    material_sets = {
        "base": list(BASE_MATERIALS),
        "base_plus_Nb2O5": [*BASE_MATERIALS, "Nb2O5"],
        "base_plus_Sc2O3": [*BASE_MATERIALS, "Sc2O3"],
        "base_plus_all_approved_new": [*BASE_MATERIALS, "Nb2O5", "Sc2O3"],
    }
    nk = load_all_nk(project_root, args.generated_nk, material_sets, config)
    stride = 5
    search_config = replace(config, wavelengths_nm=config.wavelengths_nm[::stride])
    search_nk = {material: np.asarray(values)[::stride] for material, values in nk.items()}
    budget, rng = TMMBudget(args.budget), np.random.RandomState(args.seed)
    started = time.time()
    rows = []

    incumbent = DoubleSidedStructure(INCUMBENT_FRONT, INCUMBENT_BACK)
    incumbent_metrics, _ = evaluate(incumbent, nk, config, budget)
    for label, materials in material_sets.items():
        new_materials = sorted(set(materials) - set(BASE_MATERIALS))
        rows.append({
            "material_ablation": label, "variant": "incumbent_unmodified",
            "front": incumbent.front, "back": incumbent.back,
            "metrics": incumbent_metrics, "n_parameters": 7, "tmm_calls": 142,
            "uses_new_material": False,
        })
        if not new_materials:
            continue
        for variant, front, back in material_variants(new_materials)[1:]:
            before = budget.calls
            candidate = optimize_material_sequence(
                front, back, search_nk, nk, search_config, config, budget, rng,
                maxiter=args.maxiter, popsize=args.popsize,
            )
            structure = candidate["structure"]
            rows.append({
                "material_ablation": label, "variant": variant,
                "front": structure.front, "back": structure.back,
                "metrics": candidate["metrics"],
                "n_parameters": candidate["n_parameters"],
                "tmm_calls": budget.calls - before,
                "uses_new_material": any(
                    layer.material in new_materials for layer in (*structure.front, *structure.back)
                ),
            })

    serialized = []
    for row in sorted(rows, key=lambda value: value["metrics"]["objective"]):
        serialized.append({
            "material_ablation": row["material_ablation"], "variant": row["variant"],
            "front": "/".join(f"{x.material}:{x.thickness_nm:.6f}" for x in row["front"]),
            "back": "/".join(f"{x.material}:{x.thickness_nm:.6f}" for x in row["back"]),
            "front_physical_layers": len(row["front"]), "back_physical_layers": len(row["back"]),
            "n_parameters": row["n_parameters"], "tmm_calls": row["tmm_calls"],
            "uses_new_material": row["uses_new_material"], **row["metrics"],
        })
    with (output / "elite_material_gate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serialized[0]))
        writer.writeheader(); writer.writerows(serialized)
    best_by_set = {}
    for label in material_sets:
        candidates = [row for row in serialized if row["material_ablation"] == label]
        best = min(candidates, key=lambda value: value["objective"])
        best_new = min((row for row in candidates if row["uses_new_material"]),
                       key=lambda value: value["objective"], default=None)
        best_by_set[label] = {"best": best, "best_using_new_material": best_new}
    manifest = {
        "seed": args.seed, "truth_backend": "tmm.inc_tmm on all 71 wavelengths",
        "incumbent_source": "results/double_sided_glass_study/20260728_60deg_formal_v2",
        "incumbent_objective_recomputed": incumbent_metrics["objective"],
        "tmm_budget": args.budget, "tmm_calls_used": budget.calls,
        "elapsed_seconds": time.time() - started,
        "comparison_rule": "expanded set includes the unchanged incumbent; improvement requires a new-material structure",
        "best_by_set": best_by_set,
        "training_gate_passed": any(
            value["best_using_new_material"] is not None
            and value["best_using_new_material"]["objective"] < incumbent_metrics["objective"] - 1e-3
            for label, value in best_by_set.items() if label != "base"
        ),
    }
    with (output / "elite_material_gate_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
