"""Budgeted direct-DE feasibility search with full-grid inc_tmm final truth."""

import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from .config import DoubleSidedConfig
from .contract import DoubleSidedStructure, Layer
from .physics import simulate_c, summarize


class TMMBudget:
    def __init__(self, maximum_calls):
        self.maximum_calls = int(maximum_calls)
        self.calls = 0

    def charge(self, calls):
        if self.calls + calls > self.maximum_calls:
            raise RuntimeError("TMM call budget exhausted")
        self.calls += calls


def evaluate(structure, nk_dict, config, budget, require_truth_grid=True):
    budget.charge(2 * len(config.wavelengths_nm))
    result = simulate_c(structure, nk_dict, config, require_truth_grid=require_truth_grid)
    return summarize(result), result


def random_material_sequence(rng, materials, count, nk_dict, alternating_probability=0.75):
    center = len(next(iter(nk_dict.values()))) // 2
    ordered = sorted(materials, key=lambda material: np.real(nk_dict[material][center]))
    split = max(1, len(ordered) // 2)
    low, high = ordered[:split], ordered[split:]
    sequence = []
    for index in range(count):
        pool = low if index % 2 == 0 else high
        if rng.rand() > alternating_probability:
            pool = list(materials)
        material = pool[rng.randint(len(pool))]
        if sequence and material == sequence[-1]:
            alternatives = [item for item in pool if item != sequence[-1]]
            if alternatives:
                material = alternatives[rng.randint(len(alternatives))]
        sequence.append(material)
    return sequence


def optimize_material_sequence(front_materials, back_materials, search_nk_dict, truth_nk_dict,
                               search_config, truth_config, budget, rng,
                               maxiter=4, popsize=3, truth_require_truth_grid=True):
    dimensions = len(front_materials) + len(back_materials)
    bounds = [(search_config.min_thickness_nm, search_config.max_thickness_nm)] * dimensions

    def structure_from(values):
        front = tuple(Layer(material, float(value))
                      for material, value in zip(front_materials, values[:len(front_materials)]))
        back = tuple(Layer(material, float(value))
                     for material, value in zip(back_materials, values[len(front_materials):]))
        return DoubleSidedStructure(front, back)

    def objective(values):
        try:
            metrics, _ = evaluate(
                structure_from(values), search_nk_dict, search_config, budget,
                require_truth_grid=False,
            )
            return metrics["objective"]
        except RuntimeError as exc:
            if "budget exhausted" in str(exc):
                raise
            return 1e3

    result = differential_evolution(
        objective, bounds, seed=int(rng.randint(2 ** 31 - 1)), maxiter=maxiter,
        popsize=popsize, polish=False, workers=1, updating="immediate", tol=1e-3,
    )
    structure = structure_from(result.x).merged()
    metrics, spectrum = evaluate(
        structure, truth_nk_dict, truth_config, budget,
        require_truth_grid=truth_require_truth_grid,
    )
    return {"structure": structure, "metrics": metrics, "spectrum": spectrum,
            "search_objective": float(result.fun), "n_parameters": dimensions}


def run_stage(materials, stage, nk_dict, config, budget, rng, trials=4,
              maxiter=4, popsize=3, search_stride=5):
    search_config = replace(config, wavelengths_nm=np.asarray(config.wavelengths_nm)[::search_stride])
    search_nk = {material: np.asarray(values)[::search_stride] for material, values in nk_dict.items()}
    candidates = []
    for trial in range(trials):
        count = stage.maximum if trial < max(1, trials // 2) else int(rng.randint(stage.minimum, stage.maximum + 1))
        front = random_material_sequence(rng, materials, count, nk_dict)
        back = random_material_sequence(rng, materials, count, nk_dict)
        before = budget.calls
        candidate = optimize_material_sequence(
            front, back, search_nk, nk_dict, search_config, config, budget, rng,
            maxiter, popsize
        )
        candidate.update({
            "trial": trial, "stage": [stage.minimum, stage.maximum],
            "material_set": list(materials), "tmm_calls": budget.calls - before,
        })
        candidates.append(candidate)
    return sorted(candidates, key=lambda item: item["metrics"]["objective"])


def export_feasibility(output_dir, rows, metadata, wavelengths_nm):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    serializable = []
    for rank, row in enumerate(sorted(rows, key=lambda item: item["metrics"]["objective"]), 1):
        structure = row["structure"]
        serializable.append({
            "rank": rank, "material_ablation": row["material_ablation"],
            "stage": row["stage"], "front": structure.canonical_payload()["front"],
            "back": structure.canonical_payload()["back"],
            "front_physical_layers": len(structure.front),
            "back_physical_layers": len(structure.back),
            "n_parameters": row["n_parameters"], "tmm_calls": row["tmm_calls"],
            **row["metrics"],
        })
    fields = list(serializable[0])
    with (output / "feasibility_rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(serializable)
    with (output / "feasibility_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(metadata, rankings=serializable), handle, indent=2)
    best = min(rows, key=lambda item: item["metrics"]["objective"])
    with (output / "best_spectrum.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength_nm", "Rs", "Ts", "As", "Rp", "Tp", "Ap"])
        for index, wavelength in enumerate(wavelengths_nm):
            writer.writerow([wavelength, *[
                best["spectrum"][pol][key][index]
                for pol in ("s", "p") for key in ("R", "T", "A")
            ]])
    return serializable
