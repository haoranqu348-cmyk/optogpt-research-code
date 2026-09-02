"""Compare model, one-sided composition, random, and direct DE under one TMM budget."""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from double_sided.config import BASE_MATERIALS, DoubleSidedConfig
from double_sided.contract import DoubleSidedStructure, Layer
from double_sided.data import sample_random_structure
from double_sided.physics import simulate_c, summarize
from double_sided.search import TMMBudget, optimize_material_sequence, random_material_sequence
from joint_sp.decoder import batch_greedy_decode_sp, batch_sampling_decode_sp, build_joint_logits_mask
from joint_sp.model import load_joint_sp_checkpoint
from optogpt.core.datasets.sim import load_materials


def hard_target():
    n = 71
    return np.concatenate([np.zeros(n), np.ones(n), np.zeros(n), np.ones(n)]).astype(np.float32)


def evaluate_candidate(structure, nk_dict, config, budget):
    budget.charge(2 * len(config.wavelengths_nm))
    structure = structure.merged()
    return {"structure": structure, "metrics": summarize(simulate_c(structure, nk_dict, config))}


def parse_layers(tokens):
    layers = []
    for token in tokens:
        material, thickness = token.rsplit("_", 1)
        layers.append(Layer(material, float(thickness)))
    return tuple(layers)


def random_baseline(nk_dict, config, call_limit, seed):
    budget, rng, rows = TMMBudget(call_limit), np.random.RandomState(seed), []
    while budget.maximum_calls - budget.calls >= 142:
        structure = sample_random_structure(
            rng, BASE_MATERIALS, (1, 4), (10, 500), "random", nk_dict, 10,
        )
        rows.append(evaluate_candidate(structure, nk_dict, config, budget))
    rows.sort(key=lambda row: row["metrics"]["objective"])
    return rows, budget.calls


def direct_de_baseline(nk_dict, config, call_limit, seed, maxiter, popsize):
    budget, rng, rows = TMMBudget(call_limit), np.random.RandomState(seed), []
    stride = 5
    search_config = replace(config, wavelengths_nm=config.wavelengths_nm[::stride])
    search_nk = {material: np.asarray(values)[::stride] for material, values in nk_dict.items()}
    # Conservative allowance for one DE run; remaining budget is filled by random truth evaluations.
    while budget.maximum_calls - budget.calls >= 8_000:
        front_count, back_count = int(rng.randint(1, 5)), int(rng.randint(1, 5))
        front = random_material_sequence(rng, BASE_MATERIALS, front_count, nk_dict)
        back = random_material_sequence(rng, BASE_MATERIALS, back_count, nk_dict)
        try:
            optimized = optimize_material_sequence(
                front, back, search_nk, nk_dict, search_config, config, budget, rng,
                maxiter=maxiter, popsize=popsize,
            )
        except RuntimeError as exc:
            if "budget exhausted" in str(exc):
                break
            raise
        rows.append({"structure": optimized["structure"], "metrics": optimized["metrics"]})
    while budget.maximum_calls - budget.calls >= 142:
        structure = sample_random_structure(
            rng, BASE_MATERIALS, (1, 4), (10, 500), "alternating", nk_dict, 10,
        )
        rows.append(evaluate_candidate(structure, nk_dict, config, budget))
    rows.sort(key=lambda row: row["metrics"]["objective"])
    return rows, budget.calls


def one_sided_composition(checkpoint, nk_dict, config, call_limit, seed, device):
    model, word_dict, index_dict, configs = load_joint_sp_checkpoint(checkpoint, device=device)
    logits_mask, _ = build_joint_logits_mask(word_dict, BASE_MATERIALS)
    target = hard_target()
    requested = max(2, 2 * (call_limit // 142))
    torch.manual_seed(seed)
    designs = batch_sampling_decode_sp(
        model, np.repeat(target[None, :], requested, axis=0), word_dict, index_dict,
        max_len=6, top_k=32, top_p=0.9, temperature=0.9,
        device=device, decode_batch_size=32, logits_mask=logits_mask,
    )
    greedy = batch_greedy_decode_sp(
        model, [target], word_dict, index_dict, max_len=6, device=device,
        decode_batch_size=1, logits_mask=logits_mask,
    )[0]
    designs.insert(0, greedy)
    valid = [tokens for tokens in designs if tokens and all(t.rsplit("_", 1)[0] in BASE_MATERIALS for t in tokens)]
    structures, seen = [], set()
    for index in range(0, len(valid) - 1, 2):
        front = parse_layers(valid[index])
        # Each one-sided result is air->glass; reverse it for glass->air traversal on the back.
        back = tuple(reversed(parse_layers(valid[index + 1])))
        structure = DoubleSidedStructure(front, back).merged()
        if structure.physical_hash() not in seen:
            seen.add(structure.physical_hash()); structures.append(structure)
    budget, rows = TMMBudget(call_limit), []
    for structure in structures:
        if budget.maximum_calls - budget.calls < 142:
            break
        rows.append(evaluate_candidate(structure, nk_dict, config, budget))
    rows.sort(key=lambda row: row["metrics"]["objective"])
    return rows, budget.calls, {"decoded": len(designs), "valid": len(valid), "unique_composed": len(structures)}


def serialize(method, row, rank):
    structure = row["structure"]
    return {
        "method": method, "rank": rank,
        "front": "/".join(f"{x.material}_{x.thickness_nm:g}" for x in structure.front),
        "back": "/".join(f"{x.material}_{x.thickness_nm:g}" for x in structure.back),
        "front_layers": len(structure.front), "back_layers": len(structure.back),
        **row["metrics"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-evaluation-dir", required=True)
    parser.add_argument("--joint-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--de-maxiter", type=int, default=5)
    parser.add_argument("--de-popsize", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    evaluation = Path(args.model_evaluation_dir)
    model_manifest = json.loads((evaluation / "manifest.json").read_text())
    call_limit = int(model_manifest["total_tmm_calls"])
    if call_limit < 142:
        raise ValueError("Model evaluation TMM budget is empty")
    config = DoubleSidedConfig().validate()
    root = Path(__file__).resolve().parents[2]
    nk_dict = load_materials(
        all_mats=[config.substrate, *BASE_MATERIALS], wavelengths=config.wavelengths_nm / 1000.0,
        DATABASE=str(root / "optogpt" / "nk"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_rows, random_calls = random_baseline(nk_dict, config, call_limit, args.seed)
    de_rows, de_calls = direct_de_baseline(
        nk_dict, config, call_limit, args.seed, args.de_maxiter, args.de_popsize
    )
    one_rows, one_calls, one_meta = one_sided_composition(
        args.joint_checkpoint, nk_dict, config, call_limit, args.seed, device
    )
    model_rows = []
    with (evaluation / "rankings.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "model_tmm_topk_de":
                model_rows.append(row)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    serialized = []
    for method, rows in (("random", random_rows), ("direct_de", de_rows),
                         ("one_sided_independent", one_rows)):
        serialized.extend(serialize(method, row, rank) for rank, row in enumerate(rows[:100], 1))
    with (output / "baseline_rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serialized[0]))
        writer.writeheader(); writer.writerows(serialized)
    def best(rows):
        return rows[0]["metrics"] if rows else None
    summary = {
        "equal_tmm_call_limit": call_limit,
        "model_best_de": model_rows[0] if model_rows else None,
        "random": {"calls": random_calls, "best": best(random_rows)},
        "direct_de": {"calls": de_calls, "best": best(de_rows)},
        "one_sided_independent": {"calls": one_calls, "best": best(one_rows), **one_meta},
        "historical_formal_v2_reference": {"objective": 0.10613348527109212,
                                            "budget_status": "unknown_not_in_equal_budget_ranking"},
    }
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
