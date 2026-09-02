"""Generate, TMM-rank, and optionally DE-polish double-sided model candidates."""

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from double_sided.config import DoubleSidedConfig
from double_sided.decoder import constrained_decode
from double_sided.model import load_double_sided_checkpoint
from double_sided.physics import simulate_abc, simulate_c, summarize, verify_merge_equivalence
from double_sided.search import TMMBudget, optimize_material_sequence
from optogpt.core.datasets.sim import load_materials


def top_k_sampler(temperature, top_k):
    def sample(logits):
        scaled = logits / temperature
        k = min(top_k, scaled.size(-1))
        values, indices = torch.topk(scaled, k=k, dim=-1)
        probabilities = torch.softmax(values, dim=-1)
        selected = torch.multinomial(probabilities, 1)
        return indices.gather(1, selected).squeeze(1)
    return sample


def target_spectrum():
    n = 71
    return np.concatenate([np.zeros(n), np.ones(n), np.zeros(n), np.ones(n)]).astype(np.float32)


def evaluate_structures(structures, nk_dict, config):
    rows = []
    for structure in structures:
        merged = structure.merged()
        result = simulate_c(merged, nk_dict, config)
        rows.append({
            "structure": merged, "metrics": summarize(result), "spectrum": result,
            "merge_equivalence": verify_merge_equivalence(structure, nk_dict, config),
            "tmm_calls": 2 * len(config.wavelengths_nm),
        })
    return sorted(rows, key=lambda row: row["metrics"]["objective"])


def serialize_row(row, rank, method):
    structure = row["structure"]
    return {
        "rank": rank, "method": method,
        "front_tokens": "/".join(f"{x.material}_{x.thickness_nm:g}" for x in structure.front),
        "back_tokens": "/".join(f"{x.material}_{x.thickness_nm:g}" for x in structure.back),
        "front_physical_layers": len(structure.front), "back_physical_layers": len(structure.back),
        "physical_hash": structure.physical_hash(), "tmm_calls": row["tmm_calls"],
        **row["metrics"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--de-top", type=int, default=10)
    parser.add_argument("--de-maxiter", type=int, default=20)
    parser.add_argument("--de-popsize", type=int, default=5)
    parser.add_argument("--max-layers-per-side", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    if args.candidates < 1 or args.decode_batch_size < 1 or args.de_top < 0:
        raise ValueError("Invalid candidate counts")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, word_dict, index_dict, configs, _ = load_double_sided_checkpoint(args.checkpoint, device)
    trained_maximum = configs.get("max_layers_per_side")
    if trained_maximum is not None and int(trained_maximum) != args.max_layers_per_side:
        raise ValueError(
            f"Evaluation max-layers-per-side={args.max_layers_per_side} does not match "
            f"checkpoint metadata {trained_maximum}"
        )
    allowed_materials = tuple(configs["allowed_materials"])
    config = DoubleSidedConfig(
        technical_max_layers_per_side=max(32, args.max_layers_per_side),
        allowed_materials=allowed_materials,
    ).validate()
    root = Path(__file__).resolve().parents[2]
    nk_dict = load_materials(
        all_mats=[config.substrate, *allowed_materials], wavelengths=config.wavelengths_nm / 1000.0,
        DATABASE=str(root / "optogpt" / "nk"),
    )
    target = target_spectrum()
    generated = []
    # Always retain the deterministic greedy result.
    generated.extend(constrained_decode(
        model, [target], word_dict, index_dict, allowed_materials,
        args.max_layers_per_side, device=device,
    ))
    sampler = top_k_sampler(args.temperature, args.top_k)
    remaining = max(0, args.candidates - 1)
    for start in range(0, remaining, args.decode_batch_size):
        count = min(args.decode_batch_size, remaining - start)
        generated.extend(constrained_decode(
            model, np.repeat(target[None, :], count, axis=0), word_dict, index_dict,
            allowed_materials, args.max_layers_per_side, sample_fn=sampler, device=device,
        ))
    unique = {}
    for structure in generated:
        merged = structure.merged()
        unique.setdefault(merged.physical_hash(), merged)
    started = time.time()
    ranked = evaluate_structures(list(unique.values()), nk_dict, config)

    de_rows = []
    de_tmm_calls = 0
    if args.de_top:
        stride = 5
        search_config = replace(config, wavelengths_nm=config.wavelengths_nm[::stride])
        search_nk = {material: np.asarray(values)[::stride] for material, values in nk_dict.items()}
        rng = np.random.RandomState(args.seed)
        # The budget is explicit even though this command is not comparing methods yet.
        budget = TMMBudget(10 ** 12)
        for row in ranked[:min(args.de_top, len(ranked))]:
            structure = row["structure"]
            before = budget.calls
            optimized = optimize_material_sequence(
                [x.material for x in structure.front], [x.material for x in structure.back],
                search_nk, nk_dict, search_config, config, budget, rng,
                maxiter=args.de_maxiter, popsize=args.de_popsize,
            )
            de_rows.append({
                "structure": optimized["structure"], "metrics": optimized["metrics"],
                "spectrum": optimized["spectrum"],
                "merge_equivalence": {"equivalent": True},
                "tmm_calls": budget.calls - before,
            })
            de_tmm_calls += budget.calls - before
        de_rows.sort(key=lambda row: row["metrics"]["objective"])

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    serialized = ([serialize_row(row, rank, "model_tmm_topk")
                   for rank, row in enumerate(ranked, 1)] +
                  [serialize_row(row, rank, "model_tmm_topk_de")
                   for rank, row in enumerate(de_rows, 1)])
    with (output / "rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serialized[0]))
        writer.writeheader(); writer.writerows(serialized)
    final_pool = {}
    for row in [*de_rows, *ranked]:
        key = row["structure"].physical_hash()
        previous = final_pool.get(key)
        if previous is None or row["metrics"]["objective"] < previous["metrics"]["objective"]:
            final_pool[key] = row
    final_ranked = sorted(final_pool.values(), key=lambda row: row["metrics"]["objective"])
    top = final_ranked[:20]
    top_abc = []
    spectrum_columns = []
    for rank, row in enumerate(top, 1):
        labels = simulate_abc(row["structure"], nk_dict, config)
        top_abc.append({
            "rank": rank, "tokens": row["structure"].to_tokens(),
            "A": summarize(labels["A"]), "B": summarize(labels["B"]), "C": summarize(labels["C"]),
        })
        spectrum_columns.append((rank, labels["C"]))
    (output / "top20_ABC.json").write_text(json.dumps(top_abc, indent=2))
    with (output / "top20_full_spectra.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        fields = ["wavelength_nm"]
        for rank, _ in spectrum_columns:
            fields += [f"rank{rank}_{key}{pol}" for pol in ("s", "p") for key in ("R", "T", "A")]
        writer.writerow(fields)
        for index, wavelength in enumerate(config.wavelengths_nm):
            values = [wavelength]
            for _, spectrum in spectrum_columns:
                values += [spectrum[pol][key][index]
                           for pol in ("s", "p") for key in ("R", "T", "A")]
            writer.writerow(values)
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "device": str(device),
        "requested_candidates": args.candidates, "valid_sequences": len(generated),
        "unique_physical_structures": len(unique), "legal_structure_rate": 1.0,
        "tmm_ranking_calls": 2 * 71 * len(unique),
        "de_candidates": len(de_rows), "de_tmm_calls": de_tmm_calls,
        "total_tmm_calls": 2 * 71 * len(unique) + de_tmm_calls,
        "elapsed_ranking_seconds": time.time() - started,
        "truth_backend": "tmm.inc_tmm 71 points", "strict_threshold_passes": sum(
            bool(row["metrics"]["passes_strict"]) for row in ranked
        ), "final_unique_ranked": len(final_ranked), "top20_exported": len(top),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
