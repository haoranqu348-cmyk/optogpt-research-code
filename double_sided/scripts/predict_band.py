"""Predict and optimize a double-sided coating for a selected wavelength band."""

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
from double_sided.physics import simulate_abc, simulate_c, summarize
from double_sided.search import TMMBudget, optimize_material_sequence
from optogpt.core.datasets.sim import load_materials


def parse_out_of_band_reflectances(value):
    values = []
    for item in value.split(","):
        reflectance = float(item.strip())
        if not 0.0 <= reflectance <= 1.0:
            raise ValueError("Out-of-band reflectances must lie in [0, 1]")
        if reflectance not in values:
            values.append(reflectance)
    if not values:
        raise ValueError("At least one out-of-band completion is required")
    return values


def build_target_spectrum(wavelengths_nm, wavelength_min, wavelength_max,
                          out_of_band_reflectance):
    """Keep the checkpoint's 284-value input while varying the unconstrained band."""
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    in_band = (wavelengths >= wavelength_min) & (wavelengths <= wavelength_max)
    reflectance = np.full(len(wavelengths), out_of_band_reflectance, dtype=np.float32)
    reflectance[in_band] = 0.0
    transmittance = 1.0 - reflectance
    return np.concatenate([
        reflectance, transmittance, reflectance, transmittance,
    ]).astype(np.float32)


def top_k_sampler(temperature, top_k):
    def sample(logits):
        values, indices = torch.topk(
            logits / temperature, k=min(top_k, logits.size(-1)), dim=-1
        )
        selected = torch.multinomial(torch.softmax(values, dim=-1), 1)
        return indices.gather(1, selected).squeeze(1)
    return sample


def evaluate_structures(structures, nk_dict, config):
    rows = []
    for structure in structures:
        result = simulate_c(structure, nk_dict, config, require_truth_grid=False)
        rows.append({
            "structure": structure,
            "metrics": summarize(result),
            "spectrum": result,
            "tmm_calls": 2 * len(config.wavelengths_nm),
        })
    return sorted(rows, key=lambda row: row["metrics"]["objective"])


def serialize_row(row, rank, method):
    structure = row["structure"]
    return {
        "rank": rank,
        "method": method,
        "front_tokens": "/".join(
            f"{layer.material}_{layer.thickness_nm:g}" for layer in structure.front
        ),
        "back_tokens": "/".join(
            f"{layer.material}_{layer.thickness_nm:g}" for layer in structure.back
        ),
        "front_physical_layers": len(structure.front),
        "back_physical_layers": len(structure.back),
        "physical_hash": structure.physical_hash(),
        "tmm_calls": row["tmm_calls"],
        **row["metrics"],
    }


def export_full_spectrum(path, wavelengths, labels, band_mask):
    with Path(path).open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        fields = ["wavelength_nm", "in_target_band"]
        for definition in ("A", "B", "C"):
            fields.extend(
                f"{definition}_{key}{polarization}"
                for polarization in ("s", "p") for key in ("R", "T", "A")
            )
        writer.writerow(fields)
        for index, wavelength in enumerate(wavelengths):
            values = [float(wavelength), bool(band_mask[index])]
            for definition in ("A", "B", "C"):
                values.extend(
                    float(labels[definition][polarization][key][index])
                    for polarization in ("s", "p") for key in ("R", "T", "A")
                )
            writer.writerow(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wavelength-min", type=float, default=400.0)
    parser.add_argument("--wavelength-max", type=float, default=800.0)
    parser.add_argument("--out-of-band-reflectances", default="0,0.05,0.15,0.30")
    parser.add_argument("--candidates", type=int, default=1024)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--de-top", type=int, default=32)
    parser.add_argument("--de-maxiter", type=int, default=50)
    parser.add_argument("--de-popsize", type=int, default=8)
    parser.add_argument("--de-search-stride", type=int, default=4)
    parser.add_argument("--max-layers-per-side", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    if not args.wavelength_min < args.wavelength_max:
        raise ValueError("wavelength-min must be smaller than wavelength-max")
    if args.candidates < 1 or args.decode_batch_size < 1 or args.de_top < 0:
        raise ValueError("Candidate counts are invalid")
    if args.de_search_stride < 1:
        raise ValueError("de-search-stride must be positive")
    out_of_band = parse_out_of_band_reflectances(args.out_of_band_reflectances)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, word_dict, index_dict, checkpoint_config, _ = load_double_sided_checkpoint(
        args.checkpoint, device
    )
    trained_maximum = checkpoint_config.get("max_layers_per_side")
    if trained_maximum is not None and int(trained_maximum) != args.max_layers_per_side:
        raise ValueError(
            f"Requested layer limit {args.max_layers_per_side} does not match checkpoint "
            f"metadata {trained_maximum}"
        )
    allowed_materials = tuple(checkpoint_config["allowed_materials"])
    full_config = DoubleSidedConfig(
        technical_max_layers_per_side=max(32, args.max_layers_per_side),
        allowed_materials=allowed_materials,
    ).validate()
    full_wavelengths = np.asarray(full_config.wavelengths_nm)
    band_mask = (
        (full_wavelengths >= args.wavelength_min) &
        (full_wavelengths <= args.wavelength_max)
    )
    if int(band_mask.sum()) < 2:
        raise ValueError("Selected band must contain at least two checkpoint wavelength points")
    band_config = replace(full_config, wavelengths_nm=full_wavelengths[band_mask])
    band_config.validate(require_truth_grid=False)
    root = Path(__file__).resolve().parents[2]
    full_nk = load_materials(
        all_mats=[full_config.substrate, *allowed_materials],
        wavelengths=full_wavelengths / 1000.0,
        DATABASE=str(root / "optogpt" / "nk"),
    )
    band_nk = {material: np.asarray(values)[band_mask] for material, values in full_nk.items()}
    targets = [
        build_target_spectrum(
            full_wavelengths, args.wavelength_min, args.wavelength_max, reflectance
        ) for reflectance in out_of_band
    ]

    generated_with_profile = []
    for profile_index, target in enumerate(targets[:args.candidates]):
        structure = constrained_decode(
            model, [target], word_dict, index_dict, allowed_materials,
            args.max_layers_per_side, device=device,
        )[0]
        generated_with_profile.append((structure, profile_index))
    sampler = top_k_sampler(args.temperature, args.top_k)
    remaining = max(0, args.candidates - len(generated_with_profile))
    profile_indices = [index % len(targets) for index in range(remaining)]
    for start in range(0, remaining, args.decode_batch_size):
        batch_profiles = profile_indices[start:start + args.decode_batch_size]
        batch_targets = np.asarray([targets[index] for index in batch_profiles])
        structures = constrained_decode(
            model, batch_targets, word_dict, index_dict, allowed_materials,
            args.max_layers_per_side, sample_fn=sampler, device=device,
        )
        generated_with_profile.extend(zip(structures, batch_profiles))

    unique = {}
    profile_sources = {}
    for structure, profile_index in generated_with_profile:
        merged = structure.merged()
        physical_hash = merged.physical_hash()
        unique.setdefault(physical_hash, merged)
        profile_sources.setdefault(physical_hash, set()).add(profile_index)
    started = time.time()
    ranked = evaluate_structures(list(unique.values()), band_nk, band_config)
    for row in ranked:
        row["method"] = "model_band_tmm"

    de_rows = []
    de_tmm_calls = 0
    if args.de_top:
        search_config = replace(
            band_config,
            wavelengths_nm=np.asarray(band_config.wavelengths_nm)[::args.de_search_stride],
        )
        search_nk = {
            material: np.asarray(values)[::args.de_search_stride]
            for material, values in band_nk.items()
        }
        rng = np.random.RandomState(args.seed)
        budget = TMMBudget(10 ** 12)
        for row in ranked[:min(args.de_top, len(ranked))]:
            structure = row["structure"]
            before = budget.calls
            optimized = optimize_material_sequence(
                [layer.material for layer in structure.front],
                [layer.material for layer in structure.back],
                search_nk, band_nk, search_config, band_config, budget, rng,
                maxiter=args.de_maxiter, popsize=args.de_popsize,
                truth_require_truth_grid=False,
            )
            de_rows.append({
                "structure": optimized["structure"],
                "metrics": optimized["metrics"],
                "spectrum": optimized["spectrum"],
                "tmm_calls": budget.calls - before,
                "method": "model_band_tmm_de",
            })
            de_tmm_calls += budget.calls - before
        de_rows.sort(key=lambda row: row["metrics"]["objective"])

    final_pool = {}
    for row in [*de_rows, *ranked]:
        physical_hash = row["structure"].physical_hash()
        previous = final_pool.get(physical_hash)
        if previous is None or row["metrics"]["objective"] < previous["metrics"]["objective"]:
            final_pool[physical_hash] = row
    final_ranked = sorted(final_pool.values(), key=lambda row: row["metrics"]["objective"])
    best = final_ranked[0]
    full_labels = simulate_abc(best["structure"], full_nk, full_config)
    band_labels = {
        definition: {
            polarization: {
                key: values[band_mask]
                for key, values in full_labels[definition][polarization].items()
            } for polarization in ("s", "p")
        } for definition in ("A", "B", "C")
    }
    full_metrics = {definition: summarize(full_labels[definition]) for definition in ("A", "B", "C")}
    band_metrics = {definition: summarize(band_labels[definition]) for definition in ("A", "B", "C")}

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    serialized = [
        serialize_row(row, rank, row["method"])
        for rank, row in enumerate(ranked, 1)
    ] + [
        serialize_row(row, rank, row["method"])
        for rank, row in enumerate(de_rows, 1)
    ]
    with (output / "rankings_400_800.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serialized[0]))
        writer.writeheader()
        writer.writerows(serialized)
    export_full_spectrum(
        output / "best_spectrum_400_1100.csv",
        full_wavelengths, full_labels, band_mask,
    )
    structure = best["structure"]
    best_payload = {
        "tokens": structure.to_tokens(),
        "front_physical_layers": len(structure.front),
        "back_physical_layers": len(structure.back),
        "target_band_nm": [args.wavelength_min, args.wavelength_max],
        "band_metrics_ABC": band_metrics,
        "full_400_1100_metrics_ABC": full_metrics,
    }
    (output / "best_structure.json").write_text(json.dumps(best_payload, indent=2))
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "model_input_contract": "284 values: Rs(71), Ts(71), Rp(71), Tp(71)",
        "target_band_nm": [args.wavelength_min, args.wavelength_max],
        "band_wavelength_points": int(band_mask.sum()),
        "out_of_band_reflectance_completions": out_of_band,
        "requested_candidates": args.candidates,
        "valid_sequences": len(generated_with_profile),
        "unique_physical_structures": len(unique),
        "legal_structure_rate": 1.0,
        "ranking_truth": "tmm.inc_tmm on selected band",
        "ranking_tmm_calls": 2 * int(band_mask.sum()) * len(unique),
        "de_candidates": len(de_rows),
        "de_tmm_calls": de_tmm_calls,
        "full_grid_audit_tmm_calls": 3 * 2 * len(full_wavelengths),
        "elapsed_seconds": time.time() - started,
        "best_band_objective": best["metrics"]["objective"],
        "best_method": best["method"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"manifest": manifest, "best": best_payload}, indent=2))


if __name__ == "__main__":
    main()
