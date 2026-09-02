"""Leak-resistant, mixed-source double-sided dataset generation."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import DoubleSidedConfig
from .contract import DoubleSidedStructure, Layer, assign_split
from .physics import simulate_abc, spectrum_vector, summarize


SOURCE_FAMILIES = (
    "random", "alternating", "de_elite", "ga_elite", "optogpt_candidate", "active_hard"
)


def sample_random_structure(rng, materials, layer_range, thickness_bounds,
                            family="random", nk_dict=None, thickness_step_nm=10.0):
    minimum, maximum = layer_range
    front_count = int(rng.randint(minimum, maximum + 1))
    back_count = int(rng.randint(minimum, maximum + 1))
    low, high = thickness_bounds

    def side(count, offset):
        if family == "alternating":
            if nk_dict is None:
                raise ValueError("Alternating sampling requires nk_dict")
            center = len(next(iter(nk_dict.values()))) // 2
            ordered = sorted(materials, key=lambda material: np.real(nk_dict[material][center]))
            thirds = max(1, len(ordered) // 3)
            buckets = [ordered[:thirds], ordered[thirds:-thirds], ordered[-thirds:]]
            sequence = [buckets[(index + offset) % 3][rng.randint(len(buckets[(index + offset) % 3]))]
                        for index in range(count)]
        else:
            sequence = [materials[rng.randint(len(materials))] for _ in range(count)]
        grid = np.arange(low, high + thickness_step_nm / 2.0, thickness_step_nm)
        return tuple(Layer(material, float(grid[rng.randint(len(grid))])) for material in sequence)

    return DoubleSidedStructure(side(front_count, 0), side(back_count, 1))


def sample_record(structure, source_family, nk_dict, config):
    if source_family not in SOURCE_FAMILIES:
        raise ValueError(f"Unknown source family: {source_family}")
    labels = simulate_abc(structure, nk_dict, config)
    merged = structure.merged()
    record = {
        "tokens": structure.to_tokens(),
        "merged_tokens": merged.to_tokens(),
        "front_layers_raw": len(structure.front),
        "back_layers_raw": len(structure.back),
        "front_layers_physical": len(merged.front),
        "back_layers_physical": len(merged.back),
        "physical_hash": structure.physical_hash(),
        "split_group_hash": structure.split_group_hash(),
        "source_family": source_family,
        "metrics_A": summarize(labels["A"]),
        "metrics_B": summarize(labels["B"]),
        "metrics_C": summarize(labels["C"]),
    }
    spectra = {definition: spectrum_vector(labels[definition]) for definition in ("A", "B", "C")}
    return record, spectra


def write_dataset(records, spectra, output_dir, config, seed):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seen_physical, groups = set(), {}
    split_records = {split: [] for split in ("train", "dev", "test")}
    split_spectra = {split: {key: [] for key in ("A", "B", "C")} for split in split_records}
    for record, values in zip(records, spectra):
        if record["physical_hash"] in seen_physical:
            continue
        seen_physical.add(record["physical_hash"])
        group = record["split_group_hash"]
        split = groups.setdefault(group, assign_split(group))
        record = dict(record, split=split)
        split_records[split].append(record)
        for key in values:
            split_spectra[split][key].append(values[key])

    hash_manifest = []
    for split in split_records:
        jsonl_path = output / f"structures_{split}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in split_records[split]:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        arrays = {
            key: np.asarray(values, dtype=np.float32).reshape((-1, 284))
            for key, values in split_spectra[split].items()
        }
        np.savez_compressed(output / f"spectra_ABC_{split}.npz", **arrays)
        for record in split_records[split]:
            hash_manifest.append({
                "physical_hash": record["physical_hash"],
                "split_group_hash": record["split_group_hash"], "split": split,
            })
    group_sets = {
        split: {row["split_group_hash"] for row in hash_manifest if row["split"] == split}
        for split in split_records
    }
    if any(group_sets[a] & group_sets[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise RuntimeError("Split-group leakage detected")
    with (output / "hash_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(hash_manifest, handle, indent=2)
    contract = {
        "schema_version": 1,
        "token_contract": "BOS front... SIDE_SEP back... EOS",
        "spectrum_layout": ["Rs(71)", "Ts(71)", "Rp(71)", "Tp(71)"],
        "truth_backend": "tmm.inc_tmm",
        "coherence": "coherent films, incoherent 500 um glass",
        "auxiliary_labels": {"A": "front/semi-infinite glass", "B": "front/finite glass/air",
                             "C": "front/finite glass/back/air"},
        "wavelengths_nm": np.asarray(config.wavelengths_nm).tolist(),
        "angle_deg": config.angle_deg, "seed": int(seed),
        "token_thickness_step_nm": config.token_thickness_step_nm,
        "counts": {split: len(split_records[split]) for split in split_records},
        "source_families": list(SOURCE_FAMILIES),
        "deduplication": "merged physical hash; mirror-equivalent split group hash",
    }
    with (output / "dataset_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2)
    return contract
