"""Deduplicate, split, and consolidate resumable formal-data chunks."""

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


def atomic_json(data, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    root = Path(args.data_dir).resolve()
    contract = json.loads((root / "generation_contract.json").read_text(encoding="utf-8"))
    if contract.get("status") not in ("chunks_complete", "complete"):
        raise RuntimeError("All chunks must complete before finalization")
    chunk_metadata = sorted((root / "chunks").glob("chunk_*.json"))
    if not chunk_metadata:
        raise FileNotFoundError("No chunk metadata found")
    split_handles = {
        split: (root / f"structures_{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "dev", "test")
    }
    split_arrays = {split: {key: [] for key in ("A", "B", "C")}
                    for split in split_handles}
    seen_physical, group_splits = set(), {}
    counts, sources, raw_count = Counter(), Counter(), 0
    requested_stage = tuple(int(value) for value in contract["stage_layers_per_side"])
    physical_layer_counts = []
    try:
        for metadata_path in chunk_metadata:
            prefix = metadata_path.with_suffix("")
            records = [json.loads(line) for line in prefix.with_suffix(".jsonl").read_text().splitlines()]
            spectra = np.load(prefix.with_suffix(".npz"))
            if any(len(spectra[key]) != len(records) for key in ("A", "B", "C")):
                raise RuntimeError(f"Chunk length mismatch: {prefix.name}")
            selected = {split: [] for split in split_handles}
            for index, record in enumerate(records):
                raw_count += 1
                front_physical = int(record["front_layers_physical"])
                back_physical = int(record["back_layers_physical"])
                if not (requested_stage[0] <= front_physical <= requested_stage[1] and
                        requested_stage[0] <= back_physical <= requested_stage[1]):
                    raise RuntimeError(
                        f"{prefix.name}[{index}] physical layer counts "
                        f"{front_physical}/{back_physical} violate stage {requested_stage}"
                    )
                if record["physical_hash"] in seen_physical:
                    continue
                seen_physical.add(record["physical_hash"])
                group = record["split_group_hash"]
                split = group_splits.setdefault(group, record["split"])
                if split != record["split"]:
                    raise RuntimeError("Split-group assignment conflict")
                selected[split].append(index)
                counts[split] += 1
                sources[record["source_family"]] += 1
                physical_layer_counts.extend((front_physical, back_physical))
                split_handles[split].write(json.dumps(record, sort_keys=True) + "\n")
            for split, indices in selected.items():
                if indices:
                    for key in ("A", "B", "C"):
                        split_arrays[split][key].append(spectra[key][indices])
    finally:
        for handle in split_handles.values():
            handle.close()
    for split in split_arrays:
        values = {
            key: np.concatenate(chunks, axis=0) if chunks else np.empty((0, 284), np.float32)
            for key, chunks in split_arrays[split].items()
        }
        np.savez_compressed(root / f"spectra_ABC_{split}.npz", **values)
        if len(values["C"]) != counts[split]:
            raise RuntimeError(f"Finalized {split} count mismatch")
    sets = {split: set() for split in ("train", "dev", "test")}
    for split in ("train", "dev", "test"):
        with (root / f"structures_{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                sets[split].add(json.loads(line)["split_group_hash"])
    if sets["train"] & sets["dev"] or sets["train"] & sets["test"] or sets["dev"] & sets["test"]:
        raise RuntimeError("Split-group leakage detected after finalization")
    count_map = {split: int(counts[split]) for split in ("train", "dev", "test")}
    summary = {
        "schema_version": 1, "raw_samples": raw_count,
        "unique_physical_samples": sum(count_map.values()),
        "duplicates_removed": raw_count - sum(count_map.values()),
        "counts": count_map, "source_counts": dict(sources),
        "requested_layers_per_side": list(requested_stage),
        "observed_physical_layers_per_side": {
            "minimum": min(physical_layer_counts), "maximum": max(physical_layer_counts),
        },
        "split_leakage_groups": 0, "finalized_at": datetime.now().isoformat(),
        "spectrum_layout": ["Rs(71)", "Ts(71)", "Rp(71)", "Tp(71)"],
        "auxiliary_labels": ["A", "B", "C"], "truth_backend": "tmm.inc_tmm",
    }
    atomic_json(summary, root / "dataset_contract.json")
    contract["status"] = "complete"
    contract["final_summary"] = summary
    atomic_json(contract, root / "generation_contract.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
