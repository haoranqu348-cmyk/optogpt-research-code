"""Merge finalized stage datasets without physical or split-group leakage."""

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from double_sided.contract import assign_split


SPLITS = ("train", "dev", "test")
LABELS = ("A", "B", "C")


@dataclass(frozen=True)
class SourceRef:
    dataset: int
    split: str
    index: int


def atomic_json(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def read_rows(root, split):
    structure_path = root / f"structures_{split}.jsonl"
    spectrum_path = root / f"spectra_ABC_{split}.npz"
    with structure_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    with np.load(spectrum_path) as spectra:
        shapes = {label: spectra[label].shape for label in LABELS}
    if any(shape != (len(rows), 284) for shape in shapes.values()):
        raise ValueError(f"{root}: {split} invalid A/B/C shapes {shapes}")
    return rows


def compare_duplicate_spectra(roots, first, duplicate, tolerance):
    first_path = roots[first.dataset] / f"spectra_ABC_{first.split}.npz"
    duplicate_path = roots[duplicate.dataset] / f"spectra_ABC_{duplicate.split}.npz"
    maximum = 0.0
    with np.load(first_path) as left, np.load(duplicate_path) as right:
        for label in LABELS:
            left_row = np.asarray(left[label][first.index], dtype=np.float32)
            right_row = np.asarray(right[label][duplicate.index], dtype=np.float32)
            maximum = max(maximum, float(np.max(np.abs(left_row - right_row))))
            if not np.allclose(left_row, right_row, rtol=tolerance, atol=tolerance):
                raise RuntimeError(
                    f"Duplicate physical structure has inconsistent {label} spectra: "
                    f"{first_path}[{first.index}] vs {duplicate_path}[{duplicate.index}]"
                )
    return maximum


def write_split(output, split, selected, roots):
    rows = [item[0] for item in selected]
    with (output / f"structures_{split}.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    arrays = {label: np.empty((len(selected), 284), dtype=np.float32) for label in LABELS}
    source_groups = defaultdict(list)
    for destination_index, (_, source) in enumerate(selected):
        source_groups[(source.dataset, source.split)].append((destination_index, source.index))
    for (dataset_index, source_split), mappings in source_groups.items():
        source_path = roots[dataset_index] / f"spectra_ABC_{source_split}.npz"
        destination_indices = np.asarray([item[0] for item in mappings], dtype=np.int64)
        source_indices = np.asarray([item[1] for item in mappings], dtype=np.int64)
        with np.load(source_path) as spectra:
            for label in LABELS:
                arrays[label][destination_indices] = spectra[label][source_indices]
    np.savez_compressed(output / f"spectra_ABC_{split}.npz", **arrays)


def merge_datasets(base, extras, output, duplicate_tolerance=2e-6):
    roots = [Path(base).resolve(), *[Path(path).resolve() for path in extras]]
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    seen_physical = {}
    selected = {split: [] for split in SPLITS}
    source_counts = Counter()
    duplicates = []
    observed_raw = []
    observed_physical = []
    source_dataset_counts = Counter()
    raw_count = 0

    for dataset_index, root in enumerate(roots):
        for source_split in SPLITS:
            for source_index, row in enumerate(read_rows(root, source_split)):
                raw_count += 1
                expected_split = assign_split(row["split_group_hash"])
                if row.get("split", source_split) != source_split or source_split != expected_split:
                    raise RuntimeError(
                        f"Invalid split assignment in {root}: {source_split}[{source_index}] "
                        f"must be {expected_split}"
                    )
                source = SourceRef(dataset_index, source_split, source_index)
                physical_hash = row["physical_hash"]
                if physical_hash in seen_physical:
                    duplicates.append((seen_physical[physical_hash], source))
                    continue
                seen_physical[physical_hash] = source
                normalized = dict(row, split=expected_split)
                selected[expected_split].append((normalized, source))
                source_dataset_counts[str(root)] += 1
                source_counts[normalized.get("source_family", "unknown")] += 1
                observed_raw.extend([
                    int(normalized["front_layers_raw"]), int(normalized["back_layers_raw"])
                ])
                observed_physical.extend([
                    int(normalized["front_layers_physical"]),
                    int(normalized["back_layers_physical"]),
                ])

    maximum_duplicate_difference = 0.0
    for first, duplicate in duplicates:
        maximum_duplicate_difference = max(
            maximum_duplicate_difference,
            compare_duplicate_spectra(roots, first, duplicate, duplicate_tolerance),
        )
    for split in SPLITS:
        write_split(output, split, selected[split], roots)

    group_sets = {
        split: {row["split_group_hash"] for row, _ in selected[split]} for split in SPLITS
    }
    if (group_sets["train"] & group_sets["dev"] or
            group_sets["train"] & group_sets["test"] or
            group_sets["dev"] & group_sets["test"]):
        raise RuntimeError("Split-group leakage detected after merge")
    counts = {split: len(selected[split]) for split in SPLITS}
    contract = {
        "schema_version": 2,
        "source_datasets": [str(root) for root in roots],
        "unique_samples_by_source_dataset": dict(source_dataset_counts),
        "raw_input_samples": raw_count,
        "counts": counts,
        "unique_physical_samples": sum(counts.values()),
        "duplicates_removed": len(duplicates),
        "duplicate_spectrum_tolerance": duplicate_tolerance,
        "maximum_duplicate_spectrum_difference": maximum_duplicate_difference,
        "source_counts": dict(source_counts),
        "observed_layers_per_side": {
            "raw_minimum": min(observed_raw), "raw_maximum": max(observed_raw),
            "physical_minimum": min(observed_physical),
            "physical_maximum": max(observed_physical),
        },
        "split_leakage_groups": 0,
        "split_policy": "deterministic assign_split(split_group_hash)",
        "spectrum_layout": ["Rs(71)", "Ts(71)", "Rp(71)", "Tp(71)"],
        "auxiliary_labels": list(LABELS),
        "truth_backend": "tmm.inc_tmm",
        "merge_policy": "base-first physical_hash deduplication with A/B/C equivalence check",
    }
    atomic_json(contract, output / "dataset_contract.json")
    return contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--extra", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duplicate-tolerance", type=float, default=2e-6)
    args = parser.parse_args()
    if args.duplicate_tolerance < 0:
        raise ValueError("duplicate-tolerance must be non-negative")
    contract = merge_datasets(
        args.base, args.extra, args.output, duplicate_tolerance=args.duplicate_tolerance
    )
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
