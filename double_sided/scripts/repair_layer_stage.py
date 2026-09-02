"""Filter finalized data to its declared per-side layer stage without rerunning TMM."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-layers-per-side", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.data_dir).resolve()
    if args.max_layers_per_side < 1:
        raise ValueError("max-layers-per-side must be positive")
    counts, removed, source_counts = {}, {}, Counter()
    for split in ("train", "dev", "test"):
        structure_path = root / f"structures_{split}.jsonl"
        spectrum_path = root / f"spectra_ABC_{split}.npz"
        rows, keep, input_structure_count = [], [], 0
        with structure_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                input_structure_count += 1
                row = json.loads(line)
                valid = (int(row["front_layers_raw"]) <= args.max_layers_per_side and
                         int(row["back_layers_raw"]) <= args.max_layers_per_side)
                if valid:
                    rows.append(row); keep.append(index)
                    source_counts[row["source_family"]] += 1
        spectra = np.load(spectrum_path)
        original_count = len(spectra["C"])
        if any(len(spectra[key]) != original_count for key in ("A", "B", "C")):
            raise RuntimeError(f"{split} A/B/C count mismatch")
        if input_structure_count != original_count:
            raise RuntimeError(
                f"{split} structure/spectrum count mismatch: {input_structure_count} != {original_count}"
            )
        if keep and max(keep) >= original_count:
            raise RuntimeError(f"{split} structure index exceeds spectrum count")
        structure_tmp = structure_path.with_suffix(".jsonl.tmp")
        spectrum_tmp = spectrum_path.with_suffix(".npz.tmp.npz")
        with structure_tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        np.savez_compressed(spectrum_tmp, **{
            key: spectra[key][keep].astype(np.float32, copy=False) for key in ("A", "B", "C")
        })
        spectra.close()
        os.replace(structure_tmp, structure_path)
        os.replace(spectrum_tmp, spectrum_path)
        counts[split] = len(keep)
        removed[split] = original_count - len(keep)
    contract_path = root / "dataset_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["counts"] = counts
    contract["unique_physical_samples"] = sum(counts.values())
    contract["source_counts"] = dict(source_counts)
    contract["layer_stage_repair"] = {
        "maximum_layers_per_side": args.max_layers_per_side,
        "removed": removed,
        "removed_total": sum(removed.values()),
        "tmm_recomputed": False,
        "reason": "Filtered records retain their original independently computed A/B/C labels",
    }
    temporary = contract_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, contract_path)
    generation_path = root / "generation_contract.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation["final_summary"] = contract
    generation_tmp = generation_path.with_suffix(".json.tmp")
    generation_tmp.write_text(json.dumps(generation, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(generation_tmp, generation_path)
    print(json.dumps({"counts": counts, "removed": removed,
                      "removed_total": sum(removed.values())}, indent=2))


if __name__ == "__main__":
    main()
