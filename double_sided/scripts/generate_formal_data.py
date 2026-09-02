"""Resumable, multiprocessing bootstrap data generation for double-sided training."""

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from double_sided.config import BASE_MATERIALS, DoubleSidedConfig
from double_sided.contract import DoubleSidedStructure, Layer, assign_split, merge_adjacent
from double_sided.data import sample_random_structure, sample_record
from optogpt.core.datasets.sim import load_materials


_WORKER_NK = None
_WORKER_CONFIG = None
_WORKER_ELITES = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_elite_templates(path, maximum_layers):
    if not path:
        return []
    templates = []
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        if rows and "objective" in rows[0]:
            rows.sort(key=lambda row: float(row["objective"]))
        for row in rows[:100]:
            if "front_materials" in row:
                front_materials = row["front_materials"].split("/") if row["front_materials"] else []
                back_materials = row["back_materials"].split("/") if row["back_materials"] else []
                front_values = [float(value) for value in row["front_thicknesses_nm"].split("/")]
                back_values = [float(value) for value in row["back_thicknesses_nm"].split("/")]
            elif "front_tokens" in row:
                def parse_tokens(value):
                    parsed = [token.rsplit("_", 1) for token in value.split("/") if token]
                    return [item[0] for item in parsed], [float(item[1]) for item in parsed]
                front_materials, front_values = parse_tokens(row["front_tokens"])
                back_materials, back_values = parse_tokens(row["back_tokens"])
            else:
                raise ValueError("Unsupported elite CSV schema")
            if not front_materials or not back_materials:
                continue
            if len(front_materials) > maximum_layers or len(back_materials) > maximum_layers:
                continue
            if len(front_materials) != len(front_values) or len(back_materials) != len(back_values):
                raise ValueError("Elite CSV material/thickness length mismatch")
            def quantized(materials, values):
                return tuple(Layer(material, float(np.clip(np.rint(value / 10.0) * 10.0, 10, 500)))
                             for material, value in zip(materials, values))
            templates.append(DoubleSidedStructure(
                quantized(front_materials, front_values), quantized(back_materials, back_values)
            ).merged())
    if not templates:
        raise ValueError("No elite templates satisfy the requested layer stage")
    return templates


def initialize_worker(nk_database, technical_maximum_layers, stage_maximum_layers, elite_csv=None):
    global _WORKER_NK, _WORKER_CONFIG, _WORKER_ELITES
    _WORKER_CONFIG = DoubleSidedConfig(
        technical_max_layers_per_side=technical_maximum_layers
    ).validate()
    _WORKER_NK = load_materials(
        all_mats=[_WORKER_CONFIG.substrate, *BASE_MATERIALS],
        wavelengths=_WORKER_CONFIG.wavelengths_nm / 1000.0,
        DATABASE=nk_database,
    )
    _WORKER_ELITES = load_elite_templates(elite_csv, stage_maximum_layers)


def perturb_elite(rng, structure, minimum_layers, maximum_layers):
    def side(layers):
        changed = []
        for layer in layers:
            material = (BASE_MATERIALS[rng.randint(len(BASE_MATERIALS))]
                        if rng.rand() < 0.08 else layer.material)
            delta = 10.0 * int(np.clip(np.rint(rng.normal(0.0, 2.0)), -5, 5))
            changed.append(Layer(material, float(np.clip(layer.thickness_nm + delta, 10, 500))))
        while len(merge_adjacent(changed)) < minimum_layers:
            if len(changed) >= maximum_layers:
                raise RuntimeError("Elite perturbation cannot satisfy the requested physical stage")
            position = int(rng.randint(len(changed) + 1))
            neighbors = set()
            if position > 0:
                neighbors.add(changed[position - 1].material)
            if position < len(changed):
                neighbors.add(changed[position].material)
            candidates = [material for material in BASE_MATERIALS if material not in neighbors]
            changed.insert(position, Layer(
                candidates[rng.randint(len(candidates))],
                float(rng.randint(1, 51) * 10),
            ))
        if len(changed) < maximum_layers and rng.rand() < 0.12:
            position = int(rng.randint(len(changed) + 1))
            changed.insert(position, Layer(
                BASE_MATERIALS[rng.randint(len(BASE_MATERIALS))],
                float(rng.randint(1, 51) * 10),
            ))
        if len(changed) > minimum_layers and rng.rand() < 0.08:
            del changed[int(rng.randint(len(changed)))]
        return tuple(changed)
    return DoubleSidedStructure(side(structure.front), side(structure.back))


def generate_one(payload):
    global_index, seed, stage_minimum, stage_maximum = payload
    rng = np.random.RandomState((int(seed) + 1_000_003 * int(global_index)) % (2 ** 32 - 1))
    selector = global_index % 20
    for _ in range(100):
        if _WORKER_ELITES and selector >= 17:
            family = "de_elite"
            template = _WORKER_ELITES[int(rng.randint(len(_WORKER_ELITES)))]
            structure = perturb_elite(rng, template, stage_minimum, stage_maximum)
        else:
            # With elites: 55/30/15. Without elites: deterministic 65/35 bootstrap.
            family = ("random" if (selector < 11 if _WORKER_ELITES else selector < 13)
                      else "alternating")
            structure = sample_random_structure(
                rng, BASE_MATERIALS, (stage_minimum, stage_maximum),
                (_WORKER_CONFIG.min_thickness_nm, _WORKER_CONFIG.max_thickness_nm),
                family=family, nk_dict=_WORKER_NK,
                thickness_step_nm=_WORKER_CONFIG.token_thickness_step_nm,
            )
        front_count, back_count = structure.physical_layer_counts
        if (stage_minimum <= front_count <= stage_maximum and
                stage_minimum <= back_count <= stage_maximum):
            break
    else:
        raise RuntimeError(
            f"Could not sample a physical {stage_minimum}-{stage_maximum} layer structure"
        )
    record, spectra = sample_record(structure, family, _WORKER_NK, _WORKER_CONFIG)
    record["requested_layer_stage"] = [int(stage_minimum), int(stage_maximum)]
    record["global_index"] = int(global_index)
    record["split"] = assign_split(record["split_group_hash"])
    return record, spectra


def atomic_json(data, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_chunk(chunk_dir, chunk_index, items):
    prefix = chunk_dir / f"chunk_{chunk_index:05d}"
    records_path = prefix.with_suffix(".jsonl")
    spectra_path = prefix.with_suffix(".npz")
    metadata_path = prefix.with_suffix(".json")
    with records_path.open("w", encoding="utf-8") as handle:
        for record, _ in items:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    arrays = {
        definition: np.asarray([spectra[definition] for _, spectra in items], dtype=np.float32)
        for definition in ("A", "B", "C")
    }
    np.savez_compressed(spectra_path, **arrays)
    metadata = {
        "chunk_index": chunk_index, "count": len(items),
        "global_index_first": items[0][0]["global_index"],
        "global_index_last": items[-1][0]["global_index"],
        "records_sha256": sha256(records_path), "spectra_sha256": sha256(spectra_path),
        "created_at": datetime.now().isoformat(),
    }
    atomic_json(metadata, metadata_path)
    return metadata


def validate_existing_chunk(chunk_dir, chunk_index, expected_count):
    prefix = chunk_dir / f"chunk_{chunk_index:05d}"
    metadata_path = prefix.with_suffix(".json")
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records_path, spectra_path = prefix.with_suffix(".jsonl"), prefix.with_suffix(".npz")
    if metadata.get("count") != expected_count:
        raise RuntimeError(f"Chunk {chunk_index} count changed; use a new output directory")
    if (sha256(records_path) != metadata.get("records_sha256") or
            sha256(spectra_path) != metadata.get("spectra_sha256")):
        raise RuntimeError(f"Chunk {chunk_index} hash verification failed")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--stage-min-layers", type=int, default=1)
    parser.add_argument("--stage-max-layers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--elite-csv", default=None)
    args = parser.parse_args()
    if args.samples <= 0 or args.chunk_size <= 0 or args.workers <= 0:
        raise ValueError("samples, chunk-size, and workers must be positive")
    if not 1 <= args.stage_min_layers <= args.stage_max_layers:
        raise ValueError("Invalid layer stage")
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    chunk_dir = output / "chunks"
    output.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(exist_ok=True)
    contract_path = output / "generation_contract.json"
    elite_csv = str(Path(args.elite_csv).resolve()) if args.elite_csv else None
    elite_count = len(load_elite_templates(elite_csv, args.stage_max_layers)) if elite_csv else 0
    source_mix = ({"random": 0.55, "alternating": 0.30, "de_elite": 0.15}
                  if elite_csv else {"random": 0.65, "alternating": 0.35})
    contract = {
        "schema_version": 1, "status": "in_progress",
        "samples_requested": args.samples, "chunk_size": args.chunk_size,
        "workers": args.workers, "seed": args.seed,
        "stage_layers_per_side": [args.stage_min_layers, args.stage_max_layers],
        "materials": list(BASE_MATERIALS), "source_mix": source_mix,
        "elite_csv": elite_csv, "elite_template_count": elite_count,
        "deferred_sources": ["ga_elite", "optogpt_candidate", "active_hard"],
        "truth": "A/B/C; C=tmm.inc_tmm; coherent films; incoherent 500 um glass",
        "wavelengths_nm": list(range(400, 1101, 10)), "angle_deg": 60,
        "token_contract": "BOS front SIDE_SEP back EOS",
        "thickness_tokens_nm": {"minimum": 10, "maximum": 500, "step": 10},
        "created_at": datetime.now().isoformat(),
    }
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        immutable = ("samples_requested", "chunk_size", "seed", "stage_layers_per_side",
                     "materials", "elite_csv", "source_mix")
        mismatches = {key: (existing.get(key), contract.get(key)) for key in immutable
                      if existing.get(key) != contract.get(key)}
        if mismatches:
            raise RuntimeError(f"Resume contract mismatch: {mismatches}")
        if not args.resume:
            raise FileExistsError("Generation already exists; use --resume or a new output directory")
    else:
        atomic_json(contract, contract_path)

    nk_database = str(root / "optogpt" / "nk")
    chunk_count = (args.samples + args.chunk_size - 1) // args.chunk_size
    if args.workers == 1:
        initialize_worker(
            nk_database, max(32, args.stage_max_layers), args.stage_max_layers, elite_csv
        )
        map_items = lambda payloads: map(generate_one, payloads)
        executor_context = None
    else:
        executor_context = ProcessPoolExecutor(
            max_workers=args.workers, initializer=initialize_worker,
            initargs=(nk_database, max(32, args.stage_max_layers),
                      args.stage_max_layers, elite_csv))
        map_items = lambda payloads: executor_context.map(generate_one, payloads, chunksize=8)
    try:
        for chunk_index in range(chunk_count):
            start = chunk_index * args.chunk_size
            stop = min(args.samples, start + args.chunk_size)
            expected = stop - start
            if args.resume and validate_existing_chunk(chunk_dir, chunk_index, expected):
                print(f"chunk {chunk_index + 1}/{chunk_count}: verified, skipping", flush=True)
                continue
            payloads = ((index, args.seed, args.stage_min_layers, args.stage_max_layers)
                        for index in range(start, stop))
            items = list(map_items(payloads))
            metadata = write_chunk(chunk_dir, chunk_index, items)
            print(f"chunk {chunk_index + 1}/{chunk_count}: {metadata['count']} samples", flush=True)
    finally:
        if executor_context is not None:
            executor_context.shutdown(wait=True)
    contract["status"] = "chunks_complete"
    contract["completed_at"] = datetime.now().isoformat()
    atomic_json(contract, contract_path)
    print(f"Chunks complete: {output}")


if __name__ == "__main__":
    main()
