#!/usr/bin/env python3
"""Generate same-target double-sided candidates from archived checkpoints."""

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent
ARCHIVE = Path("/Users/quhaoran/lab/20260801_145630_optogpt_complete_paper_archive")
PROJECT = ARCHIVE / "project_source"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from double_sided.config import BASE_MATERIALS, DoubleSidedConfig  # noqa: E402
from double_sided.decoder import constrained_decode  # noqa: E402
from double_sided.model import load_double_sided_checkpoint  # noqa: E402
from double_sided.physics import simulate_c, summarize  # noqa: E402
from optogpt.core.datasets.sim import load_materials, spectrum  # noqa: E402


WAVELENGTHS_UM = np.arange(0.4, 1.101, 0.01)
WAVELENGTHS_NM = (WAVELENGTHS_UM * 1000).astype(int)


def top_k_sampler(temperature=0.9, top_k=32):
    def sample(logits):
        values, indices = torch.topk(logits / temperature, k=min(top_k, logits.size(-1)), dim=-1)
        selected = torch.multinomial(torch.softmax(values, dim=-1), 1)
        return indices.gather(1, selected).squeeze(1)
    return sample


def parse_tokens(tokens):
    materials, thicknesses = [], []
    for token in tokens:
        material, thickness = token.rsplit("_", 1)
        materials.append(material)
        thicknesses.append(float(thickness))
    return materials, thicknesses


def target_from_validation():
    validation = json.loads(
        (PROJECT / "joint_sp/validation_results/formal_500k_v2_best_20260726/validation_results.json").read_text()
    )
    source = min(validation, key=lambda item: item["E_joint"])
    materials, thicknesses = parse_tokens(source["target_tokens"])
    nk_dict = load_materials(
        all_mats=["Glass_Substrate", *BASE_MATERIALS],
        wavelengths=WAVELENGTHS_UM,
        DATABASE=str(PROJECT / "optogpt/nk"),
    )
    sim_s = spectrum(materials, thicknesses, pol="s", theta=60, wavelengths=WAVELENGTHS_UM, nk_dict=nk_dict, substrate="Glass_Substrate", substrate_thick=500000)
    sim_p = spectrum(materials, thicknesses, pol="p", theta=60, wavelengths=WAVELENGTHS_UM, nk_dict=nk_dict, substrate="Glass_Substrate", substrate_thick=500000)
    target = np.asarray([*sim_s, *sim_p], dtype=np.float32)
    return source, target, nk_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-layers-per-side", type=int, choices=(8, 16), required=True)
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "paper_figures/data")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    checkpoint = ARCHIVE / "trained_models/results/double_sided_inverse_design/20260730_205309_base_material_max8_max16/models" / f"max{args.max_layers_per_side}/best_physical.pt"
    source, target, nk_dict = target_from_validation()
    model, word_dict, index_dict, checkpoint_config, _ = load_double_sided_checkpoint(str(checkpoint), device)
    allowed = tuple(checkpoint_config["allowed_materials"])
    config = DoubleSidedConfig(technical_max_layers_per_side=max(32, args.max_layers_per_side), allowed_materials=allowed).validate()
    generated = constrained_decode(
        model,
        np.repeat(target[None, :], args.num_candidates, axis=0),
        word_dict,
        index_dict,
        allowed,
        args.max_layers_per_side,
        sample_fn=top_k_sampler(),
        device=device,
    )
    unique = {}
    for structure in generated:
        unique.setdefault(structure.physical_hash(), structure.merged())
    ranked = []
    for structure in unique.values():
        result = simulate_c(structure, nk_dict, config)
        ranked.append({"structure": structure, "result": result, "metrics": summarize(result)})
    ranked.sort(key=lambda row: (-row["metrics"]["mean_Ts"], row["metrics"]["objective"]))
    if not ranked:
        raise RuntimeError("No valid double-sided candidates were retained")
    top = []
    for row in ranked[: min(16, len(ranked))]:
        st = row["structure"]
        result = row["result"]
        metrics = row["metrics"]
        top.append({
            "type": "double_sided",
            "model_variant": f"max{args.max_layers_per_side}",
            "front_materials": [layer.material for layer in st.front],
            "front_thicknesses": [layer.thickness_nm for layer in st.front],
            "back_materials": [layer.material for layer in st.back],
            "back_thicknesses": [layer.thickness_nm for layer in st.back],
            "front_layers": len(st.front), "back_layers": len(st.back),
            "n_layers": len(st.front) + len(st.back),
            "sim_Rs": result["s"]["R"].tolist(), "sim_Ts": result["s"]["T"].tolist(),
            "sim_Rp": result["p"]["R"].tolist(), "sim_Tp": result["p"]["T"].tolist(),
            **metrics,
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_contract": {
            "checkpoint": str(checkpoint), "model_variant": f"max{args.max_layers_per_side}",
            "theta_deg": 60, "wavelengths_nm": WAVELENGTHS_NM.tolist(),
            "seed": args.seed, "requested_candidates": args.num_candidates,
            "generated_unique_candidates": len(unique), "truth_backend": "tmm.inc_tmm, 71 points",
            "shared_target_index": source["index"],
        },
        "shared_target": {"archived_validation_index": source["index"], "Rs": target[:71].tolist(), "Ts": target[71:142].tolist(), "Rp": target[142:213].tolist(), "Tp": target[213:284].tolist()},
        "candidates": top,
    }
    out = args.output_dir / f"figure4_same_target_double_sided_max{args.max_layers_per_side}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(out), "generated": len(generated), "unique": len(unique), "top": [{"n_layers": c["n_layers"], "front_layers": c["front_layers"], "back_layers": c["back_layers"], "mean_Ts": c["mean_Ts"], "mean_Tp": c["mean_Tp"], "objective": c["objective"]} for c in top[:5]]}, indent=2))


if __name__ == "__main__":
    main()
