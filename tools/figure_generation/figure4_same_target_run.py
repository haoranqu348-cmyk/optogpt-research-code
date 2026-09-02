#!/usr/bin/env python3
"""Retain a same-target candidate pool for the manuscript Figure 4."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parent
PROJECT = WORKSPACE / "optogpt_project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
OPTOGPT_PACKAGE_ROOT = PROJECT / "optogpt"
if str(OPTOGPT_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTOGPT_PACKAGE_ROOT))

from joint_sp.constants import (  # noqa: E402
    ALLOWED_MATERIALS,
    BANNED_MATERIALS,
    MAX_LAYERS,
    SUBSTRATE,
    SUBSTRATE_THICK_NM,
    THETA_DEG,
)
from joint_sp.decoder import (  # noqa: E402
    build_joint_logits_mask,
    generate_candidates_sp,
    tmm_rerank_joint,
)
from joint_sp.model import load_joint_sp_checkpoint  # noqa: E402
from optogpt.core.datasets.sim import load_materials, spectrum  # noqa: E402


WAVELENGTHS_UM = np.arange(0.4, 1.101, 0.01)
WAVELENGTHS_NM = (WAVELENGTHS_UM * 1000).astype(int)


def parse_tokens(tokens):
    materials = []
    thicknesses = []
    for token in tokens:
        material, thickness = token.rsplit("_", 1)
        materials.append(material)
        thicknesses.append(int(thickness))
    return materials, thicknesses


def simulate_joint(materials, thicknesses, nk_dict):
    sim_s = spectrum(
        materials,
        thicknesses,
        pol="s",
        theta=THETA_DEG,
        wavelengths=WAVELENGTHS_UM,
        nk_dict=nk_dict,
        substrate=SUBSTRATE,
        substrate_thick=SUBSTRATE_THICK_NM,
    )
    sim_p = spectrum(
        materials,
        thicknesses,
        pol="p",
        theta=THETA_DEG,
        wavelengths=WAVELENGTHS_UM,
        nk_dict=nk_dict,
        substrate=SUBSTRATE,
        substrate_thick=SUBSTRATE_THICK_NM,
    )
    return np.asarray([*sim_s, *sim_p], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-candidates", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--objective",
        choices=("joint_error", "high_transmission"),
        default="joint_error",
    )
    parser.add_argument("--output-stem", default="figure4_same_target")
    parser.add_argument(
        "--target-profile",
        choices=("archived", "flat_high_transmission"),
        default="archived",
    )
    parser.add_argument("--target-rs", type=float, default=0.05)
    parser.add_argument("--target-ts", type=float, default=0.95)
    parser.add_argument("--target-rp", type=float, default=0.05)
    parser.add_argument("--target-tp", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "paper_figures" / "data",
    )
    args = parser.parse_args()

    checkpoint = (
        PROJECT
        / "joint_sp"
        / "formal_checkpoints_500k_v2_20260725_03"
        / "optogpt_joint_sp_500k_v2_best.pt"
    )
    validation_path = (
        PROJECT
        / "joint_sp"
        / "validation_results"
        / "formal_500k_v2_best_20260726"
        / "validation_results.json"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    validation = json.loads(validation_path.read_text())
    # Use one archived held-out target and reconstruct its exact TMM condition.
    source = min(validation, key=lambda item: item["E_joint"])
    target_materials, target_thicknesses = parse_tokens(source["target_tokens"])

    nk_dict = load_materials(
        all_mats=[SUBSTRATE] + ALLOWED_MATERIALS + list(BANNED_MATERIALS),
        wavelengths=WAVELENGTHS_UM,
        DATABASE=str(OPTOGPT_PACKAGE_ROOT / "optogpt" / "nk"),
    )
    if args.target_profile == "archived":
        target = simulate_joint(target_materials, target_thicknesses, nk_dict)
        target_source = {
            "archived_validation_index": source["index"],
            "source_target_tokens": source["target_tokens"],
            "source_materials": target_materials,
            "source_thicknesses_nm": target_thicknesses,
        }
    else:
        target = np.concatenate(
            [
                np.full(71, args.target_rs, dtype=np.float32),
                np.full(71, args.target_ts, dtype=np.float32),
                np.full(71, args.target_rp, dtype=np.float32),
                np.full(71, args.target_tp, dtype=np.float32),
            ]
        )
        target_source = {
            "archived_validation_index": None,
            "target_profile": "flat_high_transmission",
            "target_definition": {
                "Rs": args.target_rs,
                "Ts": args.target_ts,
                "Rp": args.target_rp,
                "Tp": args.target_tp,
            },
        }

    model, word_dict, index_dict, config = load_joint_sp_checkpoint(
        str(checkpoint), device=device
    )
    model.eval()
    logits_mask, _ = build_joint_logits_mask(word_dict, ALLOWED_MATERIALS)
    candidates = generate_candidates_sp(
        model,
        target,
        word_dict,
        index_dict,
        num_candidates=args.num_candidates,
        max_len=MAX_LAYERS + 2,
        device=device,
        logits_mask=logits_mask,
    )
    ranked, failures = tmm_rerank_joint(
        candidates,
        target,
        nk_dict,
        wavelengths=WAVELENGTHS_UM,
        theta=THETA_DEG,
        objective=args.objective,
    )
    if len(ranked) < 3:
        raise RuntimeError(f"Only {len(ranked)} valid candidates were retained")

    target_parts = {
        "Rs": target[0:71].tolist(),
        "Ts": target[71:142].tolist(),
        "Rp": target[142:213].tolist(),
        "Tp": target[213:284].tolist(),
    }
    record = {
        "run_contract": {
            "model": str(checkpoint),
            "architecture_version": config.get("architecture_version"),
            "spec_layout": config.get("spec_layout"),
            "theta_deg": THETA_DEG,
            "wavelengths_nm": WAVELENGTHS_NM.tolist(),
            "seed": args.seed,
            "ranking_objective": args.objective,
            "target_profile": args.target_profile,
            "requested_candidates": args.num_candidates,
            "generated_unique_candidates": len(candidates),
            "retained_tmm_candidates": len(ranked),
            "tmm_failures": failures,
        },
        "shared_target": {
            **target_source,
            **target_parts,
        },
        "candidates": ranked,
    }
    json_path = args.output_dir / f"{args.output_stem}_candidates.json"
    json_path.write_text(json.dumps(record, indent=2))

    csv_path = args.output_dir / f"{args.output_stem}_spectra.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "wavelength_nm",
                "target_Rs",
                "target_Ts",
                "target_Rp",
                "target_Tp",
                *[
                    f"candidate_{rank}_{quantity}"
                    for rank in range(1, 4)
                    for quantity in ("Rs", "Ts", "Rp", "Tp")
                ],
            ]
        )
        for index, wavelength in enumerate(WAVELENGTHS_NM):
            row = [
                wavelength,
                target_parts["Rs"][index],
                target_parts["Ts"][index],
                target_parts["Rp"][index],
                target_parts["Tp"][index],
            ]
            for candidate in ranked[:3]:
                row.extend(
                    [
                        candidate["sim_Rs"][index],
                        candidate["sim_Ts"][index],
                        candidate["sim_Rp"][index],
                        candidate["sim_Tp"][index],
                    ]
                )
            writer.writerow(row)

    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "target_profile": args.target_profile,
        "generated": len(candidates),
        "retained": len(ranked),
        "top_E_joint": [item["E_joint"] for item in ranked[:3]],
    }, indent=2))


if __name__ == "__main__":
    main()
