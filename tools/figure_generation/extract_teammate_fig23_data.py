#!/usr/bin/env python3
"""Extract the hard-data contract used by teammate paper_fig23 figures."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parent
ARCHIVE = Path("/Users/quhaoran/lab/20260801_145630_optogpt_complete_paper_archive")
SOURCE_ROOT = ARCHIVE / "project_source" / "optogpt"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import core.models.transformer  # noqa: F401,E402


def load_checkpoint_loss(name):
    checkpoint = torch.load(
        ARCHIVE / "trained_models" / "model" / name,
        map_location="cpu",
    )
    loss = checkpoint["loss_all"]
    config = checkpoint["configs"]
    get_value = lambda key, default=None: (
        config.get(key, default)
        if isinstance(config, dict)
        else getattr(config, key, default)
    )
    return {
        "checkpoint": name,
        "epoch": checkpoint.get("epoch"),
        "train_loss": loss["train_loss"],
        "dev_loss": loss["dev_loss"],
        "config": {
            "spec_dim": get_value("spec_dim", 142),
            "layers": get_value("layers", 6),
            "d_model": get_value("d_model", 1024),
            "d_ff": get_value("d_ff", 512),
            "head_num": get_value("head_num", 8),
            "dropout": get_value("dropout", 0.1),
        },
    }


def compact_example(result):
    return {
        "sample_idx": result["sample_idx"],
        "best_structure": result["best_structure"],
        "mae_R": result["mae_R"],
        "mae_T": result["mae_T"],
        "mae_total": result["mae_total"],
        "R_sim": result["R_sim"],
        "T_sim": result["T_sim"],
        "R_target": result["R_target"],
        "T_target": result["T_target"],
    }


def main():
    output_dir = WORKSPACE / "paper_fig23_editable" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    v3 = load_checkpoint_loss("optogpt_60deg_sp_v3_best.pt")
    v4 = load_checkpoint_loss("optogpt_60deg_sp_v4_best.pt")
    ultimate = load_checkpoint_loss("optogpt_60deg_sp_ultimate_best.pt")

    validation_path = (
        ARCHIVE / "paper_results" / "validation_results" / "validation_results.json"
    )
    validation = json.loads(validation_path.read_text())
    results = sorted(validation["results"], key=lambda item: item["mae_total"])
    best = results[0]
    typical = results[len(results) // 2]
    worst = results[-1]

    material_frequency = {
        item["material"]: item["count"]
        for item in validation["material_stats"]["frequency"]
    }
    layer_counts = [len(item["best_structure"]) for item in validation["results"]]
    total_mae = [item["mae_total"] for item in validation["results"]]

    # These values are explicit constants in the teammate plotting scripts.
    performance_contract = {
        "models": ["Original (0 deg)", "v3 s", "v3 p", "v4 s", "v4 p"],
        "total_mae": [0.160, 0.085, 0.032, 0.082, 0.032],
        "target_mae": 0.050,
        "validation_samples_per_polarization": 50,
    }
    material_purity = {
        "stages": ["Original data", "Dielectric output"],
        "dielectric_materials": [10, 10],
        "metal_or_semiconductor_materials": [8, 0],
    }

    record = {
        "provenance": {
            "source_script": "/Users/quhaoran/lab/paper_fig23.py",
            "training_interpretation": "joint s+p training data, per user confirmation",
            "incidence_angle_deg": 60,
            "wavelength_range_nm": [400, 1100],
            "allowed_dielectric_materials": 10,
            "validation_note": (
                "Figure 2 uses constants explicitly encoded in the teammate plotting scripts. "
                "Figure 3 uses the archived 200-sample p-conditioned validation JSON from the "
                "same ultimate checkpoint family."
            ),
        },
        "training": {"v3": v3, "v4": v4, "ultimate": ultimate},
        "performance_contract": performance_contract,
        "material_purity": material_purity,
        "validation": {
            "polarization": validation["polarization"],
            "n_samples": validation["n_samples"],
            "n_valid": validation["n_valid"],
            "summary": validation["summary"],
            "wavelengths_nm": validation["spectral_error"]["wavelengths_nm"],
            "R_mae_by_wavelength": validation["spectral_error"]["R_mae_mean"],
            "T_mae_by_wavelength": validation["spectral_error"]["T_mae_mean"],
            "material_frequency": material_frequency,
            "layer_counts": layer_counts,
            "total_mae": total_mae,
            "examples": {
                "best": compact_example(best),
                "typical": compact_example(typical),
                "worst": compact_example(worst),
            },
        },
    }
    json_path = output_dir / "teammate_fig23_data.json"
    json_path.write_text(json.dumps(record, indent=2))

    loss_csv = output_dir / "teammate_fig2_loss.csv"
    with loss_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "epoch", "train_loss", "dev_loss"])
        for stage_name, stage in (("v3", v3), ("v4", v4), ("ultimate", ultimate)):
            for index, train_loss in enumerate(stage["train_loss"]):
                writer.writerow(
                    [stage_name, index + 1, train_loss, stage["dev_loss"][index]]
                )

    validation_csv = output_dir / "teammate_fig3_validation.csv"
    with validation_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_idx", "mae_R", "mae_T", "mae_total", "layers"])
        for item in validation["results"]:
            writer.writerow(
                [
                    item["sample_idx"],
                    item["mae_R"],
                    item["mae_T"],
                    item["mae_total"],
                    len(item["best_structure"]),
                ]
            )

    print(
        json.dumps(
            {
                "json": str(json_path),
                "loss_csv": str(loss_csv),
                "validation_csv": str(validation_csv),
                "v3_epochs": len(v3["train_loss"]),
                "v4_epochs": len(v4["train_loss"]),
                "validation_samples": len(results),
                "validation_mean_mae": float(np.mean(total_mae)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
