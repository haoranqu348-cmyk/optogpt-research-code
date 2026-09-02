"""Assemble the phase-1 credibility/feasibility report without claiming training."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from double_sided.config import DoubleSidedConfig
from double_sided.contract import DoubleSidedStructure
from double_sided.physics import simulate_abc, summarize, verify_merge_equivalence
from double_sided.scripts.evaluate_robustness import INCUMBENT_BACK, INCUMBENT_FRONT
from optogpt.core.datasets.sim import load_materials


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_abc(output, config, labels):
    metrics = {definition: summarize(labels[definition]) for definition in labels}
    with (output / "ABC_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output / "ABC_full_spectra.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        fields = ["wavelength_nm"]
        for definition in ("A", "B", "C"):
            fields += [f"{definition}_{key}{pol}" for pol in ("s", "p") for key in ("R", "T", "A")]
        writer.writerow(fields)
        for index, wavelength in enumerate(config.wavelengths_nm):
            row = [wavelength]
            for definition in ("A", "B", "C"):
                row += [labels[definition][pol][key][index]
                        for pol in ("s", "p") for key in ("R", "T", "A")]
            writer.writerow(row)
    return metrics


def make_plots(output, phase_root):
    with (phase_root / "feasibility_run_01" / "feasibility_manifest.json").open() as handle:
        feasibility = json.load(handle)
    stages = feasibility["stage_summary"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = sorted({row["material_ablation"] for row in stages})
    for label in labels:
        rows = [row for row in stages if row["material_ablation"] == label]
        ax.plot([row["stage"][1] for row in rows], [row["best_objective"] for row in rows],
                marker="o", label=label)
    ax.set(xlabel="Maximum layers per side in stage", ylabel="Best full-grid positive objective")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(output / "layer_stage_pareto.png", dpi=160); plt.close(fig)

    with (phase_root / "elite_gate_run_01" / "elite_material_gate_manifest.json").open() as handle:
        gate = json.load(handle)
    incumbent = gate["incumbent_objective_recomputed"]
    material_labels, values = [], []
    for label, result in gate["best_by_set"].items():
        if label == "base":
            continue
        candidate = result["best_using_new_material"]
        material_labels.append(label.replace("base_plus_", ""))
        values.append(candidate["objective"] if candidate else np.nan)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(material_labels, values, color=["#2f6f8f", "#c65d3a", "#56865c"])
    ax.axhline(incumbent, color="#202020", linestyle="--", label="unchanged incumbent")
    ax.set(ylabel="Best full-grid positive objective", title="New-material elite gate")
    ax.grid(axis="y", alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output / "material_ablation.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feasibility-run", default="feasibility_run_02")
    parser.add_argument("--elite-gate-run", default="elite_gate_run_02")
    args = parser.parse_args()
    phase_root, output = Path(args.phase_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    config = DoubleSidedConfig().validate()
    structure = DoubleSidedStructure(INCUMBENT_FRONT, INCUMBENT_BACK)
    materials = sorted({layer.material for layer in (*structure.front, *structure.back)})
    nk = load_materials(
        all_mats=[config.substrate, *materials], wavelengths=config.wavelengths_nm / 1000.0,
        DATABASE=str(root / "optogpt" / "nk"),
    )
    labels = simulate_abc(structure, nk, config)
    metrics = write_abc(output, config, labels)
    merge_check = verify_merge_equivalence(structure, nk, config)
    # Plot helper expects stable aliases so reruns never overwrite raw search outputs.
    aliases = {
        "feasibility_run_01": phase_root / args.feasibility_run,
        "elite_gate_run_01": phase_root / args.elite_gate_run,
    }
    alias_root = output / ".plot_inputs"
    alias_root.mkdir()
    for alias, source in aliases.items():
        (alias_root / alias).symlink_to(source.resolve(), target_is_directory=True)
    make_plots(output, alias_root)
    shutil.rmtree(alias_root)

    for source, destination in (
        (root / "double_sided" / "materials" / "material_audit.yaml", "material_audit.yaml"),
        (phase_root / "nk" / "nk_manifest.json", "nk_manifest.json"),
        (phase_root / args.feasibility_run / "feasibility_rankings.csv", "layer_material_feasibility.csv"),
        (phase_root / args.elite_gate_run / "elite_material_gate.csv", "elite_material_gate.csv"),
        (phase_root / "robustness_incumbent" / "robustness_scenarios.csv", "robustness_scenarios.csv"),
        (phase_root / "robustness_incumbent" / "wavelength_worst_case.csv", "wavelength_worst_case.csv"),
        (phase_root / "data_smoke_v2" / "dataset_contract.json", "dataset_contract.json"),
        (phase_root / "data_smoke_v2" / "hash_manifest.json", "hash_manifest_smoke.json"),
    ):
        shutil.copy2(source, output / destination)

    checkpoint = root / "joint_sp" / "formal_checkpoints_500k_v2_20260725_03" / "optogpt_joint_sp_500k_v2_best.pt"
    checkpoint_metadata = {
        "required_source": str(checkpoint), "available": checkpoint.exists(),
        "sha256": sha256(checkpoint) if checkpoint.exists() else None,
        "policy": "copy into a new source_checkpoints directory; never overwrite source checkpoint",
        "weight_transfer": "inherit named rows for all old tokens; initialize SIDE_SEP/new rows only",
        "training_status": "blocked_pending_checkpoint_and_AlF3_numeric_dispersion",
    }
    with (output / "checkpoint_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(checkpoint_metadata, handle, indent=2)
    manifest = {
        "phase": "material credibility and TMM feasibility before training",
        "training_started": False, "model_checkpoint_available": checkpoint.exists(),
        "material_training_gate_passed": False,
        "reason": "Nb2O5/Sc2O3 did not improve incumbent; AlF3 numeric full-band data pending",
        "incumbent_ABC_metrics": metrics, "merge_equivalence": merge_check,
        "python": platform.python_version(), "torch": torch.__version__,
        "truth_backend": "tmm.inc_tmm, 71 points, s and p separately",
        "budget_accounting": "one call per polarization per wavelength; search evaluates C only",
        "source_runs": {"feasibility": args.feasibility_run, "elite_gate": args.elite_gate_run},
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
