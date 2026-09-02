"""Generate per-angle error data and plots for wide-angle candidates."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_JOINT_SP = _SCRIPT_DIR.parent
_PKG_ROOT = _JOINT_SP.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from joint_sp.constants import (  # noqa: E402
    ALLOWED_MATERIALS,
    BANNED_MATERIALS,
    SUBSTRATE,
)
from joint_sp.io_utils import atomic_json_dump  # noqa: E402
from joint_sp.scripts.evaluate_wide_angle import (  # noqa: E402
    DEFAULT_WAVELENGTHS_UM,
    _candidate_from_mapping,
    _get_sim_backend,
    deduplicate_candidates,
    load_structure_candidates,
    parse_angle_grid,
    simulate_angle_grid,
)


def load_attempt_candidates(paths):
    """Load every saved attempt and retain its source seed/rank identity."""
    candidates = []
    for path in paths:
        source_path = Path(path)
        with open(source_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("top_results", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"Expected a result list in {source_path}")
        seed_text = source_path.stem.removeprefix("seed_")
        for rank, item in enumerate(rows, start=1):
            candidate = _candidate_from_mapping(item)
            candidate["candidate_id"] = f"S{seed_text}-R{rank:02d}"
            candidate["source_file"] = str(source_path)
            candidate["source_rank"] = rank
            candidates.append(candidate)
    return candidates


def _load_plot_backend():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def compute_error_rows(candidates, angles, nk_dict):
    """Compute errors relative to the ideal broadband target Ts=Tp=1."""
    rows = []
    spectra = {}
    failures = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_id = candidate.get("candidate_id", f"C{candidate_index:02d}")
        try:
            matrices = simulate_angle_grid(candidate, angles, nk_dict)
        except Exception as exc:
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "structure_hash": candidate["structure_hash"],
                    "error": str(exc),
                }
            )
            continue
        spectra[candidate_id] = matrices
        for angle_index, angle in enumerate(angles):
            error_s = 1.0 - matrices["Ts"][angle_index]
            error_p = 1.0 - matrices["Tp"][angle_index]
            e_s = float(np.mean(error_s))
            e_p = float(np.mean(error_p))
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "structure_hash": candidate["structure_hash"],
                    "tokens": " | ".join(candidate["tokens"]),
                    "n_layers": candidate["n_layers"],
                    "source_file": candidate.get("source_file", ""),
                    "source_rank": candidate.get("source_rank", ""),
                    "angle_deg": float(angle),
                    "E_s": e_s,
                    "E_p": e_p,
                    "E_joint": 0.5 * (e_s + e_p),
                    "p95_error_s": float(np.percentile(error_s, 95)),
                    "p95_error_p": float(np.percentile(error_p, 95)),
                    "max_error_s": float(np.max(error_s)),
                    "max_error_p": float(np.max(error_p)),
                    "mean_Ts": float(np.mean(matrices["Ts"][angle_index])),
                    "mean_Tp": float(np.mean(matrices["Tp"][angle_index])),
                }
            )
    return rows, spectra, failures


def write_csv(path, rows):
    if not rows:
        raise ValueError("No error rows to save")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["candidate_id"], []).append(row)
    summary = []
    for candidate_id, candidate_rows in grouped.items():
        worst = max(candidate_rows, key=lambda row: row["E_joint"])
        summary.append(
            {
                "candidate_id": candidate_id,
                "structure_hash": candidate_rows[0]["structure_hash"],
                "tokens": candidate_rows[0]["tokens"],
                "n_layers": candidate_rows[0]["n_layers"],
                "source_file": candidate_rows[0].get("source_file", ""),
                "source_rank": candidate_rows[0].get("source_rank", ""),
                "mean_E_s_over_angles": float(np.mean([row["E_s"] for row in candidate_rows])),
                "mean_E_p_over_angles": float(np.mean([row["E_p"] for row in candidate_rows])),
                "mean_E_joint_over_angles": float(
                    np.mean([row["E_joint"] for row in candidate_rows])
                ),
                "worst_E_joint": worst["E_joint"],
                "worst_angle_deg": worst["angle_deg"],
                "E_joint_at_60deg": next(
                    (row["E_joint"] for row in candidate_rows if row["angle_deg"] == 60.0),
                    None,
                ),
                "E_joint_at_80deg": next(
                    (row["E_joint"] for row in candidate_rows if row["angle_deg"] == 80.0),
                    None,
                ),
            }
        )
    return sorted(summary, key=lambda row: (row["mean_E_joint_over_angles"], row["worst_E_joint"]))


def plot_all_curves(plt, rows, summary, output_dir):
    grouped = {candidate_id: [] for candidate_id in (row["candidate_id"] for row in summary)}
    for row in rows:
        grouped[row["candidate_id"]].append(row)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for axis, key, title in zip(
        axes,
        ("E_s", "E_p", "E_joint"),
        ("s-polarization error", "p-polarization error", "joint error"),
    ):
        for candidate_id, candidate_rows in grouped.items():
            ordered = sorted(candidate_rows, key=lambda row: row["angle_deg"])
            axis.plot(
                [row["angle_deg"] for row in ordered],
                [row[key] for row in ordered],
                linewidth=1.0,
                alpha=0.68,
                label=candidate_id,
            )
        axis.set_title(title)
        axis.set_xlabel("Incidence angle (deg)")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Mean absolute error to ideal T=1")
    axes[-1].legend(ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.suptitle(
        f"All {len(summary)} evaluated records: transmission error versus incidence angle"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "all_candidates_error_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(plt, rows, summary, angles, output_dir):
    candidate_ids = [row["candidate_id"] for row in summary]
    lookup = {(row["candidate_id"], row["angle_deg"]): row for row in rows}
    fig, axes = plt.subplots(1, 3, figsize=(17, 7), sharey=True)
    for axis, key, title in zip(
        axes,
        ("E_s", "E_p", "E_joint"),
        ("s error", "p error", "joint error"),
    ):
        matrix = np.asarray(
            [[lookup[(candidate_id, float(angle))][key] for angle in angles] for candidate_id in candidate_ids]
        )
        image = axis.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            extent=[float(angles[0]), float(angles[-1]), len(candidate_ids) - 0.5, -0.5],
            vmin=0.0,
            vmax=max(0.7, float(np.max(matrix))),
            cmap="magma",
        )
        axis.set_title(title)
        axis.set_xlabel("Incidence angle (deg)")
        axis.set_yticks(np.arange(len(candidate_ids)))
        axis.set_yticklabels(candidate_ids)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Candidate, ranked by mean joint error")
    fig.suptitle("Candidate x angle error heatmaps")
    fig.tight_layout()
    fig.savefig(output_dir / "candidate_angle_error_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_angle_aggregate(plt, rows, angles, output_dir):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["angle_deg"], []).append(row)
    fig, axis = plt.subplots(figsize=(10, 5.5))
    colors = {"E_s": "#d1495b", "E_p": "#227c9d", "E_joint": "#2f4858"}
    for key, label in (("E_s", "s"), ("E_p", "p"), ("E_joint", "joint")):
        median = np.asarray([np.median([row[key] for row in grouped[float(angle)]]) for angle in angles])
        lower = np.asarray([np.percentile([row[key] for row in grouped[float(angle)]], 10) for angle in angles])
        upper = np.asarray([np.percentile([row[key] for row in grouped[float(angle)]], 90) for angle in angles])
        axis.plot(angles, median, color=colors[key], linewidth=2.0, label=f"{label} median")
        axis.fill_between(angles, lower, upper, color=colors[key], alpha=0.14, label=f"{label} p10-p90")
    axis.set_xlabel("Incidence angle (deg)")
    axis.set_ylabel("Mean absolute error to ideal T=1")
    axis.set_title("Error distribution across all candidates at each angle")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "per_angle_error_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_candidate(plt, rows, summary, output_dir):
    best_id = summary[0]["candidate_id"]
    selected = sorted(
        [row for row in rows if row["candidate_id"] == best_id],
        key=lambda row: row["angle_deg"],
    )
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for key, label, color in (
        ("E_s", "E_s", "#d1495b"),
        ("E_p", "E_p", "#227c9d"),
        ("E_joint", "E_joint", "#2f4858"),
        ("p95_error_s", "s p95 error", "#f4a261"),
        ("p95_error_p", "p p95 error", "#43aa8b"),
    ):
        axis.plot(
            [row["angle_deg"] for row in selected],
            [row[key] for row in selected],
            label=label,
            color=color,
            linewidth=2.0 if key in ("E_s", "E_p", "E_joint") else 1.2,
            linestyle="-" if key in ("E_s", "E_p", "E_joint") else "--",
        )
    axis.set_xlabel("Incidence angle (deg)")
    axis.set_ylabel("Error to ideal T=1")
    axis.set_title(f"Best aggregate candidate {best_id}: {summary[0]['tokens']}")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "best_candidate_error_by_angle.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_each_angle(plt, rows, angles, output_dir):
    per_angle_dir = output_dir / "per_angle"
    per_angle_dir.mkdir(parents=True, exist_ok=True)
    for angle in angles:
        selected = sorted(
            [row for row in rows if row["angle_deg"] == float(angle)],
            key=lambda row: row["E_joint"],
        )
        positions = np.arange(len(selected))
        width = 0.27
        fig, axis = plt.subplots(figsize=(12, 5.5))
        axis.bar(positions - width, [row["E_s"] for row in selected], width, label="E_s", color="#d1495b")
        axis.bar(positions, [row["E_p"] for row in selected], width, label="E_p", color="#227c9d")
        axis.bar(positions + width, [row["E_joint"] for row in selected], width, label="E_joint", color="#2f4858")
        axis.set_xticks(positions)
        axis.set_xticklabels([row["candidate_id"] for row in selected], rotation=45, ha="right")
        axis.set_xlabel("Candidate, sorted by E_joint at this angle")
        axis.set_ylabel("Mean absolute error to ideal T=1")
        axis.set_title(f"All candidates at incidence angle {float(angle):.0f} deg")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(frameon=False, ncol=3)
        axis.set_ylim(0.0, 1.0)
        fig.tight_layout()
        fig.savefig(per_angle_dir / f"angle_{int(round(float(angle))):03d}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot wide-angle candidate errors")
    parser.add_argument("--structures", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--angles", default="0:80:1")
    parser.add_argument(
        "--keep_duplicates",
        action="store_true",
        help="Keep all saved seed/rank attempts instead of deduplicating structures",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = output_dir / "ERROR_PLOTS_COMPLETE.json"
    if complete.exists():
        raise FileExistsError(f"Refusing to overwrite completed plot set: {complete}")

    if args.keep_duplicates:
        candidates = load_attempt_candidates(args.structures)
    else:
        candidates = deduplicate_candidates(
            [candidate for path in args.structures for candidate in load_structure_candidates(path)]
        )
    angles = parse_angle_grid(args.angles)
    load_materials, _spectrum = _get_sim_backend()
    nk_dict = load_materials(
        all_mats=[SUBSTRATE] + ALLOWED_MATERIALS + list(BANNED_MATERIALS),
        wavelengths=DEFAULT_WAVELENGTHS_UM,
        DATABASE=str(_PKG_ROOT / "optogpt" / "nk"),
    )

    rows, _spectra, failures = compute_error_rows(candidates, angles, nk_dict)
    summary = candidate_summary(rows)
    write_csv(output_dir / "all_candidate_angle_errors.csv", rows)
    write_csv(output_dir / "candidate_error_summary.csv", summary)

    plt = _load_plot_backend()
    plot_all_curves(plt, rows, summary, output_dir)
    plot_heatmaps(plt, rows, summary, angles, output_dir)
    plot_angle_aggregate(plt, rows, angles, output_dir)
    plot_best_candidate(plt, rows, summary, output_dir)
    plot_each_angle(plt, rows, angles, output_dir)

    report = {
        "status": "complete",
        "error_definition": {
            "target": "broadband Ts=Tp=1",
            "E_s": "mean_wavelength(1 - Ts)",
            "E_p": "mean_wavelength(1 - Tp)",
            "E_joint": "0.5 * (E_s + E_p)",
            "tail": "p95 and maximum of 1 - T over wavelength",
        },
        "source_files": args.structures,
        "angles_deg": angles.tolist(),
        "n_raw_records": sum(len(load_structure_candidates(path)) for path in args.structures),
        "n_attempts_plotted": len(candidates),
        "n_unique_candidates": len({candidate["structure_hash"] for candidate in candidates}),
        "duplicates_retained": args.keep_duplicates,
        "n_valid_candidates": len(summary),
        "n_tmm_failures": len(failures),
        "best_by_mean_joint_error": summary[0] if summary else None,
        "candidate_ranking": summary,
        "failures": failures,
        "files": {
            "raw_csv": "all_candidate_angle_errors.csv",
            "summary_csv": "candidate_error_summary.csv",
            "curve_plot": "all_candidates_error_curves.png",
            "heatmaps": "candidate_angle_error_heatmaps.png",
            "distribution_plot": "per_angle_error_distribution.png",
            "best_candidate_plot": "best_candidate_error_by_angle.png",
            "per_angle_plot_dir": "per_angle",
            "per_angle_plot_count": len(angles),
        },
    }
    atomic_json_dump(report, output_dir / "error_plot_manifest.json")
    atomic_json_dump({"status": "complete", "n_plots": len(angles) + 4}, complete)
    print(f"Candidates plotted: {len(candidates)}")
    print(f"Error rows: {len(rows)}")
    print(f"Per-angle plots: {len(angles)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
