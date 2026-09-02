"""Compare max-8 and max-16 models using physical and robustness outcomes."""

import argparse
import csv
import json
from pathlib import Path


def as_bool(value):
    return str(value).strip().lower() == "true"


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_evaluation(label, evaluation_dir, robustness_dir):
    evaluation_dir = Path(evaluation_dir)
    robustness_dir = Path(robustness_dir)
    rankings = read_csv(evaluation_dir / "rankings.csv")
    if not rankings:
        raise ValueError(f"No rankings in {evaluation_dir}")
    best = min(rankings, key=lambda row: float(row["objective"]))
    manifest = json.loads((evaluation_dir / "manifest.json").read_text(encoding="utf-8"))
    robustness = read_csv(robustness_dir / "robustness_top20_summary.csv")
    if not robustness:
        raise ValueError(f"No robustness rows in {robustness_dir}")
    robust_best = min(robustness, key=lambda row: int(row["rank"]))
    topk_rows = [row for row in rankings if row["method"] == "model_tmm_topk"]
    technical_limit = int(label.replace("max", ""))
    newly_available = [
        row for row in topk_rows
        if max(int(row["front_physical_layers"]), int(row["back_physical_layers"])) > 8
    ] if technical_limit > 8 else []
    return {
        "label": label,
        "technical_max_layers_per_side": technical_limit,
        "best_method": best["method"],
        "best_objective": float(best["objective"]),
        "best_passes_strict": as_bool(best["passes_strict"]),
        "best_front_physical_layers": int(best["front_physical_layers"]),
        "best_back_physical_layers": int(best["back_physical_layers"]),
        "best_front_tokens": best["front_tokens"],
        "best_back_tokens": best["back_tokens"],
        "best_mean_Rs": float(best["mean_Rs"]),
        "best_mean_Rp": float(best["mean_Rp"]),
        "best_p95_Rs": float(best["p95_Rs"]),
        "best_p95_Rp": float(best["p95_Rp"]),
        "best_mean_Ts": float(best["mean_Ts"]),
        "best_mean_Tp": float(best["mean_Tp"]),
        "worst_envelope_objective": float(robust_best["worst_envelope_objective"]),
        "worst_envelope_passes_strict": as_bool(
            robust_best["worst_envelope_passes_strict"]
        ),
        "worst_envelope_mean_Rs": float(robust_best["worst_envelope_mean_Rs"]),
        "worst_envelope_mean_Rp": float(robust_best["worst_envelope_mean_Rp"]),
        "requested_candidates": int(manifest["requested_candidates"]),
        "unique_physical_structures": int(manifest["unique_physical_structures"]),
        "total_tmm_calls": int(manifest["total_tmm_calls"]),
        "generated_structures_using_layers_9_to_16": len(newly_available),
        "generated_fraction_using_layers_9_to_16": (
            len(newly_available) / len(topk_rows) if topk_rows else 0.0
        ),
        "evaluation_dir": str(evaluation_dir.resolve()),
        "robustness_dir": str(robustness_dir.resolve()),
    }


def selection_key(summary):
    return (
        not summary["worst_envelope_passes_strict"],
        not summary["best_passes_strict"],
        summary["worst_envelope_objective"],
        summary["best_objective"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max8-evaluation", required=True)
    parser.add_argument("--max16-evaluation", required=True)
    parser.add_argument("--max8-robustness", required=True)
    parser.add_argument("--max16-robustness", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summaries = [
        summarize_evaluation("max8", args.max8_evaluation, args.max8_robustness),
        summarize_evaluation("max16", args.max16_evaluation, args.max16_robustness),
    ]
    recommended = min(summaries, key=selection_key)
    by_label = {row["label"]: row for row in summaries}
    payload = {
        "models": by_label,
        "nominal_objective_improvement_max16_over_max8": (
            by_label["max8"]["best_objective"] - by_label["max16"]["best_objective"]
        ),
        "worst_envelope_improvement_max16_over_max8": (
            by_label["max8"]["worst_envelope_objective"] -
            by_label["max16"]["worst_envelope_objective"]
        ),
        "recommended_model": recommended["label"],
        "selection_rule": (
            "worst-envelope strict pass, nominal strict pass, lowest worst-envelope objective, "
            "then lowest nominal objective"
        ),
        "budget_note": (
            "Candidate and DE hyperparameters are identical; actual TMM calls are reported "
            "because DE cost depends on physical layer count."
        ),
        "interpretation": (
            "Keep both checkpoints. Select by independent 71-point inc_tmm and robustness, "
            "not token loss alone."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in summaries[0] if not key.endswith("_dir")]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
