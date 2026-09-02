"""Evaluate strict joint s/p cases at multiple candidate budgets."""

import argparse
import csv
import hashlib
import json
import pickle
import statistics
from pathlib import Path

from batch_predict_hard_joint_cases import MODEL
from interactive_predictor import InteractivePredictor
from run_prediction import load_spectrum_file


SCRIPT_DIR = Path(__file__).resolve().parent
CASES_DIR = SCRIPT_DIR / "inputs" / "strict_joint_sp_eval_20260726"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "strict_joint_sp_eval_20260726"
PROJECT_ROOT = SCRIPT_DIR.parent


def _structure_tokens(materials, thicknesses):
    return tuple(f"{m}_{t}" for m, t in zip(materials, thicknesses))


def _available_train_structures():
    result = set()
    paths = [
        PROJECT_ROOT / "data_60deg_s_500k_dielectric" / "Structure_train.pkl",
        PROJECT_ROOT / "optogpt" / "data_60deg_s" / "Structure_train.pkl",
        PROJECT_ROOT / "optogpt" / "dielectric_60deg_s" / "data" / "Structure_train.pkl",
        PROJECT_ROOT / "optogpt" / "data" / "Structure_train.pkl",
    ]
    for path in paths:
        if not path.exists():
            continue
        with path.open("rb") as handle:
            result.update(tuple(row) for row in pickle.load(handle))
    return result


def _evaluate(predictor, target, count, seed):
    results = predictor.predict_and_validate(target, num_candidates=count, seed=seed)
    if not results:
        return None, 0
    return results[0], len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, default=CASES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--counts", default="1,16,64")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = [int(v) for v in args.counts.split(",") if v.strip()]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    with (args.cases_dir / "answers.csv").open(newline="") as handle:
        answers = {row["case"]: row for row in csv.DictReader(handle)}
    sources = sorted(args.cases_dir.glob("strict_*_60deg_sp.csv"))
    train_structures = _available_train_structures()
    predictor = InteractivePredictor(model_path=str(MODEL))
    predictor.load_model()
    rows = []
    for index, source in enumerate(sources, 1):
        case = source.stem.replace("_60deg_sp", "")
        print(f"\n[strict {index}/{len(sources)}] {case}", flush=True)
        target = load_spectrum_file(source, joint_sp=True)
        answer = answers[case]
        known = tuple(
            f"{m}_{t}" for m, t in zip(
                answer["materials"].split("|"), answer["thicknesses_nm"].split("|")
            )
        )
        for count in counts:
            best, valid_count = _evaluate(predictor, target, count, args.seed + index)
            row = {
                "case": case, "candidate_count": count,
                "status": "ok" if best else "no_valid_candidate",
                "valid_candidates": valid_count,
                "known_layers": len(known),
                "known_exact_in_available_train": known in train_structures,
            }
            if best:
                predicted = tuple(best["tokens"])
                row.update({
                    "mae_total": best["mae_total"], "mae_s": best["mae_s"], "mae_p": best["mae_p"],
                    "mae_Rs": best["mae_Rs"], "mae_Ts": best["mae_Ts"],
                    "mae_Rp": best["mae_Rp"], "mae_Tp": best["mae_Tp"],
                    "predicted_layers": best["n_layers"],
                    "exact_structure_match": predicted == known,
                    "position_token_match": sum(a == b for a, b in zip(known, predicted)) / max(len(known), len(predicted)),
                    "predicted_structure": ",".join(predicted),
                })
            rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"model": str(MODEL), "cases": len(sources), "candidate_counts": counts, "rows": len(rows), "train_structure_sources_checked": len(_available_train_structures())}
    by_count = {}
    for count in counts:
        subset = [r for r in rows if r["candidate_count"] == count and r["status"] == "ok"]
        vals = [float(r["mae_total"]) for r in subset]
        by_count[str(count)] = {
            "n": len(subset),
            "mean_mae": statistics.mean(vals) if vals else None,
            "median_mae": statistics.median(vals) if vals else None,
            "best_mae": min(vals) if vals else None,
            "worst_mae": max(vals) if vals else None,
            "exact_structure_matches": sum(bool(r.get("exact_structure_match")) for r in subset),
            "mean_position_token_match": statistics.mean(float(r.get("position_token_match", 0)) for r in subset) if subset else None,
            "available_train_exact_overlaps": sum(bool(r.get("known_exact_in_available_train")) for r in subset),
        }
    summary["by_candidate_count"] = by_count
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
