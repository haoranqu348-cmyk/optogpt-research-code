"""Predict all generated hard joint s+p benchmark cases."""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from interactive_predictor import InteractivePredictor, plot_comparison
from run_prediction import load_spectrum_file


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CASES_DIR = SCRIPT_DIR / "inputs" / "hard_joint_sp_60cases"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "hard_joint_sp_predictions_20260726"
MODEL = (
    PROJECT_ROOT / "joint_sp" / "formal_checkpoints_500k_v2_20260725_03"
    / "optogpt_joint_sp_500k_v2_best.pt"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    json_dir = output_dir / "json"
    plot_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)

    with (CASES_DIR / "answers.csv").open(newline="") as handle:
        answers = {row["case"]: row for row in csv.DictReader(handle)}
    sources = sorted(CASES_DIR.glob("hard_*_60deg_sp.csv"))
    predictor = InteractivePredictor(model_path=str(MODEL))
    predictor.load_model()
    summary_rows = []
    for index, source in enumerate(sources, 1):
        case = source.stem.replace("_60deg_sp", "")
        print(f"\n[hard {index}/{len(sources)}] {case}", flush=True)
        target = load_spectrum_file(source, joint_sp=True)
        results = predictor.predict_and_validate(
            target, num_candidates=args.candidates, seed=args.seed,
        )
        if not results:
            summary_rows.append({"case": case, "status": "no_valid_candidate"})
            continue
        best = results[0]
        best.update({
            "theta_deg": predictor.theta_deg,
            "pol": predictor.pol,
            "substrate": predictor.substrate,
            "info": f"| case={case} | model={Path(predictor.model_path).stem}",
        })
        png_path = plot_dir / f"{case}.png"
        json_path = json_dir / f"{case}.json"
        plot_comparison(best, save_path=str(png_path), show=False)
        answer = answers[case]
        data = {
            "timestamp": datetime.now().isoformat(),
            "case": case,
            "model": str(predictor.model_path),
            "source_spectrum": str(source),
            "theta_deg": predictor.theta_deg,
            "pol": predictor.pol,
            "num_unique_candidates": len(results),
            "mae_total": best["mae_total"], "mae_s": best["mae_s"], "mae_p": best["mae_p"],
            "mae_Rs": best["mae_Rs"], "mae_Ts": best["mae_Ts"],
            "mae_Rp": best["mae_Rp"], "mae_Tp": best["mae_Tp"],
            "predicted_structure": {
                "materials": best["materials"],
                "thicknesses_nm": [float(v) for v in best["thicknesses"]],
                "tokens": best["tokens"],
            },
            "known_structure": {
                "materials": answer["materials"].split("|"),
                "thicknesses_nm": [float(v) for v in answer["thicknesses_nm"].split("|")],
            },
            "png": str(png_path),
        }
        data["target"] = {
            "Rs": best["Rs_target"].tolist(), "Ts": best["Ts_target"].tolist(),
            "Rp": best["Rp_target"].tolist(), "Tp": best["Tp_target"].tolist(),
        }
        data["simulated"] = {
            "Rs": best["Rs_sim"].tolist(), "Ts": best["Ts_sim"].tolist(),
            "Rp": best["Rp_sim"].tolist(), "Tp": best["Tp_sim"].tolist(),
        }
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        summary_rows.append({
            "case": case, "status": "ok", "mae_total": best["mae_total"],
            "mae_s": best["mae_s"], "mae_p": best["mae_p"],
            "known_layers": answer["n_layers"], "predicted_layers": best["n_layers"],
            "known_structure": ",".join(
                f"{m}_{t}" for m, t in zip(
                    answer["materials"].split("|"), answer["thicknesses_nm"].split("|")
                )
            ),
            "predicted_structure": ",".join(best["tokens"]),
            "png": str(png_path), "json": str(json_path),
        })

    fields = [
        "case", "status", "mae_total", "mae_s", "mae_p", "known_layers",
        "predicted_layers", "known_structure", "predicted_structure", "png", "json",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": str(MODEL), "cases": len(summary_rows),
        "successful_cases": sum(row["status"] == "ok" for row in summary_rows),
        "candidates_per_case": args.candidates, "seed": args.seed,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nHard-case batch complete: {output_dir}")


if __name__ == "__main__":
    main()
