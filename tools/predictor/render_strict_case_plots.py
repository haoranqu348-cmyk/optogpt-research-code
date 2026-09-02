"""Render the same joint s+p comparison plots for strict evaluation cases."""

import argparse
import csv
import json
from pathlib import Path

from batch_predict_hard_joint_cases import MODEL
from interactive_predictor import InteractivePredictor, plot_comparison
from run_prediction import load_spectrum_file


SCRIPT_DIR = Path(__file__).resolve().parent
CASES_DIR = SCRIPT_DIR / "inputs" / "strict_joint_sp_eval_20260726"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "strict_joint_sp_eval_20260726"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, default=CASES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    plot_dir = out / "plots"
    json_dir = out / "json"
    plot_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    with (args.cases_dir / "answers.csv").open(newline="") as handle:
        answers = {row["case"]: row for row in csv.DictReader(handle)}

    sources = sorted(args.cases_dir.glob("strict_*_60deg_sp.csv"))
    predictor = InteractivePredictor(model_path=str(MODEL))
    predictor.load_model()
    for index, source in enumerate(sources, 1):
        case = source.stem.replace("_60deg_sp", "")
        print(f"[{index}/{len(sources)}] {case}", flush=True)
        target = load_spectrum_file(source, joint_sp=True)
        results = predictor.predict_and_validate(
            target, num_candidates=args.candidates, seed=args.seed + index,
        )
        if not results:
            continue
        best = results[0]
        best.update({
            "theta_deg": predictor.theta_deg,
            "pol": predictor.pol,
            "substrate": predictor.substrate,
            "info": f"| case={case} | model={Path(predictor.model_path).stem} | candidates={args.candidates}",
        })
        png_path = plot_dir / f"{case}.png"
        plot_comparison(best, save_path=str(png_path), show=False)
        answer = answers[case]
        data = {
            "case": case,
            "model": str(predictor.model_path),
            "source_spectrum": str(source),
            "candidate_count": args.candidates,
            "seed": args.seed + index,
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
            "target": {
                "Rs": best["Rs_target"].tolist(), "Ts": best["Ts_target"].tolist(),
                "Rp": best["Rp_target"].tolist(), "Tp": best["Tp_target"].tolist(),
            },
            "simulated": {
                "Rs": best["Rs_sim"].tolist(), "Ts": best["Ts_sim"].tolist(),
                "Rp": best["Rp_sim"].tolist(), "Tp": best["Tp_sim"].tolist(),
            },
            "png": str(png_path),
        }
        (json_dir / f"{case}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Rendered strict plots in {plot_dir}")


if __name__ == "__main__":
    main()
