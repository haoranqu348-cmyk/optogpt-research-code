"""Run every model-ready spectrum in the optical test pack."""

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
PACK_DIR = SCRIPT_DIR / "inputs" / "optical_test_pack"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "optical_test_pack_predictions_20260726"
JOINT_MODEL = (
    PROJECT_ROOT / "joint_sp" / "formal_checkpoints_500k_v2_20260725_03"
    / "optogpt_joint_sp_500k_v2_best.pt"
)
SINGLE_MODEL = PROJECT_ROOT / "optogpt" / "model" / "optogpt.pt"


def read_answers():
    with (PACK_DIR / "answers.csv").open(newline="") as handle:
        return {row["case"]: row for row in csv.DictReader(handle)}


def make_json_data(case, mode, source, predictor, best, candidates, answer):
    data = {
        "timestamp": datetime.now().isoformat(),
        "case": case,
        "mode": mode,
        "source_spectrum": str(source),
        "model": str(predictor.model_path),
        "theta_deg": predictor.theta_deg,
        "pol": predictor.pol,
        "mae_R": best["mae_R"],
        "mae_T": best["mae_T"],
        "mae_total": best["mae_total"],
        "n_layers": best["n_layers"],
        "decode_method": best["decode_method"],
        "materials": best["materials"],
        "thicknesses": [float(value) for value in best["thicknesses"]],
        "tokens": best["tokens"],
        "num_unique_candidates": len(candidates),
        "known_structure": {
            "materials": answer["materials"].split("|"),
            "thicknesses_nm": [float(value) for value in answer["thicknesses_nm"].split("|")],
        },
    }
    if predictor.is_joint_sp:
        data.update({
            "model_type": "joint_sp",
            "mae_s": best["mae_s"], "mae_p": best["mae_p"],
            "mae_Rs": best["mae_Rs"], "mae_Ts": best["mae_Ts"],
            "mae_Rp": best["mae_Rp"], "mae_Tp": best["mae_Tp"],
            "target": {
                "Rs": best["Rs_target"].tolist(), "Ts": best["Ts_target"].tolist(),
                "Rp": best["Rp_target"].tolist(), "Tp": best["Tp_target"].tolist(),
            },
            "simulated": {
                "Rs": best["Rs_sim"].tolist(), "Ts": best["Ts_sim"].tolist(),
                "Rp": best["Rp_sim"].tolist(), "Tp": best["Tp_sim"].tolist(),
            },
        })
    return data


def run_group(predictor, files, mode, output_dir, candidates, seed, answers):
    predictor.load_model()
    rows = []
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(files, 1):
        case = source.stem.replace("_60deg_sp", "").replace("_0deg", "")
        print(f"\n[{mode} {index}/{len(files)}] {case}", flush=True)
        target = load_spectrum_file(source, joint_sp=predictor.is_joint_sp)
        results = predictor.predict_and_validate(
            target, num_candidates=candidates, seed=seed,
        )
        if not results:
            rows.append({"case": case, "mode": mode, "status": "no_valid_candidate"})
            continue
        best = results[0]
        best.update({
            "theta_deg": predictor.theta_deg,
            "pol": predictor.pol,
            "substrate": predictor.substrate,
            "info": f"| case={case} | model={Path(predictor.model_path).stem}",
        })
        png_path = mode_dir / f"{case}.png"
        json_path = mode_dir / f"{case}.json"
        plot_comparison(best, save_path=str(png_path), show=False)
        data = make_json_data(
            case, mode, source, predictor, best, results, answers[case]
        )
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        rows.append({
            "case": case,
            "mode": mode,
            "status": "ok",
            "mae_total": best["mae_total"],
            "mae_s": best.get("mae_s", ""),
            "mae_p": best.get("mae_p", ""),
            "mae_R": best["mae_R"],
            "mae_T": best["mae_T"],
            "known_layers": len(answers[case]["materials"].split("|")),
            "predicted_layers": best["n_layers"],
            "known_structure": ",".join(
                f"{material}_{thickness}" for material, thickness in zip(
                    answers[case]["materials"].split("|"),
                    answers[case]["thicknesses_nm"].split("|"),
                )
            ),
            "predicted_structure": ",".join(best["tokens"]),
            "png": str(png_path),
            "json": str(json_path),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    answers = read_answers()

    joint_predictor = InteractivePredictor(model_path=str(JOINT_MODEL))
    joint_files = sorted((PACK_DIR / "joint_sp").glob("*.csv"))
    rows = run_group(
        joint_predictor, joint_files, "joint_sp", output_dir,
        args.candidates, args.seed, answers,
    )

    single_predictor = InteractivePredictor(
        model_path=str(SINGLE_MODEL), theta_deg=0, pol="s"
    )
    single_files = sorted((PACK_DIR / "single_pol").glob("*.csv"))
    rows.extend(run_group(
        single_predictor, single_files, "single_pol", output_dir,
        args.candidates, args.seed, answers,
    ))

    fields = [
        "case", "mode", "status", "mae_total", "mae_s", "mae_p", "mae_R",
        "mae_T", "known_layers", "predicted_layers", "known_structure",
        "predicted_structure", "png", "json",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "candidates_per_case": args.candidates,
        "seed": args.seed,
        "successful_cases": sum(row.get("status") == "ok" for row in rows),
        "total_cases": len(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\nBatch complete: {output_dir}")


if __name__ == "__main__":
    main()
