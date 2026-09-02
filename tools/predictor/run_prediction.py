"""一键预测脚本 — 直接调用 interactive_predictor 核心函数
用法: python run_prediction.py [spectrum_file] [--model model_path] [--theta 0] [--pol s] [--candidates 8]
"""
import sys, os, traceback, json, argparse
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "optogpt" / "optogpt"))
sys.path.insert(0, str(PROJECT_ROOT / "optogpt"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from interactive_predictor import (
    InteractivePredictor, plot_comparison,
    OUTPUT_DIR, INPUT_DIR, DEVICE, _parse_and_interpolate_csv, _parse_joint_csv
)


def resolve_csv_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    return (INPUT_DIR / path).resolve()


def load_spectrum_file(path, joint_sp=False, index=0):
    """Load a single spectrum using the format required by the model."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, "r") as handle:
            parser = _parse_joint_csv if joint_sp else _parse_and_interpolate_csv
            return parser(handle.read())
    if suffix == ".npy":
        values = np.load(path)
    elif suffix == ".pkl":
        with open(path, "rb") as handle:
            values = pickle.load(handle)
    else:
        raise ValueError("光谱文件必须是 .csv、.pkl 或 .npy")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        return values
    if not 0 <= index < len(values):
        raise IndexError(f"光谱索引 {index} 超出范围 [0, {len(values) - 1}]")
    return values[index]


def main():
    parser = argparse.ArgumentParser(description="一键预测")
    parser.add_argument("spectrum", nargs="?", default="target_spectrum.csv",
                        help="光谱文件 .csv/.pkl/.npy (默认: inputs/target_spectrum.csv)")
    parser.add_argument("--index", type=int, default=0,
                        help=".pkl/.npy 中的样本索引")
    parser.add_argument("--model", default=None,
                        help="模型路径 (默认: optogpt/model/optogpt.pt)")
    parser.add_argument("--pretrained", default=None,
                        help="词表回退 checkpoint (默认: optogpt/model/optogpt.pt)")
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--pol", choices=["s", "p"], default="s")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-layers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None,
                        help="输出目录 (默认: predictor/outputs)")
    parser.add_argument("--architecture", choices=["auto", "legacy", "relu"],
                        default="auto",
                        help="Checkpoint 前向架构；旧原始模型用 legacy，当前微调模型用 relu")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("  OptoGPT 一键预测", flush=True)
    print("=" * 60, flush=True)

    # 模型
    model_path = args.model or str(PROJECT_ROOT / "optogpt" / "model" / "optogpt.pt")
    pretrained_path = args.pretrained or str(
        PROJECT_ROOT / "optogpt" / "model" / "optogpt.pt"
    )
    print(f"\n[1] 加载模型: {model_path}", flush=True)
    predictor = InteractivePredictor(
        model_path=model_path,
        pretrained_path=pretrained_path,
        theta_deg=args.theta,
        pol=args.pol,
        architecture=args.architecture,
    )
    predictor.load_model()

    # 光谱
    spectrum_path = resolve_csv_path(args.spectrum)
    print(f"\n[2] 加载目标光谱: {spectrum_path}", flush=True)
    target_spec = load_spectrum_file(
        spectrum_path, joint_sp=predictor.is_joint_sp, index=args.index
    )
    layout = "Rs/Ts/Rp/Tp" if predictor.is_joint_sp else "R/T"
    print(f"  光谱维度: {len(target_spec)} ({layout})", flush=True)

    # 预测
    theta, pol = predictor.theta_deg, predictor.pol
    print(f"\n[3] 生成 {args.candidates} 个候选 (θ={theta}°, {pol})...", flush=True)
    results = predictor.predict_and_validate(
        target_spec,
        num_candidates=args.candidates,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_layers=args.max_layers,
        seed=args.seed,
    )

    if not results:
        print("❌ 无有效候选!", flush=True)
        return

    results.sort(key=lambda x: x["mae_total"])
    best = results[0]
    best["theta_deg"] = theta
    best["pol"] = pol
    best["substrate"] = "Glass_Substrate"
    pol_text = "s+p" if predictor.is_joint_sp else f"{pol}-pol"
    best["info"] = f"| θ={theta}° {pol_text} | model={Path(model_path).stem}"

    print(f"\n[4] 最佳结构 (MAE_total={best['mae_total']:.4f}):", flush=True)
    if predictor.is_joint_sp:
        print(f"   s-pol MAE: {best['mae_s']:.4f}  |  "
              f"p-pol MAE: {best['mae_p']:.4f}", flush=True)
    else:
        print(f"   R MAE: {best['mae_R']:.4f}  |  T MAE: {best['mae_T']:.4f}", flush=True)
    for i, (m, t) in enumerate(zip(best["materials"], best["thicknesses"])):
        print(f"   层{i+1}: {m:8s}  {t:6.0f} nm", flush=True)

    # 保存
    output_dir = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = str(output_dir / f"prediction_{ts}.png")
    json_path = str(output_dir / f"prediction_{ts}.json")

    print(f"\n[5] 保存...", flush=True)
    plot_comparison(best, save_path=png_path, show=False)
    print(f"   图像: {png_path}", flush=True)

    json_data = {
        "timestamp": datetime.now().isoformat(), "model": model_path,
        "theta_deg": theta, "pol": pol,
        "mae_R": best["mae_R"], "mae_T": best["mae_T"], "mae_total": best["mae_total"],
        "n_layers": best["n_layers"],
        "decode_method": best["decode_method"],
        "materials": best["materials"],
        "thicknesses": [float(t) for t in best["thicknesses"]],
        "tokens": best["tokens"],
        "num_unique_candidates": len(results),
        "sampling": {
            "requested_candidates": args.candidates,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "max_layers": args.max_layers,
            "seed": args.seed,
        },
    }
    if predictor.is_joint_sp:
        json_data.update({
            "model_type": "joint_sp",
            "mae_s": best["mae_s"], "mae_p": best["mae_p"],
            "mae_Rs": best["mae_Rs"], "mae_Ts": best["mae_Ts"],
            "mae_Rp": best["mae_Rp"], "mae_Tp": best["mae_Tp"],
            "target": {
                "Rs": best["Rs_target"].tolist(),
                "Ts": best["Ts_target"].tolist(),
                "Rp": best["Rp_target"].tolist(),
                "Tp": best["Tp_target"].tolist(),
            },
            "simulated": {
                "Rs": best["Rs_sim"].tolist(),
                "Ts": best["Ts_sim"].tolist(),
                "Rp": best["Rp_sim"].tolist(),
                "Tp": best["Tp_sim"].tolist(),
            },
        })
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"   JSON: {json_path}", flush=True)

    # 可复制的结构字符串
    struct_str = ",".join(f"{m}_{int(t)}" for m, t in zip(best["materials"], best["thicknesses"]))
    print(f"\n  结构: {struct_str}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
