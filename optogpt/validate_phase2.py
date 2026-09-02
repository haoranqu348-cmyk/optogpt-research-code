"""
Phase 2 Comprehensive Validation: OptoGPT 60deg s-pol.

Features:
  - Multi-candidate decoding (greedy, top-k/p sampling, temperature)
  - TMM physical re-ranking
  - Comparison: original vs fine-tuned model vs nearest-neighbor baseline
  - Statistical reports (CSV, JSON, plots)
  - Grouping by layer count, material types
  - Checkpoint resume & chunked saving
  - Supports RTX 4070 Laptop 8GB (configurable batch sizes)

Usage:
    python validate_phase2.py --test_spec data_60deg_s/Spectrum_test.pkl \\
        --test_struct data_60deg_s/Structure_test.pkl \\
        --model model/optogpt_60deg_s_best.pt \\
        --original_model model/optogpt.pt \\
        --output_dir validation_results/phase2 \\
        --num_samples 100 --num_candidates 32 \\
        --train_spec data_60deg_s/Spectrum_train.pkl \\
        --train_struct data_60deg_s/Structure_train.pkl
"""

import os
import sys
import json
import time
import argparse
import pickle as pkl
import numpy as np
import torch
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # optogpt/ (script is in optogpt/optogpt/)
sys.path.insert(0, str(PROJECT_ROOT / "optogpt"))

from core.models.transformer import make_model_I, subsequent_mask
from core.datasets.sim import spectrum, load_materials, mats as default_mats
from core.datasets.datasets import PAD, UNK

# Multi-candidate decoder
from multi_candidate_decoder import (
    batch_greedy_decode, batch_sampling_decode,
    generate_candidates, parse_structure, is_valid_structure,
    structure_to_tuple, generate_candidates_batch,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed physical conditions
THETA_DEG = 60
POLARIZATION = "s"
WAVELENGTHS = np.arange(0.4, 1.1 + 1e-3, 0.01)
SUBSTRATE = "Glass_Substrate"
SUBSTRATE_THICK = 500000
N_WL = len(WAVELENGTHS)


# ============================================================
# Model loading
# ============================================================

def _get_config(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_model(ckpt_path, pretrained_path=None):
    """Load model and vocabulary. Falls back to pretrained for vocab if needed."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    configs = ckpt.get("configs", {})
    word_dict = _get_config(configs, "struc_word_dict")
    index_dict = _get_config(configs, "struc_index_dict")

    pretrained_configs = {}
    if (word_dict is None) and pretrained_path is not None:
        pretrained = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
        pretrained_configs = pretrained.get("configs", {})
        word_dict = _get_config(pretrained_configs, "struc_word_dict", {})
        index_dict = _get_config(pretrained_configs, "struc_index_dict", {})

    if not word_dict:
        raise ValueError("Checkpoint missing struc_word_dict and no pretrained fallback")

    def _get(key, default):
        val = _get_config(configs, key)
        if val is None:
            val = _get_config(pretrained_configs, key, default)
        return val

    model = make_model_I(
        src_vocab=_get("spec_dim", 142),
        tgt_vocab=_get("struc_dim", len(word_dict)),
        N=_get("layers", 6),
        d_model=_get("d_model", 1024),
        d_ff=_get("d_ff", 512),
        h=_get("head_num", 8),
        dropout=_get("dropout", 0.1),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, word_dict, index_dict


# ============================================================
# TMM Simulation & Re-ranking
# ============================================================

def tmm_simulate(materials, thicknesses, nk_dict, theta_deg=THETA_DEG,
                  pol=POLARIZATION, wavelengths=WAVELENGTHS,
                  substrate=SUBSTRATE, substrate_thick=SUBSTRATE_THICK):
    """Run TMM simulation. Returns (R, T, success, error_msg)."""
    try:
        result = spectrum(
            materials=materials, thickness=thicknesses,
            pol=pol, theta=theta_deg, wavelengths=wavelengths,
            nk_dict=nk_dict, substrate=substrate,
            substrate_thick=substrate_thick,
        )
        half = len(result) // 2
        R = np.array(result[:half], dtype=np.float64)
        T = np.array(result[half:], dtype=np.float64)
        if np.any(np.isnan(R)) or np.any(np.isnan(T)):
            return None, None, False, "NaN in result"
        if np.any(np.isinf(R)) or np.any(np.isinf(T)):
            return None, None, False, "Inf in result"
        return R, T, True, "ok"
    except Exception as e:
        return None, None, False, str(e)


def compute_errors(R_sim, T_sim, R_target, T_target):
    """Compute all error metrics."""
    R_sim = np.asarray(R_sim, dtype=np.float64)
    T_sim = np.asarray(T_sim, dtype=np.float64)
    R_target = np.asarray(R_target, dtype=np.float64)
    T_target = np.asarray(T_target, dtype=np.float64)

    mae_R = np.mean(np.abs(R_sim - R_target))
    mae_T = np.mean(np.abs(T_sim - T_target))
    # Total MAE = mean over all 142 spectral points (R+T concatenated)
    total_mae = np.mean(np.abs(
        np.concatenate([R_sim, T_sim]) - np.concatenate([R_target, T_target])
    ))
    max_R_err = np.max(np.abs(R_sim - R_target))
    max_T_err = np.max(np.abs(T_sim - T_target))
    avg_A = np.mean(1 - R_sim - T_sim)  # average absorption A = 1 - R - T

    return {
        "mae_R": float(mae_R),
        "mae_T": float(mae_T),
        "total_mae": float(total_mae),
        "max_R_err": float(max_R_err),
        "max_T_err": float(max_T_err),
        "avg_absorption": float(avg_A),
    }


def tmm_rerank(candidates, spec_target, nk_dict):
    """Re-rank candidates by TMM simulation Total MAE."""
    R_target = np.array(spec_target[:N_WL])
    T_target = np.array(spec_target[N_WL:])

    ranked = []
    for cand in candidates:
        R_sim, T_sim, ok, err = tmm_simulate(
            cand["materials"], cand["thicknesses"], nk_dict)
        if not ok:
            cand["tmm_error"] = err
            cand["tmm_success"] = False
            ranked.append(cand)
            continue

        errors = compute_errors(R_sim, T_sim, R_target, T_target)
        cand.update({
            "R_sim": R_sim.tolist(),
            "T_sim": T_sim.tolist(),
            "tmm_success": True,
            **errors,
        })
        ranked.append(cand)

    # Sort by total_mae (put failures at end)
    ranked.sort(key=lambda c: c.get("total_mae", 1e9))
    return ranked


# ============================================================
# Nearest Neighbor Baseline
# ============================================================

def nearest_neighbor_baseline(spec_target, train_specs, train_structs, nk_dict):
    """Find closest training spectrum and compute its TMM errors."""
    spec_target = np.array(spec_target)
    train_specs = np.array(train_specs)
    maes = np.mean(np.abs(train_specs - spec_target), axis=1)
    best_idx = np.argmin(maes)

    best_struct = train_structs[best_idx]
    mats, thick = parse_structure(best_struct)

    R_target = spec_target[:N_WL]
    T_target = spec_target[N_WL:]
    R_sim, T_sim, ok, err = tmm_simulate(mats, thick, nk_dict)
    if not ok:
        return None

    errors = compute_errors(R_sim, T_sim, R_target, T_target)
    return {
        "best_train_idx": int(best_idx),
        "tokens": best_struct,
        "materials": mats,
        "thicknesses": thick,
        "n_layers": len(mats),
        "total_thickness": sum(thick),
        **errors,
    }


# ============================================================
# Statistics
# ============================================================

def compute_stats(values, prefix=""):
    """Compute descriptive statistics for a list of values."""
    if not values:
        return {}
    arr = np.array(values)
    return {
        f"{prefix}mean": float(np.mean(arr)),
        f"{prefix}median": float(np.median(arr)),
        f"{prefix}std": float(np.std(arr)),
        f"{prefix}p90": float(np.percentile(arr, 90)),
        f"{prefix}p95": float(np.percentile(arr, 95)),
        f"{prefix}max": float(np.max(arr)),
        f"{prefix}min": float(np.min(arr)),
    }


def group_stats(results, group_key_fn):
    """Group results by a key function and compute per-group statistics."""
    groups = defaultdict(list)
    for r in results:
        key = group_key_fn(r)
        groups[key].append(r)
    group_stats = {}
    for key, items in groups.items():
        mae_t = [it.get("total_mae", 1.0) for it in items]
        mae_r = [it.get("mae_R", 1.0) for it in items]
        mae_t_vals = [it.get("mae_T", 1.0) for it in items]
        group_stats[str(key)] = {
            "count": len(items),
            **compute_stats(mae_t, "total_mae_"),
            **compute_stats(mae_r, "R_mae_"),
            **compute_stats(mae_t_vals, "T_mae_"),
        }
    return group_stats


def plot_comparison_curves(all_model_results, save_path):
    """Plot MAE improvement with more candidates."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    configs_order = ["greedy_N1", "sampling_N8", "sampling_N32", "sampling_N64"]
    labels = ["Greedy (N=1)", "Sampling (N=8)", "Sampling (N=32)", "Sampling (N=64)"]

    for ax, metric in zip(axes, ["total_mae", "mae_R", "mae_T"]):
        for model_name, config_results in all_model_results.items():
            means = []
            for cfg in configs_order:
                if cfg in config_results:
                    vals = [r.get(metric, 1.0) for r in config_results[cfg] if r is not None]
                    means.append(np.mean(vals) if vals else None)
                else:
                    means.append(None)
            ax.plot(range(len(labels)), means, "o-", label=model_name, linewidth=2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_ylabel(f"{metric} (mean)")
        ax.set_title(f"{metric} vs candidates")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("MAE Improvement with Multi-Candidate Decoding (θ=60°, s-pol)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison curves saved: {save_path}")


# ============================================================
# Main validation routine
# ============================================================

def run_validation_for_model(model, word_dict, index_dict, nk_dict,
                              test_specs, test_structs, train_specs, train_structs,
                              configs, model_name, output_dir,
                              n_samples=100, start_idx=0):
    """
    Run comprehensive validation for one model across multiple decoding configs.
    Saves intermediate results per chunk.
    """
    max_len = _get_config(configs if isinstance(configs, dict) else vars(configs),
                          "max_len", 22)
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    decoding_configs = [
        ("greedy_N1", "greedy", 1),
        ("sampling_N8", "sampling", 8),
        ("sampling_N32", "sampling", 32),
        ("sampling_N64", "sampling", 64),
    ]

    all_results = {cfg_name: [] for cfg_name, _, _ in decoding_configs}

    end_idx = min(start_idx + n_samples, len(test_specs))
    total = end_idx - start_idx

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Samples: {start_idx} to {end_idx} ({total} total)")
    print(f"{'='*60}")

    for i in range(start_idx, end_idx):
        spec_target = test_specs[i]
        gt_struct = test_structs[i] if test_structs is not None else None
        R_target = np.array(spec_target[:N_WL])
        T_target = np.array(spec_target[N_WL:])

        if (i - start_idx) % 10 == 0:
            print(f"  [{model_name}] Sample {i - start_idx + 1}/{total}...")

        for cfg_name, mode, n_cand in decoding_configs:
            # Generate candidates
            if mode == "greedy":
                candidates = generate_candidates(
                    model, spec_target, word_dict, index_dict,
                    num_candidates=1, max_len=max_len, max_layers=max_len-2,
                    top_k=10, top_p=0.9, temperature=1.0,
                    nk_dict=nk_dict, device=DEVICE, decode_batch_size=8,
                )
            else:
                candidates = generate_candidates(
                    model, spec_target, word_dict, index_dict,
                    num_candidates=n_cand, max_len=max_len, max_layers=max_len-2,
                    top_k=10, top_p=0.9, temperature=1.0,
                    nk_dict=nk_dict, device=DEVICE, decode_batch_size=8,
                )

            # TMM re-rank
            ranked = tmm_rerank(candidates, spec_target, nk_dict)
            valid_ranked = [c for c in ranked if c.get("tmm_success")]

            # Nearest neighbor baseline (only once per target, use first config)
            if cfg_name == "greedy_N1" and train_specs is not None:
                nn_result = nearest_neighbor_baseline(
                    spec_target, train_specs, train_structs, nk_dict)

            # Build result entry
            best = valid_ranked[0] if valid_ranked else None
            entry = {
                "sample_idx": i,
                "gt_structure": gt_struct,
                "n_candidates_generated": len(candidates),
                "n_valid_candidates": len(valid_ranked),
                "n_unique_after_dedup": len(candidates),
                "has_valid": len(valid_ranked) > 0,
            }
            if best:
                entry.update({
                    "best_tokens": best["tokens"],
                    "best_materials": best["materials"],
                    "best_thicknesses": best["thicknesses"],
                    "best_n_layers": len(best["materials"]),
                    "best_total_thickness": sum(best["thicknesses"]),
                    "mae_R": best["mae_R"],
                    "mae_T": best["mae_T"],
                    "total_mae": best["total_mae"],
                    "max_R_err": best["max_R_err"],
                    "max_T_err": best["max_T_err"],
                    "avg_absorption": best["avg_absorption"],
                })
            # Also store top-5 candidates
            if valid_ranked:
                entry["top5_candidates"] = [
                    {
                        "tokens": c["tokens"],
                        "total_mae": c.get("total_mae", 1.0),
                        "mae_R": c.get("mae_R", 1.0),
                        "mae_T": c.get("mae_T", 1.0),
                    }
                    for c in valid_ranked[:5]
                ]
            all_results[cfg_name].append(entry)

        # Nearest neighbor entry
        if nn_result:
            nn_entry = {
                "sample_idx": i,
                "method": "nearest_neighbor",
                **nn_result,
            }
            if "nn_results" not in all_results:
                all_results["nn_results"] = []
            all_results["nn_results"].append(nn_entry)

    # Compute statistics per config
    stats = {}
    for cfg_name, results in all_results.items():
        valid_results = [r for r in results if r.get("has_valid")]
        if not valid_results:
            stats[cfg_name] = {"n_total": len(results), "n_valid": 0}
            continue

        mae_totals = [r["total_mae"] for r in valid_results]
        mae_Rs = [r["mae_R"] for r in valid_results]
        mae_Ts = [r["mae_T"] for r in valid_results]
        n_layers = [r.get("best_n_layers", 0) for r in valid_results]

        stats[cfg_name] = {
            "n_total": len(results),
            "n_valid": len(valid_results),
            "valid_rate": len(valid_results) / max(len(results), 1),
            "tmm_success_rate": len(valid_results) / max(len(results), 1),
            **compute_stats(mae_totals, "total_mae_"),
            **compute_stats(mae_Rs, "R_mae_"),
            **compute_stats(mae_Ts, "T_mae_"),
            "avg_n_layers": float(np.mean(n_layers)),
            "avg_candidates": float(np.mean([
                r.get("n_valid_candidates", 0) for r in valid_results])),
        }

    return all_results, stats


def main():
    parser = argparse.ArgumentParser(description="Phase 2 comprehensive validation")
    parser.add_argument("--test_spec", type=str, default="data_60deg_s/Spectrum_test.pkl")
    parser.add_argument("--test_struct", type=str, default="data_60deg_s/Structure_test.pkl")
    parser.add_argument("--train_spec", type=str, default="data_60deg_s/Spectrum_train.pkl")
    parser.add_argument("--train_struct", type=str, default="data_60deg_s/Structure_train.pkl")
    parser.add_argument("--model", type=str, default="model/optogpt_60deg_s_best.pt")
    parser.add_argument("--original_model", type=str, default="model/optogpt.pt")
    parser.add_argument("--output_dir", type=str, default="validation_results/phase2")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing results JSON to resume from")
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="Force CPU execution (avoids CUDA indexing issues)")
    args = parser.parse_args()

    if args.cpu:
        global DEVICE
        DEVICE = torch.device("cpu")
        # Also tell the decoder module to use CPU
        import multi_candidate_decoder as mcd
        mcd.DEVICE = torch.device("cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve paths
    test_spec_path = (PROJECT_ROOT / args.test_spec).resolve()
    test_struct_path = (PROJECT_ROOT / args.test_struct).resolve() if args.test_struct else None
    train_spec_path = (PROJECT_ROOT / args.train_spec).resolve() if args.train_spec else None
    train_struct_path = (PROJECT_ROOT / args.train_struct).resolve() if args.train_struct else None
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Output: {output_dir}")

    # Load test data
    with open(test_spec_path, "rb") as f:
        test_specs = pkl.load(f)
    if not isinstance(test_specs, np.ndarray):
        test_specs = np.array(test_specs)

    test_structs = None
    if test_struct_path and test_struct_path.exists():
        with open(test_struct_path, "rb") as f:
            test_structs = pkl.load(f)

    # Load train data for nearest neighbor
    train_specs = None
    train_structs = None
    if train_spec_path and train_spec_path.exists():
        with open(train_spec_path, "rb") as f:
            train_specs = pkl.load(f)
    if train_struct_path and train_struct_path.exists():
        with open(train_struct_path, "rb") as f:
            train_structs = pkl.load(f)

    # Load nk data
    print("Loading nk data...")
    nk_dir = str(PROJECT_ROOT / "optogpt" / "nk")
    nk_dict = load_materials(all_mats=default_mats, wavelengths=WAVELENGTHS,
                              DATABASE=nk_dir)
    substrate_nk = load_materials(all_mats=[SUBSTRATE], wavelengths=WAVELENGTHS,
                                   DATABASE=nk_dir)
    nk_dict.update(substrate_nk)

    # Models to evaluate
    pretrained_path = str((PROJECT_ROOT / args.original_model).resolve())
    model_configs = [
        ("fine_tuned", str((PROJECT_ROOT / args.model).resolve()), pretrained_path),
        ("original", pretrained_path, None),
    ]

    all_model_results = {}
    all_model_stats = {}
    nn_results_global = None

    for model_name, ckpt_path, pretrained_fb in model_configs:
        if not Path(ckpt_path).exists():
            print(f"SKIP: {model_name} not found at {ckpt_path}")
            continue

        print(f"\nLoading {model_name} model: {ckpt_path}")
        model, word_dict, index_dict = load_model(ckpt_path, pretrained_fb)

        results, stats = run_validation_for_model(
            model, word_dict, index_dict, nk_dict,
            test_specs, test_structs, train_specs, train_structs,
            {}, model_name, output_dir / model_name,
            n_samples=args.num_samples, start_idx=args.start_idx,
        )

        all_model_results[model_name] = results
        all_model_stats[model_name] = stats

        # First model's NN results is the global baseline
        if nn_results_global is None and "nn_results" in results:
            nn_results_global = results["nn_results"]

        # Save per-model results
        with open(output_dir / model_name / "results.json", "w") as f:
            json.dump({"stats": stats, "results": results}, f, indent=2, default=str)

    # Add NN baseline to stats
    if nn_results_global:
        valid_nn = [r for r in nn_results_global if r is not None]
        if valid_nn:
            nn_stats = {
                "n_total": len(nn_results_global),
                "n_valid": len(valid_nn),
                **compute_stats([r["total_mae"] for r in valid_nn], "total_mae_"),
                **compute_stats([r["mae_R"] for r in valid_nn], "R_mae_"),
                **compute_stats([r["mae_T"] for r in valid_nn], "T_mae_"),
            }
            all_model_stats["nearest_neighbor"] = nn_stats
            all_model_results["nearest_neighbor"] = {"nn_results": nn_results_global}

    # ---- Summary CSV ----
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["model", "config", "n_total", "n_valid", "valid_rate",
                   "total_mae_mean", "total_mae_median", "total_mae_std",
                   "total_mae_p90", "total_mae_p95", "total_mae_max",
                   "R_mae_mean", "R_mae_std", "T_mae_mean", "T_mae_std"]
        writer.writerow(header)
        for model_name, stats in all_model_stats.items():
            if model_name == "nearest_neighbor":
                s = stats
                writer.writerow([
                    "nearest_neighbor", "train_retrieval",
                    s.get("n_total", 0), s.get("n_valid", 0), "",
                    s.get("total_mae_mean", ""), s.get("total_mae_median", ""),
                    s.get("total_mae_std", ""), s.get("total_mae_p90", ""),
                    s.get("total_mae_p95", ""), s.get("total_mae_max", ""),
                    s.get("R_mae_mean", ""), s.get("R_mae_std", ""),
                    s.get("T_mae_mean", ""), s.get("T_mae_std", ""),
                ])
            else:
                for cfg_name, s in stats.items():
                    writer.writerow([
                        model_name, cfg_name,
                        s.get("n_total", 0), s.get("n_valid", 0),
                        s.get("valid_rate", ""),
                        s.get("total_mae_mean", ""), s.get("total_mae_median", ""),
                        s.get("total_mae_std", ""), s.get("total_mae_p90", ""),
                        s.get("total_mae_p95", ""), s.get("total_mae_max", ""),
                        s.get("R_mae_mean", ""), s.get("R_mae_std", ""),
                        s.get("T_mae_mean", ""), s.get("T_mae_std", ""),
                    ])
    print(f"\nSummary CSV: {csv_path}")

    # ---- Group statistics ----
    # Combine all fine-tuned valid results for grouping
    ft_valid = []
    if "fine_tuned" in all_model_results:
        for cfg_results in all_model_results["fine_tuned"].values():
            if isinstance(cfg_results, list):
                ft_valid.extend([r for r in cfg_results if r.get("has_valid")])

    if ft_valid:
        # By layer count
        def layer_group(r):
            n = r.get("best_n_layers", 0)
            if n <= 1: return "1_layer"
            if n <= 3: return "2-3_layers"
            if n <= 6: return "4-6_layers"
            return "7+_layers"
        layer_stats = group_stats(ft_valid, layer_group)

        # By metal presence
        def metal_group(r):
            mats = r.get("best_materials", [])
            has_ag = "Ag" in mats
            has_tin = "TiN" in mats
            has_al = "Al" in mats
            if has_ag or has_tin or has_al:
                parts = []
                if has_ag: parts.append("Ag")
                if has_tin: parts.append("TiN")
                if has_al: parts.append("Al")
                return "+".join(parts)
            return "no_absorber"
        metal_stats = group_stats(ft_valid, metal_group)

        with open(output_dir / "grouped_stats.json", "w") as f:
            json.dump({
                "by_layer_count": layer_stats,
                "by_absorber": metal_stats,
            }, f, indent=2, default=str)

    # ---- Comparison plot ----
    plot_path = output_dir / "comparison_curves.png"
    plot_comparison_curves(all_model_results, str(plot_path))

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print("PHASE 2 VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total MAE definition: mean(|R_sim-R_target|, |T_sim-T_target|) over all 142 points")
    print()

    for model_name, stats in all_model_stats.items():
        print(f"--- {model_name} ---")
        if isinstance(stats, dict) and "total_mae_mean" in stats:
            # Direct stats dict (e.g., nearest_neighbor)
            print(f"  Total MAE: mean={stats['total_mae_mean']:.4f}, "
                  f"std={stats['total_mae_std']:.4f}, "
                  f"P95={stats['total_mae_p95']:.4f}")
        else:
            for cfg_name, s in stats.items():
                if isinstance(s, dict) and "total_mae_mean" in s:
                    print(f"  {cfg_name}: Total MAE mean={s['total_mae_mean']:.4f}, "
                          f"std={s['total_mae_std']:.4f}, "
                          f"valid={s.get('n_valid',0)}/{s.get('n_total',0)}")

    print(f"\nAll results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
