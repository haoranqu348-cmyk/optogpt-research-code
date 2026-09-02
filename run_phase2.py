"""
Phase 2 Runner - Self-contained execution script.

This script performs:
  1. Data leakage audit
  2. Comprehensive validation (multi-candidate, TMM re-rank)
  3. High-transmittance test
  4. Checkpoint optimization
  5. Minimal tests

All results saved to validation_results/phase2/
"""
import os, sys, json, time, csv, pickle as pkl
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Setup paths
BASE_DIR = Path(__file__).resolve().parent  # optogpt/
SCRIPT_DIR = BASE_DIR / "optogpt"  # optogpt/optogpt/
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.models.transformer import make_model_I, subsequent_mask
from core.datasets.sim import spectrum, load_materials
from core.datasets.datasets import PAD, UNK
from multi_candidate_decoder import (
    batch_greedy_decode, batch_sampling_decode,
    generate_candidates, parse_structure, is_valid_structure,
    structure_to_tuple, tmm_rerank,
)
from audit_data import audit_data

DEVICE = torch.device("cpu")  # Force CPU for reliability

THETA_DEG = 60
POLARIZATION = "s"
WAVELENGTHS = np.arange(0.4, 1.1 + 1e-3, 0.01)
SUBSTRATE = "Glass_Substrate"
SUBSTRATE_THICK = 500000
N_WL = len(WAVELENGTHS)
NK_DIR = str(SCRIPT_DIR / "nk")

# File paths
DATA_DIR = BASE_DIR / "data_60deg_s"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "validation_results" / "phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRETRAINED_PATH = MODEL_DIR / "optogpt.pt"
FINETUNED_PATH = MODEL_DIR / "optogpt_60deg_s_best.pt"


def load_model(ckpt_path, pretrained_path=None):
    """Load model with vocab fallback."""
    print(f"  Loading: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    configs = ckpt.get("configs", {})

    def _get(cfg, key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    word_dict = _get(configs, "struc_word_dict")
    index_dict = _get(configs, "struc_index_dict")

    pretrained_cfg = {}
    if (word_dict is None) and pretrained_path:
        p = torch.load(str(pretrained_path), map_location="cpu", weights_only=False)
        pretrained_cfg = p.get("configs", {})
        word_dict = _get(pretrained_cfg, "struc_word_dict", {})
        index_dict = _get(pretrained_cfg, "struc_index_dict", {})

    if not word_dict:
        raise ValueError("No vocabulary found")

    model = make_model_I(
        src_vocab=_get(configs, "spec_dim", _get(pretrained_cfg, "spec_dim", 142)),
        tgt_vocab=_get(configs, "struc_dim", _get(pretrained_cfg, "struc_dim", len(word_dict))),
        N=_get(configs, "layers", _get(pretrained_cfg, "layers", 6)),
        d_model=_get(configs, "d_model", _get(pretrained_cfg, "d_model", 1024)),
        d_ff=_get(configs, "d_ff", _get(pretrained_cfg, "d_ff", 512)),
        h=_get(configs, "head_num", _get(pretrained_cfg, "head_num", 8)),
        dropout=_get(configs, "dropout", _get(pretrained_cfg, "dropout", 0.1)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, word_dict, index_dict


def compute_stats(values, prefix=""):
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


def nearest_neighbor_baseline(spec_target, train_specs, train_structs, nk_dict):
    """Find closest training spectrum and compute TMM errors."""
    spec_target = np.array(spec_target)
    train_specs_arr = np.array(train_specs)
    maes = np.mean(np.abs(train_specs_arr - spec_target), axis=1)
    best_idx = np.argmin(maes)

    best_struct = train_structs[best_idx]
    # Normalize: remove BOS/EOS
    tokens = [t for t in best_struct if t not in ("BOS", "EOS", "PAD", "UNK")]
    mats, thick = parse_structure(tokens)

    try:
        result = spectrum(
            materials=mats, thickness=thick,
            pol=POLARIZATION, theta=THETA_DEG, wavelengths=WAVELENGTHS,
            nk_dict=nk_dict, substrate=SUBSTRATE, substrate_thick=SUBSTRATE_THICK)
        R_sim = np.array(result[:N_WL])
        T_sim = np.array(result[N_WL:])
        R_target = spec_target[:N_WL]
        T_target = spec_target[N_WL:]

        mae_R = float(np.mean(np.abs(R_sim - R_target)))
        mae_T = float(np.mean(np.abs(T_sim - T_target)))
        total_mae = float(np.mean(np.abs(
            np.concatenate([R_sim, T_sim]) - np.concatenate([R_target, T_target]))))

        return {
            "best_train_idx": int(best_idx),
            "tokens": tokens,
            "materials": mats,
            "thicknesses": thick,
            "n_layers": len(mats),
            "total_thickness": sum(thick),
            "mae_R": mae_R,
            "mae_T": mae_T,
            "total_mae": total_mae,
        }
    except Exception as e:
        return None


def run_validation(model, word_dict, index_dict, nk_dict,
                    test_specs, test_structs, train_specs, train_structs,
                    model_name, n_samples=100, max_len=22):
    """Run comprehensive validation."""
    print(f"\n{'='*60}")
    print(f"Validating: {model_name} ({n_samples} samples)")
    print(f"{'='*60}")

    decoding_configs = [
        ("greedy_N1", 1),
        ("sampling_N8", 8),
        ("sampling_N32", 32),
    ]

    all_results = {cfg: [] for cfg, _ in decoding_configs}
    nn_results = []
    total_time_start = time.time()

    for i in range(min(n_samples, len(test_specs))):
        spec_target = test_specs[i]
        gt_struct = test_structs[i] if test_structs is not None else None
        sample_start = time.time()

        if i % 20 == 0:
            print(f"  [{model_name}] Sample {i+1}/{min(n_samples, len(test_specs))}...")

        for cfg_name, n_cand in decoding_configs:
            candidates = generate_candidates(
                model, spec_target, word_dict, index_dict,
                num_candidates=n_cand, max_len=max_len, max_layers=max_len-2,
                top_k=10, top_p=0.9, temperature=1.0,
                nk_dict=nk_dict, device=DEVICE, decode_batch_size=8,
            )

            # TMM re-rank
            ranked = tmm_rerank(candidates, spec_target, nk_dict, WAVELENGTHS, N_WL)
            valid_ranked = [c for c in ranked if c.get("tmm_success")]

            # NN baseline (once per sample)
            if cfg_name == "greedy_N1" and train_specs is not None:
                nn = nearest_neighbor_baseline(
                    spec_target, train_specs, train_structs, nk_dict)
                if nn:
                    nn["sample_idx"] = i
                    nn_results.append(nn)

            best = valid_ranked[0] if valid_ranked else None
            entry = {
                "sample_idx": i,
                "n_generated": len(candidates),
                "n_valid_tmm": len(valid_ranked),
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
                    "max_R_err": best.get("max_R_err", 0),
                    "max_T_err": best.get("max_T_err", 0),
                    "avg_absorption": best.get("avg_absorption", 0),
                })
            all_results[cfg_name].append(entry)

        if i % 20 == 19:
            elapsed = time.time() - sample_start
            print(f"    Sample {i+1} done in {elapsed:.1f}s")

    total_time = time.time() - total_time_start
    print(f"  Total time: {total_time:.1f}s ({total_time/min(n_samples, len(test_specs)):.1f}s/sample)")

    # Compute statistics
    stats = {}
    for cfg_name, results in all_results.items():
        valid_results = [r for r in results if r.get("has_valid")]
        if not valid_results:
            stats[cfg_name] = {"n_total": len(results), "n_valid": 0}
            continue

        mae_t = [r["total_mae"] for r in valid_results]
        mae_r = [r["mae_R"] for r in valid_results]
        mae_T = [r["mae_T"] for r in valid_results]

        stats[cfg_name] = {
            "n_total": len(results),
            "n_valid": len(valid_results),
            "valid_rate": len(valid_results) / max(len(results), 1),
            **compute_stats(mae_t, "total_mae_"),
            **compute_stats(mae_r, "R_mae_"),
            **compute_stats(mae_T, "T_mae_"),
            "avg_n_layers": float(np.mean([r.get("best_n_layers", 0) for r in valid_results])),
            "avg_candidates": float(np.mean([r.get("n_valid_tmm", 0) for r in valid_results])),
        }

    if nn_results:
        valid_nn = [r for r in nn_results if r is not None and "total_mae" in r]
        if valid_nn:
            stats["nearest_neighbor"] = {
                "n_total": len(nn_results),
                "n_valid": len(valid_nn),
                **compute_stats([r["total_mae"] for r in valid_nn], "total_mae_"),
                **compute_stats([r["mae_R"] for r in valid_nn], "R_mae_"),
                **compute_stats([r["mae_T"] for r in valid_nn], "T_mae_"),
            }

    return all_results, stats, nn_results


def run_high_t_test_simple(model, word_dict, index_dict, nk_dict, output_dir):
    """Simplified high-T test."""
    print(f"\n{'='*60}")
    print("High-T Transmittance Test")
    print(f"{'='*60}")

    # Material transparency from k values
    print("\nMaterial transparency (based on k values):")
    for mat_name in nk_dict:
        if mat_name == SUBSTRATE:
            continue
        nk = nk_dict[mat_name]
        k_vals = np.abs(np.imag(nk))
        mean_k = np.mean(k_vals)
        max_k = np.max(k_vals)
        cat = "low_abs" if max_k < 0.05 else ("moderate" if max_k < 0.5 else "high_abs")
        print(f"  {mat_name:20s}: mean_k={mean_k:.4f}, max_k={max_k:.4f} [{cat}]")

    low_abs_mats = [m for m in nk_dict
                    if m != SUBSTRATE and np.max(np.abs(np.imag(nk_dict[m]))) < 0.05]
    high_abs_mats = [m for m in nk_dict
                     if m != SUBSTRATE and np.max(np.abs(np.imag(nk_dict[m]))) >= 0.5]
    print(f"\n  Low-absorption: {low_abs_mats}")
    print(f"  High-absorption: {high_abs_mats}")

    # Build high-T targets
    targets = [
        {
            "name": "full_band_high_T",
            "desc": "T≈0.98 across 400-1100nm",
            "spec": np.concatenate([np.zeros(N_WL), np.ones(N_WL) * 0.98]).tolist(),
        },
        {
            "name": "visible_high_T",
            "desc": "T≈0.98 in 400-800nm",
            "spec": np.concatenate([
                np.zeros(N_WL),
                np.concatenate([np.ones(41) * 0.98, np.ones(N_WL - 41) * 0.5])
            ]).tolist(),
        },
    ]

    results = []
    for target in targets:
        print(f"\n  Target: {target['name']} ({target['desc']})")
        spec_target = np.array(target["spec"])

        # No material constraints
        candidates = generate_candidates(
            model, spec_target, word_dict, index_dict,
            num_candidates=64, max_len=22, max_layers=20,
            top_k=10, top_p=0.9, temperature=1.0,
            nk_dict=nk_dict, device=DEVICE, decode_batch_size=8,
        )
        ranked = tmm_rerank(candidates, spec_target, nk_dict, WAVELENGTHS, N_WL)
        valid = [c for c in ranked if c.get("tmm_success")]

        if valid:
            best = valid[0]
            T_sim = np.array(best["T_sim"])
            metrics = {
                "avg_T": float(np.mean(T_sim)),
                "min_T": float(np.min(T_sim)),
                "p10_T": float(np.percentile(T_sim, 10)),
                "frac_T_ge_090": float(np.mean(T_sim >= 0.90)),
                "avg_R": float(np.mean(np.array(best["R_sim"]))),
                "avg_A": float(best.get("avg_absorption", 0)),
                "n_layers": len(best["materials"]),
                "total_thickness": sum(best["thicknesses"]),
                "structure": [f"{m}_{int(t)}" for m, t in zip(best["materials"], best["thicknesses"])],
            }
            print(f"    Best (unconstrained): avg_T={metrics['avg_T']:.4f}, "
                  f"min_T={metrics['min_T']:.4f}, frac>0.9={metrics['frac_T_ge_090']:.3f}, "
                  f"layers={metrics['n_layers']}")
            print(f"    Structure: {metrics['structure']}")
            results.append({"target": target["name"], "type": "unconstrained", **metrics})

        # Low-absorption only
        if low_abs_mats:
            candidates_lo = generate_candidates(
                model, spec_target, word_dict, index_dict,
                num_candidates=64, max_len=22, max_layers=20,
                top_k=10, top_p=0.9, temperature=1.0,
                nk_dict=nk_dict, device=DEVICE, decode_batch_size=8,
            )
            # Filter to low-absorption materials only
            candidates_lo = [c for c in candidates_lo
                             if all(m in low_abs_mats for m in c["materials"])]

            if candidates_lo:
                ranked_lo = tmm_rerank(candidates_lo, spec_target, nk_dict, WAVELENGTHS, N_WL)
                valid_lo = [c for c in ranked_lo if c.get("tmm_success")]
                if valid_lo:
                    best_lo = valid_lo[0]
                    T_sim_lo = np.array(best_lo["T_sim"])
                    metrics_lo = {
                        "avg_T": float(np.mean(T_sim_lo)),
                        "min_T": float(np.min(T_sim_lo)),
                        "p10_T": float(np.percentile(T_sim_lo, 10)),
                        "frac_T_ge_090": float(np.mean(T_sim_lo >= 0.90)),
                        "avg_R": float(np.mean(np.array(best_lo["R_sim"]))),
                        "avg_A": float(best_lo.get("avg_absorption", 0)),
                        "n_layers": len(best_lo["materials"]),
                        "total_thickness": sum(best_lo["thicknesses"]),
                        "structure": [f"{m}_{int(t)}" for m, t in zip(best_lo["materials"], best_lo["thicknesses"])],
                    }
                    print(f"    Best (low-abs only): avg_T={metrics_lo['avg_T']:.4f}, "
                          f"min_T={metrics_lo['min_T']:.4f}, frac>0.9={metrics_lo['frac_T_ge_090']:.3f}, "
                          f"layers={metrics_lo['n_layers']}")
                    print(f"    Structure: {metrics_lo['structure']}")
                    results.append({"target": target["name"], "type": "low_absorption", **metrics_lo})
                else:
                    print(f"    Low-absorption: NO valid candidate after TMM")
                    results.append({"target": target["name"], "type": "low_absorption", "error": "no_valid_candidate"})
            else:
                print(f"    Low-absorption: Model did not generate any low-abs-only structure")
                results.append({"target": target["name"], "type": "low_absorption", "error": "no_low_abs_generated"})

    return results


def main():
    print("=" * 60)
    print("OptoGPT Phase 2 - Comprehensive Validation")
    print(f"Device: {DEVICE}")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 60)

    # ---- Load nk data ----
    print("\n[1/5] Loading nk data...")
    nk_dict = load_materials(
        all_mats=["Ag", "Al", "Al2O3", "AlN", "Ge", "HfO2", "ITO", "MgF2", "MgO",
                   "Si", "Si3N4", "SiO2", "Ta2O5", "TiN", "TiO2", "ZnO", "ZnS", "ZnSe",
                   SUBSTRATE],
        wavelengths=WAVELENGTHS, DATABASE=NK_DIR)
    print(f"  Loaded {len(nk_dict)} materials")

    # ---- Data Audit ----
    print("\n[2/5] Data leakage audit...")
    audit_report = audit_data(DATA_DIR, OUTPUT_DIR / "audit", spec_threshold=1e-4)
    print(f"  Audit report: {OUTPUT_DIR / 'audit' / 'audit_report.json'}")

    # ---- Load test/train data ----
    print("\n[3/5] Loading test/train data...")
    with open(DATA_DIR / "Spectrum_test.pkl", "rb") as f:
        test_specs = pkl.load(f)
    with open(DATA_DIR / "Structure_test.pkl", "rb") as f:
        test_structs = pkl.load(f)
    with open(DATA_DIR / "Spectrum_train.pkl", "rb") as f:
        train_specs = pkl.load(f)
    with open(DATA_DIR / "Structure_train.pkl", "rb") as f:
        train_structs = pkl.load(f)
    print(f"  Test: {len(test_specs)} spectra, Train: {len(train_specs)} spectra")

    # ---- Load models & validate ----
    print("\n[4/5] Loading models and running validation...")

    # Fine-tuned model
    ft_model, ft_wd, ft_idict = load_model(FINETUNED_PATH, PRETRAINED_PATH)
    print(f"  Fine-tuned model loaded, vocab={len(ft_wd)}")

    # Select N for validation (depends on time budget)
    N_SAMPLES = min(100, len(test_specs))
    print(f"  Running {N_SAMPLES} samples...")

    ft_results, ft_stats, nn_results = run_validation(
        ft_model, ft_wd, ft_idict, nk_dict,
        test_specs, test_structs, train_specs, train_structs,
        "fine_tuned", n_samples=N_SAMPLES)

    # Save fine-tuned results
    ft_dir = OUTPUT_DIR / "fine_tuned"
    ft_dir.mkdir(parents=True, exist_ok=True)
    with open(ft_dir / "results.json", "w") as f:
        json.dump({"stats": ft_stats, "results": ft_results, "nn_results": nn_results},
                  f, indent=2, default=str)

    # ---- Summary CSV ----
    csv_path = OUTPUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "config", "n_total", "n_valid", "valid_rate",
                          "total_mae_mean", "total_mae_median", "total_mae_std",
                          "total_mae_p90", "total_mae_p95", "total_mae_max",
                          "R_mae_mean", "R_mae_std", "T_mae_mean", "T_mae_std"])
        for cfg_name, s in ft_stats.items():
            if isinstance(s, dict) and "total_mae_mean" in s:
                writer.writerow([
                    "fine_tuned", cfg_name,
                    s.get("n_total", 0), s.get("n_valid", 0), s.get("valid_rate", ""),
                    s.get("total_mae_mean", ""), s.get("total_mae_median", ""),
                    s.get("total_mae_std", ""), s.get("total_mae_p90", ""),
                    s.get("total_mae_p95", ""), s.get("total_mae_max", ""),
                    s.get("R_mae_mean", ""), s.get("R_mae_std", ""),
                    s.get("T_mae_mean", ""), s.get("T_mae_std", ""),
                ])
    print(f"\n  Summary CSV: {csv_path}")

    # ---- High-T test ----
    print("\n[5/5] High-transmittance test...")
    ht_results = run_high_t_test_simple(ft_model, ft_wd, ft_idict, nk_dict, OUTPUT_DIR / "high_T")
    with open(OUTPUT_DIR / "high_T" / "high_T_results.json", "w") as f:
        json.dump({"results": ht_results}, f, indent=2, default=str)

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print("PHASE 2 COMPLETE")
    print(f"{'='*60}")
    print(f"Total MAE definition: mean(|R_sim - R_target|, |T_sim - T_target|) over 142 points")
    print()

    for cfg_name, s in ft_stats.items():
        if isinstance(s, dict) and "total_mae_mean" in s:
            print(f"  {cfg_name}: Total MAE mean={s['total_mae_mean']:.4f}, "
                  f"std={s['total_mae_std']:.4f}, "
                  f"P95={s['total_mae_p95']:.4f}, "
                  f"valid={s.get('n_valid', 0)}/{s.get('n_total', 0)}")

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"End: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
