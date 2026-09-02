"""
High-Transmittance Specialized Test for OptoGPT 60deg s-pol.

Tests the model's ability to design anti-reflection / high-transmission coatings.

Target categories:
  1. Full-band high-T (400-1100nm)
  2. Visible-band high-T (400-800nm)
  3. Flat high-T with minimum T constraint

Generates a material transparency report from nk data (k values),
supports allowed/blocked material configurations.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent  # optogpt/optogpt/
sys.path.insert(0, str(PROJECT_ROOT))

from core.models.transformer import make_model_I
from core.datasets.sim import spectrum, load_materials, mats as default_mats
from multi_candidate_decoder import generate_candidates, tmm_rerank

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THETA_DEG = 60
POLARIZATION = "s"
WAVELENGTHS = np.arange(0.4, 1.1 + 1e-3, 0.01)
SUBSTRATE = "Glass_Substrate"
SUBSTRATE_THICK = 500000
N_WL = len(WAVELENGTHS)


def load_model_and_vocab(ckpt_path, pretrained_path=None):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    configs = ckpt.get("configs", {})

    def _get(cfg, key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    word_dict = _get(configs, "struc_word_dict")
    index_dict = _get(configs, "struc_index_dict")

    pretrained_cfg = {}
    if (word_dict is None) and pretrained_path:
        p = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
        pretrained_cfg = p.get("configs", {})
        word_dict = _get(pretrained_cfg, "struc_word_dict", {})
        index_dict = _get(pretrained_cfg, "struc_index_dict", {})

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


def material_transparency_report(nk_dict, wavelengths, k_threshold=0.01):
    """
    Generate a transparency report based on k (extinction coefficient) values.
    Low k = transparent (low absorption).
    """
    report = {}
    wl_nm = wavelengths * 1000
    for mat, nk in nk_dict.items():
        k_vals = np.abs(np.imag(nk))
        mean_k = np.mean(k_vals)
        max_k = np.max(k_vals)
        # Proportion of wavelengths with k < threshold
        transparent_frac = np.mean(k_vals < k_threshold)

        # Classify
        if max_k < k_threshold:
            category = "highly_transparent"
        elif mean_k < k_threshold * 2:
            category = "mostly_transparent"
        elif mean_k < k_threshold * 10:
            category = "moderate_absorption"
        else:
            category = "strong_absorption"

        report[mat] = {
            "mean_k": float(mean_k),
            "max_k": float(max_k),
            "transparent_fraction": float(transparent_frac),
            "category": category,
        }

    return report


def build_high_t_targets():
    """Build three categories of high-transmittance targets."""
    targets = []

    # Category 1: Full-band high-T (400-1100nm)
    # R ≈ 0, T ≈ 1 across all wavelengths
    R = np.zeros(N_WL)
    T = np.ones(N_WL) * 0.98  # slightly below 1 to be physically reasonable
    targets.append({
        "name": "full_band_high_T",
        "description": "T≈0.98 across 400-1100nm, R≈0",
        "R_target": R,
        "T_target": T,
        "spec_target": np.concatenate([R, T]).tolist(),
        "min_T_required": 0.90,
    })

    # Category 2: Visible-band high-T (400-800nm, indices 0-40)
    R2 = np.zeros(N_WL)
    T2 = np.zeros(N_WL)
    T2[:41] = 0.98  # visible band high T
    T2[41:] = 0.50  # NIR less constrained
    targets.append({
        "name": "visible_high_T",
        "description": "T≈0.98 in 400-800nm, relaxed NIR",
        "R_target": R2,
        "T_target": T2,
        "spec_target": np.concatenate([R2, T2]).tolist(),
        "min_T_required": 0.85,
    })

    # Category 3: Flat high-T with minimum constraint
    R3 = np.zeros(N_WL)
    T3 = np.ones(N_WL) * 0.95
    targets.append({
        "name": "flat_high_T_095",
        "description": "Flat T≈0.95, no dips below 0.90",
        "R_target": R3,
        "T_target": T3,
        "spec_target": np.concatenate([R3, T3]).tolist(),
        "min_T_required": 0.90,
    })

    return targets


def compute_high_t_metrics(R_sim, T_sim, wavelengths, min_T_required=0.90):
    """Compute specialized high-T metrics."""
    T_arr = np.asarray(T_sim)
    R_arr = np.asarray(R_sim)
    A_arr = 1 - R_arr - T_arr

    return {
        "avg_T": float(np.mean(T_arr)),
        "min_T": float(np.min(T_arr)),
        "p10_T": float(np.percentile(T_arr, 10)),
        "frac_T_above_090": float(np.mean(T_arr >= 0.90)),
        "frac_T_above_095": float(np.mean(T_arr >= 0.95)),
        "frac_T_above_min": float(np.mean(T_arr >= min_T_required)),
        "avg_R": float(np.mean(R_arr)),
        "max_R": float(np.max(R_arr)),
        "avg_A": float(np.mean(A_arr)),
        "max_A": float(np.max(A_arr)),
    }


def tmm_simulate_safe(materials, thicknesses, nk_dict):
    """Safe TMM simulation wrapper."""
    try:
        result = spectrum(
            materials=materials, thickness=thicknesses,
            pol=POLARIZATION, theta=THETA_DEG,
            wavelengths=WAVELENGTHS, nk_dict=nk_dict,
            substrate=SUBSTRATE, substrate_thick=SUBSTRATE_THICK,
        )
        R = np.array(result[:N_WL], dtype=np.float64)
        T = np.array(result[N_WL:], dtype=np.float64)
        if np.any(np.isnan(R)) or np.any(np.isnan(T)):
            return None, None, False
        if np.any(np.isinf(R)) or np.any(np.isinf(T)):
            return None, None, False
        return R, T, True
    except Exception:
        return None, None, False


def run_high_t_test(model, word_dict, index_dict, nk_dict, targets,
                     transparency_report, output_dir,
                     num_candidates=64, allowed_materials=None,
                     blocked_materials=None):
    """Run high-transmittance test for all target categories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Build material constraint filter
    def material_allowed(m):
        if allowed_materials is not None and m not in allowed_materials:
            return False
        if blocked_materials is not None and m in blocked_materials:
            return False
        return True

    for target in targets:
        print(f"\n--- Target: {target['name']} ---")
        print(f"  {target['description']}")

        spec_target = np.array(target["spec_target"])
        candidates = generate_candidates(
            model, spec_target, word_dict, index_dict,
            num_candidates=num_candidates, max_len=22, max_layers=20,
            top_k=10, top_p=0.9, temperature=1.0,
            nk_dict=nk_dict, device=DEVICE, decode_batch_size=8,
        )

        # Filter by material constraints
        if allowed_materials or blocked_materials:
            candidates = [
                c for c in candidates
                if all(material_allowed(m) for m in c["materials"])
            ]

        # TMM simulate
        simulated = []
        for cand in candidates:
            R_sim, T_sim, ok = tmm_simulate_safe(
                cand["materials"], cand["thicknesses"], nk_dict)
            if not ok:
                continue
            metrics = compute_high_t_metrics(
                R_sim, T_sim, WAVELENGTHS, target["min_T_required"])
            metrics.update({
                "materials": cand["materials"],
                "thicknesses": cand["thicknesses"],
                "n_layers": len(cand["materials"]),
                "total_thickness": sum(cand["thicknesses"]),
            })
            simulated.append(metrics)

        # Sort by avg_T descending (best transmission first)
        simulated.sort(key=lambda x: x["avg_T"], reverse=True)

        if simulated:
            best = simulated[0]
            print(f"  Best: avg_T={best['avg_T']:.4f}, min_T={best['min_T']:.4f}, "
                  f"frac>0.90={best['frac_T_above_090']:.3f}, "
                  f"layers={best['n_layers']}, thickness={best['total_thickness']:.0f}nm")
            print(f"    Structure: {best['materials']}")
            print(f"    Thicknesses: {best['thicknesses']}")
        else:
            print("  NO valid candidates generated!")

        results.append({
            "target": target["name"],
            "description": target["description"],
            "n_candidates_tested": len(simulated),
            "top5": simulated[:5] if simulated else [],
            "best": simulated[0] if simulated else None,
        })

        # Plot
        if simulated:
            plot_high_t_result(
                target, simulated[:3], WAVELENGTHS,
                str(output_dir / f"{target['name']}.png"))

    return results


def plot_high_t_result(target, top_candidates, wavelengths, save_path):
    """Plot the best high-T candidate spectra."""
    wl_nm = wavelengths * 1000
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Target
    ax1.plot(wl_nm, target["R_target"] * 100, "b-", linewidth=2, label="R target")
    ax2.plot(wl_nm, target["T_target"] * 100, "r-", linewidth=2, label="T target")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, cand in enumerate(top_candidates):
        R_sim, T_sim, ok = tmm_simulate_safe(
            cand["materials"], cand["thicknesses"],
            load_materials(all_mats=default_mats, wavelengths=WAVELENGTHS,
                           DATABASE=str(PROJECT_ROOT / "nk")))
        if not ok:
            continue
        label_suffix = f" (avgT={cand['avg_T']:.3f})"
        ax1.plot(wl_nm, R_sim * 100, "--", color=colors[i % 3],
                 alpha=0.7, label=f"Candidate {i+1}{label_suffix}")
        ax2.plot(wl_nm, T_sim * 100, "--", color=colors[i % 3],
                 alpha=0.7, label=f"Candidate {i+1}{label_suffix}")

    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectance (%)")
    ax1.set_title(f"R: {target['name']}")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)

    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Transmittance (%)")
    ax2.set_title(f"T: {target['name']}")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)

    fig.suptitle(f"High-T Test: {target['description']} (θ={THETA_DEG}°, {POLARIZATION}-pol)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="High-T transmittance test")
    parser.add_argument("--model", type=str, default="../model/optogpt_60deg_s_best.pt")
    parser.add_argument("--pretrained", type=str, default="../model/optogpt.pt")
    parser.add_argument("--output_dir", type=str, default="../validation_results/phase2/high_T")
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed_materials", type=str, default=None,
                        help="Comma-separated list of allowed materials")
    parser.add_argument("--blocked_materials", type=str, default=None,
                        help="Comma-separated list of blocked materials")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_path = (PROJECT_ROOT / args.model).resolve()
    pretrained_path = (PROJECT_ROOT / args.pretrained).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()

    # Load model
    print(f"Loading model: {model_path}")
    model, word_dict, index_dict = load_model_and_vocab(
        str(model_path), str(pretrained_path))

    # Load nk data
    nk_dir = str(PROJECT_ROOT / "nk")
    nk_dict = load_materials(all_mats=default_mats, wavelengths=WAVELENGTHS,
                              DATABASE=nk_dir)
    substrate_nk = load_materials(all_mats=[SUBSTRATE], wavelengths=WAVELENGTHS,
                                   DATABASE=nk_dir)
    nk_dict.update(substrate_nk)

    # Material transparency report
    print("\n=== Material Transparency Report (based on k values) ===")
    report = material_transparency_report(nk_dict, WAVELENGTHS)
    for mat, info in sorted(report.items(), key=lambda x: x[1]["mean_k"]):
        print(f"  {mat:20s}: mean_k={info['mean_k']:.4f}, max_k={info['max_k']:.4f}, "
              f"transparent_frac={info['transparent_fraction']:.2f}, {info['category']}")

    # Identify low-absorption materials (mean_k < 0.01)
    low_abs_mats = [m for m, info in report.items()
                    if info["category"] in ("highly_transparent", "mostly_transparent")
                    and m != SUBSTRATE]
    high_abs_mats = [m for m, info in report.items()
                     if info["category"] == "strong_absorption"]
    print(f"\n  Low-absorption materials: {low_abs_mats}")
    print(f"  Strong-absorption materials: {high_abs_mats}")

    # Build targets
    targets = build_high_t_targets()
    print(f"\nBuilt {len(targets)} high-T target categories")

    # Parse material constraints
    allowed = None
    blocked = None
    if args.allowed_materials:
        allowed = [m.strip() for m in args.allowed_materials.split(",")]
    if args.blocked_materials:
        blocked = [m.strip() for m in args.blocked_materials.split(",")]

    # Test 1: No material constraints
    print("\n" + "="*60)
    print("TEST 1: No material constraints")
    print("="*60)
    results_unconstrained = run_high_t_test(
        model, word_dict, index_dict, nk_dict, targets,
        report, output_dir / "unconstrained",
        num_candidates=args.num_candidates)

    # Test 2: Only low-absorption materials
    print("\n" + "="*60)
    print("TEST 2: Low-absorption materials only")
    print("="*60)
    results_constrained = run_high_t_test(
        model, word_dict, index_dict, nk_dict, targets,
        report, output_dir / "low_absorption_only",
        num_candidates=args.num_candidates,
        allowed_materials=low_abs_mats if low_abs_mats else None,
        blocked_materials=high_abs_mats)

    # Save comprehensive report
    full_report = {
        "timestamp": datetime.now().isoformat(),
        "model": str(model_path),
        "theta_deg": THETA_DEG,
        "polarization": POLARIZATION,
        "material_transparency": report,
        "low_absorption_materials": low_abs_mats,
        "strong_absorption_materials": high_abs_mats,
        "unconstrained": results_unconstrained,
        "low_absorption_only": results_constrained,
    }
    with open(output_dir / "high_T_report.json", "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*60)
    print("HIGH-T TEST SUMMARY")
    print("="*60)
    for test_name, test_results in [("Unconstrained", results_unconstrained),
                                      ("Low-absorption only", results_constrained)]:
        print(f"\n--- {test_name} ---")
        for r in test_results:
            if r["best"]:
                b = r["best"]
                print(f"  {r['target']}: avg_T={b['avg_T']:.4f}, "
                      f"min_T={b['min_T']:.4f}, "
                      f"frac>0.90={b['frac_T_above_090']:.3f}, "
                      f"layers={b['n_layers']}, "
                      f"thickness={b['total_thickness']:.0f}nm")
            else:
                print(f"  {r['target']}: NO VALID RESULT")

    print(f"\nFull report: {output_dir / 'high_T_report.json'}")


if __name__ == "__main__":
    main()
