"""
TMM Physical Validation for OptoGPT 60° s-pol fine-tuned model.

This script:
1. Loads the fine-tuned model checkpoint
2. Takes target spectra from the test set (60°, s-pol)
3. Generates candidate multilayer structures via greedy decode
4. Re-simulates candidates with TMM at theta=60, pol="s"
5. Computes MAE (R, T, overall) and ranks candidates
6. Outputs best structures and comparison plots

Usage:
    python validate_60deg_s.py --model model/optogpt_60deg_s_best.pt --test_spec data_60deg_s/Spectrum_test.pkl --test_struct data_60deg_s/Structure_test.pkl --num_samples 20
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent  # optogpt/optogpt/
sys.path.insert(0, str(PROJECT_ROOT))

from core.datasets.datasets import PAD, UNK
from core.models.transformer import make_model_I, subsequent_mask
from core.datasets.sim import spectrum, load_materials, mats as default_mats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default conditions (overridable via --theta / --pol)
WAVELENGTHS = np.arange(0.4, 1.1 + 1e-3, 0.01)  # 400-1100nm, step 10nm
SUBSTRATE = "Glass_Substrate"
SUBSTRATE_THICK = 500000


def _get_config(cfg, key, default=None):
    """Get config value from either dict or Namespace."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_model(ckpt_path, pretrained_path=None):
    """Load fine-tuned model and vocabulary from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    configs = ckpt.get("configs", {})
    word_dict = _get_config(configs, "struc_word_dict")
    index_dict = _get_config(configs, "struc_index_dict")

    # Fall back to pretrained checkpoint for vocabulary & hyperparams
    pretrained_configs = {}
    if (word_dict is None) and pretrained_path is not None:
        pretrained = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
        pretrained_configs = pretrained.get("configs", {})
        word_dict = _get_config(pretrained_configs, "struc_word_dict", {})
        index_dict = _get_config(pretrained_configs, "struc_index_dict", {})

    if not word_dict:
        raise ValueError("Checkpoint missing struc_word_dict")

    # Determine model hyperparams (from finetuned config, fallback to pretrained)
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
    return model, word_dict, index_dict, configs


def greedy_decode(model, spec, word_dict, index_dict, max_len=22):
    """Greedy decode: given a spectrum, generate a structure sequence."""
    bos_id = word_dict.get("BOS", 2)
    eos_id = word_dict.get("EOS", 3)

    ys = torch.ones(1, 1, dtype=torch.long).fill_(bos_id).to(DEVICE)
    src = torch.tensor([spec], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    design = []
    with torch.no_grad():
        for _ in range(max_len - 1):
            trg_mask = subsequent_mask(ys.size(1))
            trg_mask = trg_mask.to(DEVICE)
            out = model(src, ys, None, trg_mask)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            if next_word == eos_id:
                break
            ys = torch.cat([ys, torch.tensor([[next_word]], device=DEVICE)], dim=1)
            sym = index_dict.get(next_word, "UNK")
            if sym not in ("UNK", "EOS", "BOS", "PAD"):
                design.append(sym)

    return design


def parse_structure(tokens):
    """Parse token list like ['SiO2_50', 'TiO2_100', ...] into materials and thicknesses."""
    materials = []
    thicknesses = []
    for tok in tokens:
        if "_" not in tok:
            continue  # skip invalid tokens
        parts = tok.rsplit("_", 1)
        if len(parts) != 2:
            continue
        mat, thick_str = parts
        try:
            thick = float(thick_str)
        except ValueError:
            continue
        materials.append(mat)
        thicknesses.append(thick)
    return materials, thicknesses


def is_valid_structure(materials, thicknesses, max_layers=20):
    """Check if the decoded structure is physically valid."""
    if len(materials) == 0 or len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    # Check all materials are known
    for m in materials:
        if m in ("EOS", "BOS", "UNK", "PAD", ""):
            return False
    # Check thicknesses are positive
    for t in thicknesses:
        if t <= 0 or t > 1000:  # reasonable thickness range in nm
            return False
    return True


def validate_sample(spec_target, model, word_dict, index_dict, nk_dict, num_candidates=5,
                    theta_deg=60, pol="s"):
    """Generate candidates for one target spectrum and compute TMM errors."""
    designs = []
    for _ in range(num_candidates):
        tokens = greedy_decode(model, spec_target, word_dict, index_dict)
        materials, thicknesses = parse_structure(tokens)
        if is_valid_structure(materials, thicknesses):
            try:
                sim_spec = spectrum(
                    materials, thicknesses,
                    pol=pol, theta=theta_deg,
                    wavelengths=WAVELENGTHS, nk_dict=nk_dict,
                    substrate=SUBSTRATE, substrate_thick=SUBSTRATE_THICK,
                )
            except Exception:
                continue

            if np.any(np.isnan(sim_spec)) or np.any(np.isinf(sim_spec)):
                continue

            n_wl = len(WAVELENGTHS)
            R_sim = np.array(sim_spec[:n_wl])
            T_sim = np.array(sim_spec[n_wl:])
            R_target = np.array(spec_target[:n_wl])
            T_target = np.array(spec_target[n_wl:])

            mae_R = np.mean(np.abs(R_sim - R_target))
            mae_T = np.mean(np.abs(T_sim - T_target))
            mae_total = (mae_R + mae_T) / 2.0

            designs.append({
                "materials": materials,
                "thicknesses": thicknesses,
                "tokens": tokens,
                "mae_R": mae_R,
                "mae_T": mae_T,
                "mae_total": mae_total,
                "R_sim": R_sim,
                "T_sim": T_sim,
            })

    if not designs:
        return None
    designs.sort(key=lambda d: d["mae_total"])
    return designs


def compute_spectral_mae_per_wavelength(all_results, wavelengths):
    """Compute per-wavelength MAE statistics across all validated samples."""
    n_wl = len(wavelengths)
    R_errors = np.zeros((len(all_results), n_wl))
    T_errors = np.zeros((len(all_results), n_wl))

    for i, r in enumerate(all_results):
        if "R_sim" in r and "R_target" in r:
            R_errors[i] = np.abs(np.array(r["R_sim"]) - np.array(r["R_target"]))
        if "T_sim" in r and "T_target" in r:
            T_errors[i] = np.abs(np.array(r["T_sim"]) - np.array(r["T_target"]))

    R_mean = np.mean(R_errors, axis=0)
    T_mean = np.mean(T_errors, axis=0)
    R_std = np.std(R_errors, axis=0)
    T_std = np.std(T_errors, axis=0)
    return R_mean, T_mean, R_std, T_std


def compute_material_stats(all_results, word_dict):
    """Compute material selection frequency from decoded structures."""
    from collections import Counter
    mat_counter = Counter()
    thickness_counter = Counter()

    for r in all_results:
        for tok in r.get("best_structure", []):
            if "_" in tok:
                parts = tok.rsplit("_", 1)
                if len(parts) == 2:
                    mat_counter[parts[0]] += 1
                    try:
                        thickness_counter[int(float(parts[1]))] += 1
                    except ValueError:
                        pass

    # Sort by frequency
    mat_ranking = mat_counter.most_common()
    thick_ranking = thickness_counter.most_common(10)
    return mat_ranking, thick_ranking


def plot_spectral_error(wavelengths, R_mean, R_std, T_mean, T_std, save_path,
                        theta_deg=60, pol="s"):
    """Plot per-wavelength MAE with std bands."""
    wl_nm = wavelengths * 1000
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.fill_between(wl_nm, R_mean - R_std, R_mean + R_std, alpha=0.3, color="red")
    ax1.plot(wl_nm, R_mean, "r-", linewidth=2)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("MAE (R)")
    ax1.set_title("Per-Wavelength Reflectance Error")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(wl_nm, T_mean - T_std, T_mean + T_std, alpha=0.3, color="blue")
    ax2.plot(wl_nm, T_mean, "b-", linewidth=2)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("MAE (T)")
    ax2.set_title("Per-Wavelength Transmittance Error")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Spectral Error Distribution (θ={theta_deg}°, {pol}-pol)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Spectral error plot saved: {save_path}")


def plot_comparison(spec_target, best_design, save_path, sample_idx,
                    theta_deg=60, pol="s"):
    """Plot target vs reconstructed R+T spectra."""
    n_wl = len(WAVELENGTHS)
    R_target = np.array(spec_target[:n_wl])
    T_target = np.array(spec_target[n_wl:])
    R_sim = best_design["R_sim"]
    T_sim = best_design["T_sim"]
    wl_nm = WAVELENGTHS * 1000  # convert to nm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(wl_nm, R_target, "b-", linewidth=2, label="Target R")
    ax1.plot(wl_nm, R_sim, "r--", linewidth=2, label=f"Reconstructed R (MAE={best_design['mae_R']:.4f})")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectance")
    ax1.set_title(f"Reflectance Comparison (θ={theta_deg}°, {pol}-pol)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(wl_nm, T_target, "b-", linewidth=2, label="Target T")
    ax2.plot(wl_nm, T_sim, "r--", linewidth=2, label=f"Reconstructed T (MAE={best_design['mae_T']:.4f})")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Transmittance")
    ax2.set_title(f"Transmittance Comparison (θ={theta_deg}°, {pol}-pol)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Print structure info
    struct_str = " | ".join(
        f"{m} {t:.0f}nm" for m, t in zip(best_design["materials"], best_design["thicknesses"])
    )
    fig.suptitle(f"Sample #{sample_idx}\nStructure: {struct_str}\n"
                 f"R MAE={best_design['mae_R']:.4f}, T MAE={best_design['mae_T']:.4f}, "
                 f"Total MAE={best_design['mae_total']:.4f}",
                 fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="TMM validation for 60° s-pol model")
    parser.add_argument("--model", type=str, default="../model/optogpt_60deg_s_best.pt",
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--test_spec", type=str, default="../data_60deg_s/Spectrum_test.pkl",
                        help="Path to test spectrum data (.pkl)")
    parser.add_argument("--test_struct", type=str, default="../data_60deg_s/Structure_test.pkl",
                        help="Path to test structure data (.pkl) for reference")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of test samples to validate")
    parser.add_argument("--num_candidates", type=int, default=5,
                        help="Number of candidates per sample")
    parser.add_argument("--output_dir", type=str, default="../validation_results",
                        help="Directory to save results and plots")
    parser.add_argument("--pretrained", type=str, default="../model/optogpt.pt",
                        help="Path to original pretrained checkpoint (for vocab fallback)")
    parser.add_argument("--theta", type=float, default=60,
                        help="Incidence angle in degrees")
    parser.add_argument("--pol", type=str, default="s", choices=["s", "p"],
                        help="Polarization: s or p")
    args = parser.parse_args()

    # Resolve paths relative to script location
    model_path = (PROJECT_ROOT / args.model).resolve()
    test_spec_path = (PROJECT_ROOT / args.test_spec).resolve()
    test_struct_path = (PROJECT_ROOT / args.test_struct).resolve() if args.test_struct else None
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Model: {model_path}")
    print(f"Test data: {test_spec_path}")

    # ---- Load model ----
    print("\nLoading model...")
    pretrained_path = (PROJECT_ROOT / args.pretrained).resolve() if args.pretrained else None
    model, word_dict, index_dict, configs = load_model(str(model_path), str(pretrained_path) if pretrained_path else None)
    print(f"  Vocab size: {len(word_dict)}, PAD={word_dict.get('PAD')}, "
          f"BOS={word_dict.get('BOS')}, EOS={word_dict.get('EOS')}")

    # ---- Load test data ----
    import pickle as pkl
    with open(test_spec_path, "rb") as f:
        test_specs = pkl.load(f)
    if not isinstance(test_specs, np.ndarray):
        test_specs = np.array(test_specs)

    test_structs = None
    if test_struct_path and test_struct_path.exists():
        with open(test_struct_path, "rb") as f:
            test_structs = pkl.load(f)

    n_samples = min(args.num_samples, len(test_specs))
    print(f"  Test spectra: {len(test_specs)}, validating {n_samples}")

    # ---- Load nk data ----
    print("\nLoading material nk data...")
    nk_dir = str(PROJECT_ROOT / "nk")
    nk_dict = load_materials(
        all_mats=default_mats,
        wavelengths=WAVELENGTHS,
        DATABASE=nk_dir,
    )
    # Also load substrate
    substrate_nk = load_materials(all_mats=[SUBSTRATE], wavelengths=WAVELENGTHS,
                                   DATABASE=nk_dir)
    nk_dict.update(substrate_nk)
    print(f"  Loaded {len(nk_dict)} materials")

    # ---- Validate ----
    print(f"\n{'='*60}")
    print(f"Validating {n_samples} samples (θ={args.theta}°, {args.pol}-pol)...")
    print(f"{'='*60}")

    all_results = []
    n_wl = len(WAVELENGTHS)
    for i in range(n_samples):
        spec = test_specs[i]
        gt_struct = test_structs[i] if test_structs else None

        designs = validate_sample(spec, model, word_dict, index_dict, nk_dict,
                                  num_candidates=args.num_candidates,
                                  theta_deg=args.theta, pol=args.pol)
        if designs is None or len(designs) == 0:
            print(f"  Sample {i}: No valid design generated")
            continue

        best = designs[0]
        print(f"  Sample {i}: R_MAE={best['mae_R']:.4f}, T_MAE={best['mae_T']:.4f}, "
              f"Total={best['mae_total']:.4f}, "
              f"Layers={len(best['materials'])}")

        # Plot best candidate
        plot_path = output_dir / f"sample_{i:03d}.png"
        plot_comparison(spec, best, str(plot_path), i,
                        theta_deg=args.theta, pol=args.pol)

        all_results.append({
            "sample_idx": i,
            "gt_structure": gt_struct,
            "best_structure": best["tokens"],
            "materials": best["materials"],
            "thicknesses": best["thicknesses"],
            "mae_R": float(best["mae_R"]),
            "mae_T": float(best["mae_T"]),
            "mae_total": float(best["mae_total"]),
            "n_candidates": len(designs),
            "R_sim": best["R_sim"].tolist(),
            "T_sim": best["T_sim"].tolist(),
            "R_target": np.array(spec[:n_wl]).tolist(),
            "T_target": np.array(spec[n_wl:]).tolist(),
        })

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    if all_results:
        mae_Rs = [r["mae_R"] for r in all_results]
        mae_Ts = [r["mae_T"] for r in all_results]
        mae_totals = [r["mae_total"] for r in all_results]

        print(f"  Valid samples: {len(all_results)}/{n_samples}")
        print(f"  R  MAE: mean={np.mean(mae_Rs):.4f}, std={np.std(mae_Rs):.4f}, "
              f"min={np.min(mae_Rs):.4f}, max={np.max(mae_Rs):.4f}")
        print(f"  T  MAE: mean={np.mean(mae_Ts):.4f}, std={np.std(mae_Ts):.4f}, "
              f"min={np.min(mae_Ts):.4f}, max={np.max(mae_Ts):.4f}")
        print(f"  Total MAE: mean={np.mean(mae_totals):.4f}, std={np.std(mae_totals):.4f}, "
              f"min={np.min(mae_totals):.4f}, max={np.max(mae_totals):.4f}")

        # Find best overall
        best_idx = np.argmin(mae_totals)
        best_result = all_results[best_idx]
        struct_str = " | ".join(
            f"{m} {t:.0f}nm" for m, t in zip(best_result["materials"], best_result["thicknesses"])
        )
        print(f"\n  BEST STRUCTURE (sample {best_result['sample_idx']}):")
        print(f"    {struct_str}")
        print(f"    R MAE={best_result['mae_R']:.6f}, T MAE={best_result['mae_T']:.6f}, "
              f"Total={best_result['mae_total']:.6f}")

    # Save results JSON
    results_path = output_dir / "validation_results.json"

    # ---- Spectral error analysis & material stats ----    
    if all_results:
        # Per-wavelength error
        R_err_mean, T_err_mean, R_err_std, T_err_std = compute_spectral_mae_per_wavelength(
            all_results, WAVELENGTHS)
        spectral_plot_path = output_dir / "spectral_error_distribution.png"
        plot_spectral_error(WAVELENGTHS, R_err_mean, R_err_std, T_err_mean, T_err_std,
                           str(spectral_plot_path),
                           theta_deg=args.theta, pol=args.pol)

        # Material usage statistics
        mat_ranking, thick_ranking = compute_material_stats(all_results, word_dict)
        print(f"\n  Material usage frequency (top 10):")
        for mat, count in mat_ranking[:10]:
            pct = 100.0 * count / sum(c for _, c in mat_ranking)
            print(f"    {mat}: {count} ({pct:.1f}%)")

        print(f"\n  Thickness frequency (top 10):")
        for thick, count in thick_ranking[:10]:
            pct = 100.0 * count / sum(c for _, c in thick_ranking)
            print(f"    {thick}nm: {count} ({pct:.1f}%)")

        # Identify worst wavelength regions
        worst_R_idx = np.argmax(R_err_mean)
        worst_T_idx = np.argmax(T_err_mean)
        print(f"\n  Worst R wavelength: {WAVELENGTHS[worst_R_idx]*1000:.0f}nm "
              f"(MAE={R_err_mean[worst_R_idx]:.4f}±{R_err_std[worst_R_idx]:.4f})")
        print(f"  Worst T wavelength: {WAVELENGTHS[worst_T_idx]*1000:.0f}nm "
              f"(MAE={T_err_mean[worst_T_idx]:.4f}±{T_err_std[worst_T_idx]:.4f})")

        # Best & worst 5 samples
        sorted_results = sorted(all_results, key=lambda r: r["mae_total"])
        print(f"\n  TOP 5 (best) samples:")
        for r in sorted_results[:5]:
            struct_str = " | ".join(
                f"{m} {t:.0f}nm" for m, t in zip(r["materials"], r["thicknesses"]))
            print(f"    #{r['sample_idx']}: Total={r['mae_total']:.6f}, "
                  f"R={r['mae_R']:.6f}, T={r['mae_T']:.6f}, "
                  f"L={len(r['materials'])}, [{struct_str}]")

        print(f"\n  BOTTOM 5 (worst) samples:")
        for r in sorted_results[-5:]:
            struct_str = " | ".join(
                f"{m} {t:.0f}nm" for m, t in zip(r["materials"], r["thicknesses"]))
            print(f"    #{r['sample_idx']}: Total={r['mae_total']:.6f}, "
                  f"R={r['mae_R']:.6f}, T={r['mae_T']:.6f}, "
                  f"L={len(r['materials'])}, [{struct_str}]")
    with open(results_path, "w") as f:
        summary_dict = {
            "theta_deg": args.theta,
            "polarization": args.pol,
            "model": str(model_path),
            "n_samples": n_samples,
            "n_valid": len(all_results),
            "summary": {
                "R_mae_mean": float(np.mean(mae_Rs)) if all_results else None,
                "R_mae_std": float(np.std(mae_Rs)) if all_results else None,
                "T_mae_mean": float(np.mean(mae_Ts)) if all_results else None,
                "T_mae_std": float(np.std(mae_Ts)) if all_results else None,
                "total_mae_mean": float(np.mean(mae_totals)) if all_results else None,
                "total_mae_std": float(np.std(mae_totals)) if all_results else None,
            },
        }
        if all_results:
            summary_dict["material_stats"] = {
                "frequency": [{"material": m, "count": c} for m, c in mat_ranking],
                "thickness_top10": [{"thickness_nm": t, "count": c} for t, c in thick_ranking],
            }
            summary_dict["spectral_error"] = {
                "wavelengths_nm": (WAVELENGTHS * 1000).tolist(),
                "R_mae_mean": R_err_mean.tolist(),
                "R_mae_std": R_err_std.tolist(),
                "T_mae_mean": T_err_mean.tolist(),
                "T_mae_std": T_err_std.tolist(),
            }
        summary_dict["results"] = all_results
        json.dump(summary_dict, f, indent=2, default=str)

    print(f"\n  Results saved: {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
