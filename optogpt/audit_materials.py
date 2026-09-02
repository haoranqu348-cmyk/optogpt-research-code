"""
Material nk/Absorption Audit for High-Transmission Dielectric Design.

Audits all 19 materials + substrate in 400-1100nm range:
  - max(k), mean(k), k at key wavelengths
  - Single-layer TMM absorption at 60° s-pol
  - Classification: dielectric / slightly / moderately / strongly absorbing

Output: audit_report.json + console table
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Path setup — robust: find the optogpt package root (contains core/, nk/)
SCRIPT_DIR = Path(__file__).resolve().parent

def _find_pkg_root(start: Path) -> Path:
    """Walk up until we find a directory containing 'core' subdirectory."""
    p = start.resolve()
    for _ in range(6):
        if (p / "core").is_dir():
            return p
        if (p / "optogpt" / "core").is_dir():
            return p / "optogpt"
        p = p.parent
    raise RuntimeError(f"Cannot find optogpt package root (no core/ found) from {start}")

PKG_ROOT = _find_pkg_root(SCRIPT_DIR)
sys.path.insert(0, str(PKG_ROOT))

from core.datasets.sim import load_materials, spectrum, wavelengths

# Fixed simulation conditions
THETA_DEG = 60
POLARIZATION = "s"
SUBSTRATE = "Glass_Substrate"
SUBSTRATE_THICK = 500000
NK_DATABASE = str(PKG_ROOT / "nk")

# All materials in the checkpoint (from sim.py mats list)
ALL_MATS = [
    "Ag", "Al", "Al2O3", "AlN", "Ge", "HfO2", "ITO",
    "MgF2", "MgO", "Si", "Si3N4", "SiO2", "Ta2O5", "TiN",
    "TiO2", "ZnO", "ZnS", "ZnSe", "Glass_Substrate"
]

# Final decision for strict-dielectric high-T design (Step 1 audit result)
BANNED_MATERIALS = {"Ag", "Al", "TiN", "Ge", "Si", "ITO", "ZnS", "ZnSe"}
ALLOWED_MATERIALS = [
    "Al2O3", "AlN", "HfO2", "MgF2", "MgO",
    "Si3N4", "SiO2", "Ta2O5", "TiO2", "ZnO",
]
# Note: TiO2, Si3N4, ZnO have marginal edge absorption at ~400nm only.
# They are kept for design diversity; monitor absorption in validation.

# Test thickness for single-layer absorption audit
TEST_THICKNESS = 50  # nm


def classify_material(max_k, mean_k, mean_A_single):
    """Classify material based on k values and TMM absorption."""
    if max_k < 0.001 and mean_A_single < 0.01:
        return "dielectric"
    elif max_k < 0.01 and mean_A_single < 0.05:
        return "slightly_absorbing"
    elif max_k < 0.1:
        return "moderately_absorbing"
    else:
        return "strongly_absorbing"


def audit_material(mat, nk_dict, wavelengths_um, wavelengths_nm):
    """Run full audit for one material."""
    nk = nk_dict[mat]  # complex refractive index array
    k_vals = np.abs(nk.imag)  # extinction coefficient

    # Find indices for 400-1100nm range in wavelengths array
    # wavelengths is in microns: 0.4 to 1.1 step 0.01 → indices 0 to 70
    n_pts = len(wavelengths_um)

    max_k = float(np.max(k_vals))
    mean_k = float(np.mean(k_vals))
    min_k = float(np.min(k_vals))

    # k at key wavelengths (find closest indices)
    key_wls_nm = [400, 550, 700, 900, 1100]
    k_at_key = {}
    for wl_nm in key_wls_nm:
        idx = np.argmin(np.abs(wavelengths_nm - wl_nm))
        k_at_key[f"{wl_nm}nm"] = float(k_vals[idx])

    # Single-layer TMM simulation at 60° s-pol
    try:
        result = spectrum(
            materials=[mat],
            thickness=[TEST_THICKNESS],
            pol=POLARIZATION,
            theta=THETA_DEG,
            wavelengths=wavelengths_um,
            nk_dict=nk_dict,
            substrate=SUBSTRATE,
            substrate_thick=SUBSTRATE_THICK,
        )
        half = len(result) // 2
        R = np.array(result[:half])
        T = np.array(result[half:])
        A = 1.0 - R - T  # absorption at each wavelength

        mean_R = float(np.mean(R))
        mean_T = float(np.mean(T))
        mean_A = float(np.mean(A))
        max_A = float(np.max(A))

        # T at key wavelengths
        T_at_key = {}
        for wl_nm in key_wls_nm:
            idx = np.argmin(np.abs(wavelengths_nm - wl_nm))
            T_at_key[f"{wl_nm}nm"] = float(T[idx])

        # A at key wavelengths
        A_at_key = {}
        for wl_nm in key_wls_nm:
            idx = np.argmin(np.abs(wavelengths_nm - wl_nm))
            A_at_key[f"{wl_nm}nm"] = float(A[idx])

    except Exception as e:
        mean_R, mean_T, mean_A, max_A = None, None, None, None
        T_at_key, A_at_key = {}, {}
        print(f"  WARNING: TMM failed for {mat}: {e}")

    # Classification
    classification = classify_material(max_k, mean_k, mean_A if mean_A is not None else 1.0)

    # Recommendation for high-T design
    is_banned = mat in BANNED_MATERIALS
    if classification in ("strongly_absorbing", "moderately_absorbing"):
        recommendation = "BAN — high absorption"
    elif classification == "slightly_absorbing":
        if mean_A is not None and mean_A > 0.03:
            recommendation = "WARN — marginal absorption, consider banning for strict high-T"
        else:
            recommendation = "ALLOW — but monitor edge absorption"
    else:
        recommendation = "ALLOW — clean dielectric"

    if is_banned:
        recommendation = "BAN — user-specified banned list"

    return {
        "material": mat,
        "max_k": max_k,
        "mean_k": mean_k,
        "min_k": min_k,
        "k_at_key_wavelengths": k_at_key,
        "single_layer_50nm_60deg_s": {
            "mean_R": mean_R,
            "mean_T": mean_T,
            "mean_A": mean_A,
            "max_A": max_A,
            "T_at_key_wavelengths": T_at_key,
            "A_at_key_wavelengths": A_at_key,
        },
        "classification": classification,
        "user_banned": is_banned,
        "recommendation": recommendation,
    }


def main():
    print("=" * 80)
    print("Material nk/Absorption Audit for High-Transmission Dielectric Design")
    print(f"Theta={THETA_DEG}°, pol={POLARIZATION}, substrate={SUBSTRATE}")
    print(f"Wavelength range: 400-1100nm, single-layer thickness: {TEST_THICKNESS}nm")
    print("=" * 80)

    # Load all nk data
    print(f"\nLoading nk data from: {NK_DATABASE}")
    nk_dict = load_materials(all_mats=ALL_MATS, wavelengths=wavelengths, DATABASE=NK_DATABASE)
    print(f"Loaded {len(nk_dict)} materials")

    wavelengths_nm = wavelengths * 1000  # convert microns to nm

    # Audit each material
    results = []
    for mat in ALL_MATS:
        print(f"\n--- {mat} ---")
        result = audit_material(mat, nk_dict, wavelengths, wavelengths_nm)
        results.append(result)

        # Print summary
        r = result
        sl = r["single_layer_50nm_60deg_s"]
        print(f"  k: max={r['max_k']:.2e}, mean={r['mean_k']:.2e}")
        if sl["mean_A"] is not None:
            print(f"  Single-layer 50nm: R={sl['mean_R']:.4f}, T={sl['mean_T']:.4f}, A={sl['mean_A']:.4f}, max_A={sl['max_A']:.4f}")
        print(f"  Classification: {r['classification']}")
        print(f"  Recommendation: {r['recommendation']}")

    # ---- Summary Tables ----
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY TABLE")
    print("=" * 80)

    # Table 1: k-value summary
    print(f"\n{'Material':<20} {'max(k)':>10} {'mean(k)':>10} {'k@400nm':>10} {'k@550nm':>10} {'k@700nm':>10} {'k@900nm':>10} {'k@1100nm':>10} {'Class':<22}")
    print("-" * 120)
    for r in results:
        kk = r["k_at_key_wavelengths"]
        print(f"{r['material']:<20} {r['max_k']:>10.2e} {r['mean_k']:>10.2e} "
              f"{kk['400nm']:>10.2e} {kk['550nm']:>10.2e} {kk['700nm']:>10.2e} "
              f"{kk['900nm']:>10.2e} {kk['1100nm']:>10.2e} {r['classification']:<22}")

    # Table 2: Single-layer TMM absorption
    print(f"\n{'Material':<20} {'mean(R)':>10} {'mean(T)':>10} {'mean(A)':>10} {'max(A)':>10} {'A@400nm':>10} {'A@550nm':>10} {'A@700nm':>10} {'Recommendation':<30}")
    print("-" * 130)
    for r in results:
        sl = r["single_layer_50nm_60deg_s"]
        if sl["mean_A"] is not None:
            Ak = sl["A_at_key_wavelengths"]
            print(f"{r['material']:<20} {sl['mean_R']:>10.4f} {sl['mean_T']:>10.4f} "
                  f"{sl['mean_A']:>10.4f} {sl['max_A']:>10.4f} "
                  f"{Ak['400nm']:>10.4f} {Ak['550nm']:>10.4f} {Ak['700nm']:>10.4f} "
                  f"{r['recommendation']:<30}")
        else:
            print(f"{r['material']:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} "
                  f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'TMM FAILED':<30}")

    # Table 3: Final allowed/banned decision
    print("\n" + "=" * 80)
    print("FINAL MATERIAL DECISION")
    print("=" * 80)

    allowed = []
    banned = []
    warn = []
    for r in results:
        if "BAN" in r["recommendation"]:
            banned.append(r["material"])
        elif "WARN" in r["recommendation"]:
            warn.append(r["material"])
        else:
            allowed.append(r["material"])

    print(f"\nALLOWED ({len(allowed)}): {allowed}")
    print(f"WARN ({len(warn)}): {warn}")
    print(f"BANNED ({len(banned)}): {banned}")

    # Check which allowed materials have concerning absorption at edges
    print("\n--- Edge Absorption Warnings ---")
    for r in results:
        if r["material"] in allowed:
            Ak = r["single_layer_50nm_60deg_s"].get("A_at_key_wavelengths", {})
            max_edge_A = max(Ak.get("400nm", 0), Ak.get("1100nm", 0))
            if max_edge_A > 0.03:
                print(f"  {r['material']}: max edge A = {max_edge_A:.4f} "
                      f"(A@400={Ak.get('400nm', 0):.4f}, A@1100={Ak.get('1100nm', 0):.4f})")

    # ---- Save results ----
    output = {
        "audit_time": datetime.now().isoformat(),
        "conditions": {
            "theta_deg": THETA_DEG,
            "polarization": POLARIZATION,
            "substrate": SUBSTRATE,
            "wavelength_range_nm": [400, 1100],
            "test_thickness_nm": TEST_THICKNESS,
            "nk_database": NK_DATABASE,
        },
        "banned_materials": sorted(BANNED_MATERIALS),
        "allowed_materials": sorted(ALLOWED_MATERIALS),
        "final_allowed": allowed,
        "final_warn": warn,
        "final_banned": banned,
        "materials": results,
    }

    # Resolve output directory
    # PKG_ROOT is inner optogpt/optogpt/; dielectric_60deg_s/ is sibling to it
    project_root = PKG_ROOT.parent  # outer optogpt/
    results_dir = project_root / "dielectric_60deg_s" / "results" / "audit"
    if not results_dir.exists():
        results_dir = PKG_ROOT / "audit_results"  # fallback
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "material_audit.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nAudit report saved to: {output_path}")

    # Also save as readable text
    txt_path = results_dir / "material_audit.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Material nk/Absorption Audit Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Theta={THETA_DEG}°, pol={POLARIZATION}, {TEST_THICKNESS}nm single layer\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"--- {r['material']} ---\n")
            f.write(f"  Classification: {r['classification']}\n")
            f.write(f"  Recommendation: {r['recommendation']}\n")
            f.write(f"  k: max={r['max_k']:.2e}, mean={r['mean_k']:.2e}\n")
            sl = r["single_layer_50nm_60deg_s"]
            if sl["mean_A"] is not None:
                f.write(f"  Single-layer TMM: R={sl['mean_R']:.4f}, T={sl['mean_T']:.4f}, A={sl['mean_A']:.4f}\n")
            f.write("\n")

    print(f"Text report saved to: {txt_path}")


if __name__ == "__main__":
    main()
