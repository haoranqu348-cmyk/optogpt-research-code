#!/usr/bin/env python3
"""Compute Figure 1 spectra with a semi-infinite glass substrate."""

import csv
import json
from pathlib import Path

import numpy as np
from tmm import coh_tmm


ROOT = Path(__file__).resolve().parent
NK_DIR = ROOT / "optogpt_project" / "optogpt" / "optogpt" / "nk"
OUT = ROOT / "paper_figures" / "data" / "figure1_mgf2_classic_demo.json"


def load_nk(path):
    raw = np.genfromtxt(path, delimiter=",", names=True)
    wavelength_um = raw["wl"]
    return wavelength_um, raw["n"] + 1j * raw["k"]


def interpolate_nk(wavelengths_um, source_wavelengths, source_nk):
    return np.interp(wavelengths_um, source_wavelengths, source_nk.real) + 1j * np.interp(
        wavelengths_um, source_wavelengths, source_nk.imag
    )


def simulate_reflectance(materials, thicknesses_nm, theta_deg, wavelengths_nm, material_nk, glass_nk):
    values = []
    pol = "s" if theta_deg == 0 else None
    for index, wavelength_nm in enumerate(wavelengths_nm):
        for polarization in (("s", "p") if pol is None else (pol,)):
            values.append(
                100
                * coh_tmm(
                    polarization,
                    [1, *[material_nk[name][index] for name in materials], glass_nk[index]],
                    [np.inf, *thicknesses_nm, np.inf],
                    np.deg2rad(theta_deg),
                    wavelength_nm,
                )["R"]
            )
    if pol is not None:
        return {"s": values, "p": values}
    return {
        "s": values[0::2],
        "p": values[1::2],
    }


def main():
    wavelengths_um = np.arange(0.4, 1.101, 0.01)
    wavelengths_nm = (wavelengths_um * 1000).astype(int)
    mg_wl, mg_nk = load_nk(NK_DIR / "MgF2.csv")
    gl_wl, gl_nk = load_nk(NK_DIR / "Glass_Substrate.csv")
    mg = interpolate_nk(wavelengths_um, mg_wl, mg_nk)
    glass = interpolate_nk(wavelengths_um, gl_wl, gl_nk)

    center_nm = 550
    center_index = int(np.argmin(np.abs(wavelengths_nm - center_nm)))
    mg_center_n = float(mg[center_index].real)
    thickness_nm = center_nm / (4 * mg_center_n)

    classic_normal = simulate_reflectance(
        ["MgF2"], [thickness_nm], 0, wavelengths_nm, {"MgF2": mg}, glass
    )
    classic_oblique = simulate_reflectance(
        ["MgF2"], [thickness_nm], 60, wavelengths_nm, {"MgF2": mg}, glass
    )
    curves = {
        "R_0_s": classic_normal["s"],
        "R_0_p": classic_normal["p"],
        "R_60_s": classic_oblique["s"],
        "R_60_p": classic_oblique["p"],
    }

    # Actual s-priority AI candidate from the joint model high-transmission pool.
    selection = json.loads(
        (ROOT / "paper_figures" / "data" / "figure4_s_priority_diverse_selection.json").read_text()
    )
    candidate = selection["selected_candidates"][0]
    candidate_materials = []
    candidate_thicknesses = []
    candidate_nk = {"MgF2": mg}
    for token in candidate["tokens"]:
        material, thickness = token.rsplit("_", 1)
        candidate_materials.append(material)
        candidate_thicknesses.append(int(thickness))
        if material not in candidate_nk:
            material_wl, material_nk = load_nk(NK_DIR / f"{material}.csv")
            candidate_nk[material] = interpolate_nk(wavelengths_um, material_wl, material_nk)
    candidate_reflectance = simulate_reflectance(
        candidate_materials, candidate_thicknesses, 60, wavelengths_nm, candidate_nk, glass
    )
    curves["R_60_ai_s"] = candidate_reflectance["s"]
    curves["R_60_ai_p"] = candidate_reflectance["p"]
    ai_rs = np.asarray(candidate_reflectance["s"], dtype=float) / 100
    ai_rp = np.asarray(candidate_reflectance["p"], dtype=float) / 100
    target_r = 0.05
    ai_mean_ts = float(np.mean(1 - ai_rs))
    ai_mean_tp = float(np.mean(1 - ai_rp))
    ai_e_s = float(np.mean(np.abs(ai_rs - target_r)))
    ai_e_p = float(np.mean(np.abs(ai_rp - target_r)))

    record = {
        "design_contract": {
            "material": "MgF2",
            "substrate": "Glass_Substrate",
            "design_wavelength_nm": center_nm,
            "quarter_wave_thickness_nm": thickness_nm,
            "mgf2_n_at_design_wavelength": mg_center_n,
            "model": "coherent multilayer air/film/glass TMM with semi-infinite substrate; Figure 1 curves use this contract",
            "literature_basis": "https://www.rp-photonics.com/anti_reflection_coatings.html",
            "literature_note": "Single-layer quarter-wave AR coating; MgF2 on glass is a classic practical example.",
        },
        "wavelengths_nm": wavelengths_nm.tolist(),
        "curves_percent": curves,
        "ai_candidate": {
            "tokens": candidate["tokens"],
            "E_s": ai_e_s,
            "E_p": ai_e_p,
            "E_joint": (ai_e_s + ai_e_p) / 2,
            "mean_Ts": ai_mean_ts,
            "mean_Tp": ai_mean_tp,
            "model": "semi-infinite glass substrate recomputation",
        },
    }
    OUT.write_text(json.dumps(record, indent=2))
    csv_path = OUT.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength_nm", *curves.keys()])
        for index, wavelength_nm in enumerate(wavelengths_nm):
            writer.writerow([wavelength_nm, *[values[index] for values in curves.values()]])
    print(json.dumps({"json": str(OUT), "csv": str(csv_path), "thickness_nm": thickness_nm, "n": mg_center_n}, indent=2))


if __name__ == "__main__":
    main()
