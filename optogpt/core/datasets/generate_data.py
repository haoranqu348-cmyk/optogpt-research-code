"""
Generate training data for OptoGPT at fixed 60° s-polarization.

This script generates multilayer thin-film structures, computes their
R+T spectra at theta=60°, pol="s" using the project's existing TMM
simulation, and saves them in a format directly readable by OptoGPT's
PrepareData loader.

Usage:
    # Generate 10K smoke-test samples:
    python core/datasets/generate_data.py --num_samples 10000 --output_dir ./data_60deg_s

    # Resume from checkpoint:
    python core/datasets/generate_data.py --num_samples 10000 --output_dir ./data_60deg_s --resume

    # Generate train/val/test split:
    python core/datasets/generate_data.py --num_samples 10000 --output_dir ./data_60deg_s --split 0.8 0.1 0.1
"""

import os
import sys
import json
import time
import argparse
import pickle as pkl
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch

# ---- Path setup ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # optogpt/optogpt/
sys.path.insert(0, str(PROJECT_ROOT))

from core.datasets.sim import spectrum, load_materials, wavelengths

# ---- Constants ----
# These MUST match the pretrained checkpoint exactly
CHECKPOINT_PATH = PROJECT_ROOT.parent / "model" / "optogpt.pt"  # optogpt/model/optogpt.pt
NK_DATABASE = str(PROJECT_ROOT / "nk")

# Fixed simulation conditions for this stage
THETA_DEG = 60          # incidence angle in degrees
POLARIZATION = "p"      # s-polarization
INCIDENT_MEDIUM = "air" # incident medium (n=1 in spectrum())
SUBSTRATE = "Glass_Substrate"
SUBSTRATE_THICK = 500000  # nm (effectively semi-infinite)

# Wavelength: 400–1100 nm, step 10 nm → 71 points
# The wavelengths array from sim.py is in microns: 0.4 to 1.1 step 0.01
WAVELENGTHS_NM = np.arange(400, 1101, 10)  # 71 points
N_WAVELENGTHS = len(WAVELENGTHS_NM)

# Structure generation constraints
MIN_LAYERS = 1
MAX_LAYERS = 20  # based on max_len=22 in config (BOS + materials + EOS)

# ---- Dielectric-only material filter ----
# Exclude conductive/metallic materials to focus on dielectric multilayers.
# Conductors removed: Ag, Al, Ge, ITO, Si, TiN, ZnS, ZnSe
DIELECTRIC_MATERIALS = [
    "TiO2", "Si3N4", "ZnO", "Al2O3", "AlN",
    "HfO2", "MgF2", "MgO", "SiO2", "Ta2O5",
]


def load_checkpoint_materials(checkpoint_path):
    """Extract materials and valid thicknesses from pretrained checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    wd = ckpt["configs"].struc_word_dict

    mat_thicknesses = defaultdict(list)
    for token in wd:
        if token in ("UNK", "PAD", "BOS", "EOS"):
            continue
        parts = token.split("_")
        if len(parts) == 2:
            mat_thicknesses[parts[0]].append(int(parts[1]))

    # Sort and validate
    materials = sorted(mat_thicknesses.keys())
    for mat in materials:
        mat_thicknesses[mat] = sorted(mat_thicknesses[mat])

    print(f"Loaded {len(materials)} materials from checkpoint: {materials}")
    print(f"  Total material-thickness tokens: {sum(len(v) for v in mat_thicknesses.values())}")
    return materials, mat_thicknesses


def random_structure(materials, mat_thicknesses, rng, min_layers=MIN_LAYERS, max_layers=MAX_LAYERS):
    """Generate a random multilayer structure compatible with the checkpoint vocab."""
    n_layers = rng.integers(min_layers, max_layers + 1)
    structure = []
    for _ in range(n_layers):
        mat = rng.choice(materials)
        thick = rng.choice(mat_thicknesses[mat])
        structure.append(f"{mat}_{int(thick)}")
    return structure


def simulate_structure(structure, nk_dict, theta_deg, pol):
    """Run TMM simulation for a structure. Returns (R_list, T_list) or (None, None) on failure."""
    materials_list = []
    thickness_list = []
    for token in structure:
        mat, thick_str = token.split("_")
        materials_list.append(mat)
        thickness_list.append(float(thick_str))

    try:
        result = spectrum(
            materials=materials_list,
            thickness=thickness_list,
            pol=pol,
            theta=theta_deg,
            wavelengths=wavelengths,
            nk_dict=nk_dict,
            substrate=SUBSTRATE,
            substrate_thick=SUBSTRATE_THICK,
        )
        # result is R + T concatenated: [R_0..R_70, T_0..T_70]
        half = len(result) // 2
        R = np.array(result[:half], dtype=np.float64)
        T = np.array(result[half:], dtype=np.float64)
        return R, T
    except Exception as e:
        return None, None


def validate_spectrum(R, T, structure, tol=1e-6):
    """
    Check physical validity of a simulated spectrum.

    Returns (is_valid, warnings).
    """
    warnings = []

    # 1. Check for NaN / Inf
    if np.any(np.isnan(R)) or np.any(np.isnan(T)):
        return False, ["NaN in R or T"]
    if np.any(np.isinf(R)) or np.any(np.isinf(T)):
        return False, ["Inf in R or T"]

    # 2. Check reasonable ranges: R, T should be in [0, 1] with small tolerance
    if np.any(R < -tol) or np.any(R > 1 + tol):
        warnings.append(f"R out of [0,1]: min={R.min():.6f}, max={R.max():.6f}")
    if np.any(T < -tol) or np.any(T > 1 + tol):
        warnings.append(f"T out of [0,1]: min={T.min():.6f}, max={T.max():.6f}")

    # 3. Check R+T <= 1 (conservation of energy; allow small numerical error)
    rt_sum = R + T
    if np.any(rt_sum > 1 + 0.01):  # 1% tolerance for numerical issues
        worst = rt_sum.max()
        warnings.append(f"R+T > 1: max={worst:.6f} at index {np.argmax(rt_sum)}")

    # 4. Check for exactly zero T (possible with metals) — not an error, just note
    if np.all(T < tol) and np.all(R < 0.01):
        warnings.append("Both R and T near zero — possible simulation issue")

    return True, warnings


def save_chunk(structures, specs, output_dir, chunk_idx, prefix="chunk"):
    """Save a chunk of generated data as pickle files."""
    os.makedirs(output_dir, exist_ok=True)
    struct_path = os.path.join(output_dir, f"{prefix}_{chunk_idx:04d}_struct.pkl")
    spec_path = os.path.join(output_dir, f"{prefix}_{chunk_idx:04d}_spec.pkl")

    with open(struct_path, "wb") as f:
        pkl.dump(structures, f)
    with open(spec_path, "wb") as f:
        pkl.dump(specs, f)

    return struct_path, spec_path


def merge_chunks(output_dir, output_prefix, chunk_prefix="chunk"):
    """Merge all chunk files into final train/dev/test splits."""
    all_structs = []
    all_specs = []

    struct_files = sorted(Path(output_dir).glob(f"{chunk_prefix}_*_struct.pkl"))
    spec_files = sorted(Path(output_dir).glob(f"{chunk_prefix}_*_spec.pkl"))

    for sf, spf in zip(struct_files, spec_files):
        with open(sf, "rb") as f:
            all_structs.extend(pkl.load(f))
        with open(spf, "rb") as f:
            all_specs.extend(pkl.load(f))

    print(f"Merged {len(struct_files)} chunks: {len(all_structs)} total samples")

    # Convert specs to float32 for training
    all_specs = [s.astype(np.float32) for s in all_specs]

    return all_structs, all_specs


def deduplicate_structures(structures, specs):
    """Remove duplicate structures by hashing token sequences. Keeps first occurrence."""
    seen = set()
    unique_idx = []
    for i, s in enumerate(structures):
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_idx.append(i)
    n_dup = len(structures) - len(unique_idx)
    if n_dup > 0:
        print(f"  Removed {n_dup} duplicate structures ({n_dup/len(structures)*100:.1f}%)")
    return [structures[i] for i in unique_idx], np.array([specs[i] for i in unique_idx])


def split_data(structures, specs, split_ratios, rng):
    """Split data into train/val/test with given ratios. No structure leakage."""
    n = len(structures)
    indices = rng.permutation(n)

    ratios = np.array(split_ratios)
    ratios = ratios / ratios.sum()
    cumsum = np.cumsum(ratios)
    split_points = (cumsum[:-1] * n).astype(int)

    splits = np.split(indices, split_points)
    return splits[0], splits[1] if len(splits) > 2 else splits[1], splits[-1]


def main():
    parser = argparse.ArgumentParser(description="Generate 60° s-pol training data for OptoGPT")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of valid samples to generate")
    parser.add_argument("--output_dir", type=str, default="./data_60deg_s", help="Output directory")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Samples per chunk (for checkpointing)")
    parser.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help="Train/val/test split ratios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from existing chunks")
    parser.add_argument("--merge_only", action="store_true", help="Only merge existing chunks, don't generate")
    parser.add_argument("--min_layers", type=int, default=1, help="Minimum number of layers")
    parser.add_argument("--max_layers", type=int, default=20, help="Maximum number of layers")
    parser.add_argument("--max_failures", type=int, default=100,
                        help="Max consecutive simulation failures before aborting")
    args = parser.parse_args()

    # ---- Setup ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load checkpoint materials
    materials, mat_thicknesses = load_checkpoint_materials(str(CHECKPOINT_PATH))

    # ---- Filter to dielectric-only ----
    materials = [m for m in materials if m in DIELECTRIC_MATERIALS]
    mat_thicknesses = {m: t for m, t in mat_thicknesses.items() if m in DIELECTRIC_MATERIALS}
    excluded = [m for m in DIELECTRIC_MATERIALS if m not in materials]
    if excluded:
        print(f"WARNING: These dielectric materials not found in checkpoint vocab: {excluded}")
    print(f"After dielectric filter: {len(materials)} materials: {materials}")
    print(f"  Excluded conductors: Ag, Al, Ge, ITO, Si, TiN, ZnS, ZnSe")

    # Load nk data from correct path (include substrate!)
    print(f"Loading nk data from: {NK_DATABASE}")
    all_mats_to_load = materials + [SUBSTRATE]
    nk_dict = load_materials(all_mats=all_mats_to_load, wavelengths=wavelengths, DATABASE=NK_DATABASE)
    print(f"Loaded nk for {len(nk_dict)} materials")

    # ---- Save generation config ----
    config = {
        "description": "60° s-polarization training data for OptoGPT fine-tuning",
        "theta_deg": THETA_DEG,
        "polarization": POLARIZATION,
        "incident_medium": INCIDENT_MEDIUM,
        "substrate": SUBSTRATE,
        "substrate_thick_nm": SUBSTRATE_THICK,
        "wavelengths_nm": WAVELENGTHS_NM.tolist(),
        "n_wavelengths": N_WAVELENGTHS,
        "spectrum_dim": N_WAVELENGTHS * 2,
        "materials": materials,
        "material_count": len(materials),
        "min_layers": args.min_layers,
        "max_layers": args.max_layers,
        "seed": args.seed,
        "num_samples_target": args.num_samples,
        "checkpoint_source": str(CHECKPOINT_PATH),
        "generated_at": datetime.now().isoformat(),
    }
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # ---- Generate or merge ----
    if args.merge_only:
        all_structs, all_specs = merge_chunks(output_dir, "data", "chunk")
    else:
        # Determine start chunk
        if args.resume:
            existing_chunks = sorted(output_dir.glob("chunk_*_struct.pkl"))
            if existing_chunks:
                last_chunk = existing_chunks[-1].stem
                start_chunk = int(last_chunk.split("_")[1]) + 1
                # Count existing valid samples
                existing_count = 0
                for sf in existing_chunks:
                    with open(sf, "rb") as f:
                        existing_count += len(pkl.load(f))
                print(f"Resuming from chunk {start_chunk}, {existing_count} existing samples")
            else:
                start_chunk = 0
                existing_count = 0
        else:
            start_chunk = 0
            existing_count = 0
            # Clean old chunks
            for f in output_dir.glob("chunk_*"):
                f.unlink()

        structures = []
        specs = []
        n_generated = existing_count
        n_failures = 0
        consecutive_failures = 0
        chunk_idx = start_chunk
        chunk_structs = []
        chunk_specs = []

        t_start = time.time()
        print(f"Generating {args.num_samples} samples (theta={THETA_DEG}°, pol={POLARIZATION})...")

        while n_generated < args.num_samples:
            struc = random_structure(materials, mat_thicknesses, rng, args.min_layers, args.max_layers)
            R, T = simulate_structure(struc, nk_dict, THETA_DEG, POLARIZATION)

            if R is None:
                n_failures += 1
                consecutive_failures += 1
                if consecutive_failures >= args.max_failures:
                    print(f"ERROR: {args.max_failures} consecutive failures. Aborting.")
                    print("  Check nk data, simulation parameters, and materials.")
                    break
                continue

            consecutive_failures = 0
            is_valid, warnings = validate_spectrum(R, T, struc)

            if not is_valid:
                n_failures += 1
                continue

            # Combine R+T into full spectrum
            full_spec = np.concatenate([R, T]).astype(np.float32)

            chunk_structs.append(struc)
            chunk_specs.append(full_spec)

            if len(chunk_structs) >= args.chunk_size:
                struct_path, spec_path = save_chunk(chunk_structs, chunk_specs, output_dir, chunk_idx)
                n_generated += len(chunk_structs)
                elapsed = time.time() - t_start
                rate = n_generated / elapsed if elapsed > 0 else 0
                print(f"  Chunk {chunk_idx:04d}: {n_generated}/{args.num_samples} samples "
                      f"({rate:.1f} samp/s), {n_failures} failures")
                chunk_structs = []
                chunk_specs = []
                chunk_idx += 1

        # Save remaining
        if chunk_structs:
            struct_path, spec_path = save_chunk(chunk_structs, chunk_specs, output_dir, chunk_idx)
            n_generated += len(chunk_structs)
            print(f"  Final chunk {chunk_idx:04d}: {n_generated}/{args.num_samples} samples")

        elapsed = time.time() - t_start
        print(f"\nGeneration complete: {n_generated} valid samples in {elapsed:.1f}s")
        print(f"  Failures: {n_failures}")
        print(f"  Rate: {n_generated / elapsed:.1f} samp/s")

    # ---- Merge and split ----
    all_structs, all_specs = merge_chunks(output_dir, "data", "chunk")

    # Deduplicate before splitting (same structure → same spectrum, keep first)
    all_structs, all_specs = deduplicate_structures(all_structs, all_specs)

    # Combine into final dataset files
    train_idx, val_idx, test_idx = split_data(all_structs, all_specs, args.split, rng)

    def save_split(structs, specs, indices, name):
        struct_out = [structs[i] for i in indices]
        spec_out = np.array([specs[i] for i in indices], dtype=np.float32)

        with open(output_dir / f"Structure_{name}.pkl", "wb") as f:
            pkl.dump(struct_out, f)
        with open(output_dir / f"Spectrum_{name}.pkl", "wb") as f:
            pkl.dump(spec_out, f)
        print(f"  {name}: {len(struct_out)} structures, spec shape {spec_out.shape}")

    save_split(all_structs, all_specs, train_idx, "train")
    save_split(all_structs, all_specs, val_idx, "dev")
    save_split(all_structs, all_specs, test_idx, "test")

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print(f"Data generation summary:")
    print(f"  Output: {output_dir.resolve()}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Theta: {THETA_DEG}°, Polarization: {POLARIZATION}")
    print(f"  Spectrum shape: ({N_WAVELENGTHS*2},)  [R_400..R_1100, T_400..T_1100]")
    print(f"  Materials: {len(materials)} ({', '.join(materials)})")
    print(f"  Seed: {args.seed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
