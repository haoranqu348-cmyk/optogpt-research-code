"""
joint_sp/constants.py — Unified constants for joint s+p polarization model.

This is the single source of truth for all joint_sp modules.
All other files import from here; hardcoded theta, materials, dimensions are forbidden.
"""

import numpy as np

# ============================================================
# Simulation Parameters
# ============================================================
THETA_DEG = 60
THETA_RAD = THETA_DEG * np.pi / 180.0
WAVELENGTHS_NM = np.arange(400, 1101, 10)  # 71 points, 400–1100 nm
N_WAVELENGTHS = len(WAVELENGTHS_NM)          # 71
WAVELENGTHS_UM = WAVELENGTHS_NM / 1000.0

# ============================================================
# Spectrum Layout
# ============================================================
# Joint spectrum: [Rs(71), Ts(71), Rp(71), Tp(71)]
SPEC_LAYOUT = ["Rs", "Ts", "Rp", "Tp"]
BRANCH_DIM = 142       # single polarization: R(71) + T(71)
SPEC_DIM = 284         # joint: Rs + Ts + Rp + Tp

# ============================================================
# Materials
# ============================================================
ALLOWED_MATERIALS = [
    "Al2O3", "AlN", "HfO2", "MgF2", "MgO",
    "Si3N4", "SiO2", "Ta2O5", "TiO2", "ZnO",
]
BANNED_MATERIALS = {"Ag", "Al", "TiN", "Ge", "Si", "ITO", "ZnS", "ZnSe"}

# ============================================================
# Structure Constraints
# ============================================================
MIN_LAYERS = 1
MAX_LAYERS = 20
SUBSTRATE = "Glass_Substrate"
SUBSTRATE_THICK_NM = 500000  # 500 µm

# ============================================================
# Paths (relative to optogpt/ project root)
# ============================================================
PRETRAINED_CKPT = "model/optogpt.pt"
NK_DATABASE = "optogpt/nk"

# ============================================================
# Special Token IDs (must match pretrained checkpoint)
# ============================================================
UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3
SPECIAL_IDS = {UNK_ID, PAD_ID, BOS_ID, EOS_ID}

# ============================================================
# GA/PSO thresholds
# ============================================================
MIN_IMPROVEMENT = 1e-6  # minimum E_joint improvement to accept perturbed result

# ============================================================
# Structure normalization
# ============================================================

def normalize_structure_tokens(tokens, word_dict=None, allowed_materials=None):
    """
    Normalize tokens: quietly strip BOS/EOS/PAD, validate the rest.
    UNK always raises. Used for compatibility / generic normalization.
    """
    if allowed_materials is None:
        allowed_materials = set(ALLOWED_MATERIALS)
    else:
        allowed_materials = set(allowed_materials)

    # Strip BOS/EOS/PAD (but NOT UNK)
    cleaned = [t for t in tokens if t not in ("BOS", "EOS", "PAD")]

    if any(t == "UNK" or t.startswith("UNK") for t in tokens):
        raise ValueError(f"UNK token found in structure: {tokens}")

    # Validate remaining tokens
    return _validate_layer_tokens(cleaned, word_dict, allowed_materials, tokens)


def validate_disk_structure_tokens(tokens, word_dict=None, allowed_materials=None):
    """
    STRICT validator for disk Structure_*.pkl files.
    BOS, EOS, PAD, UNK are ALL forbidden. Must return clean layer tokens.
    """
    if allowed_materials is None:
        allowed_materials = set(ALLOWED_MATERIALS)
    else:
        allowed_materials = set(allowed_materials)

    for t in tokens:
        if t in ("BOS", "EOS", "PAD", "UNK"):
            raise ValueError(f"Disk structure contains forbidden token '{t}': {tokens}")

    return _validate_layer_tokens(tokens, word_dict, allowed_materials, tokens)


def _validate_layer_tokens(tokens, word_dict, allowed_materials, original_tokens):
    """Shared validation for layer tokens (materials, thickness, vocab, layer count)."""
    for t in tokens:
        if '_' not in t:
            raise ValueError(f"Invalid token format (missing '_'): '{t}' in {original_tokens}")
        parts = t.rsplit('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid token format: '{t}' in {original_tokens}")
        mat, thick_str = parts
        if mat not in allowed_materials:
            raise ValueError(f"Material '{mat}' not in allowed set. Token: '{t}'")
        if not thick_str.isdigit():
            raise ValueError(f"Thickness not integer: '{t}'")
        if thick_str != str(int(thick_str)):
            raise ValueError(f"Thickness must use canonical decimal form: '{t}'")
        if int(thick_str) <= 0:
            raise ValueError(f"Thickness must be positive: '{t}'")
        if word_dict is not None and t not in word_dict:
            raise ValueError(f"Token '{t}' not in vocabulary")

    n_layers = len(tokens)
    if n_layers < MIN_LAYERS or n_layers > MAX_LAYERS:
        raise ValueError(f"Layer count {n_layers} not in [{MIN_LAYERS}, {MAX_LAYERS}]")

    return tokens


def structure_hash_from_tokens(tokens):
    """Compute SHA-256 hash from pure layer tokens (without BOS/EOS)."""
    # Normalize first
    cleaned = normalize_structure_tokens(tokens)
    import hashlib
    return hashlib.sha256("|".join(cleaned).encode()).hexdigest()


def validate_joint_spectrum(spec, context="joint spectrum", energy_tolerance=5e-4):
    """Validate the [Rs, Ts, Rp, Tp] numerical and passive-energy contract."""
    value = np.asarray(spec, dtype=np.float32)
    if value.shape != (SPEC_DIM,):
        raise ValueError(f"{context} shape {value.shape}, expected ({SPEC_DIM},)")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{context} contains NaN/Inf")
    for offset, pol in ((0, "s"), (BRANCH_DIM, "p")):
        reflectance = value[offset:offset + N_WAVELENGTHS]
        transmittance = value[offset + N_WAVELENGTHS:offset + BRANCH_DIM]
        if np.min(reflectance) < -1e-5 or np.min(transmittance) < -1e-5:
            raise ValueError(f"{context} contains negative {pol}-polarization R/T")
        if (np.max(reflectance) > 1.0001 or np.max(transmittance) > 1.0001 or
                np.max(reflectance + transmittance) > 1.0 + energy_tolerance):
            raise ValueError(f"{context} violates passive {pol}-polarization energy bounds")
    return value
