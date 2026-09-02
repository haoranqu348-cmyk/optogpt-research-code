"""
joint_sp/self_improving/ood_targets.py — Generate out-of-distribution joint targets.

Produces 284-dim [Rs, Ts, Rp, Tp] target spectra for self-improving:
  - Broadband high-T: Rs=0, Ts=1, Rp=0, Tp=1
  - Gaussian/Double-Gaussian passband targets
  - DBR-like: random structures -> TMM -> physical targets
  - Random R/T curves satisfying R+T<=1 constraint
"""

import numpy as np
from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_JOINT_SP = _SCRIPT_DIR.parent
_PKG_ROOT = _JOINT_SP.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from joint_sp.constants import (
    SPEC_DIM, BRANCH_DIM, WAVELENGTHS_NM, N_WAVELENGTHS,
    ALLOWED_MATERIALS, THETA_DEG, SUBSTRATE, SUBSTRATE_THICK_NM,
)


def generate_broadband_high_T(n=1):
    """Broadband high-T: Rs=0, Ts=1, Rp=0, Tp=1 (lossless)."""
    n_pts = BRANCH_DIM // 2  # 71
    Rs = np.zeros(n_pts, dtype=np.float32)
    Ts = np.ones(n_pts, dtype=np.float32)
    Rp = np.zeros(n_pts, dtype=np.float32)
    Tp = np.ones(n_pts, dtype=np.float32)
    target = np.concatenate([Rs, Ts, Rp, Tp])
    return np.tile(target, (n, 1)) if n > 1 else target


def _gaussian(x, center, sigma, amplitude=1.0):
    """Gaussian function."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def generate_gaussian_targets(n=10, rng=None):
    """
    Generate Gaussian passband targets.
    T = gaussian, R = 1 - T (lossless approximation, clamped).
    Applied identically for s and p (unpolarized target).
    """
    if rng is None:
        rng = np.random.RandomState()

    n_pts = BRANCH_DIM // 2
    wl_nm = WAVELENGTHS_NM.astype(np.float32)
    targets = []

    for _ in range(n):
        center = rng.uniform(500, 1000)  # center wavelength nm
        sigma = rng.uniform(30, 200)     # width nm
        amplitude = rng.uniform(0.7, 1.0)

        T = _gaussian(wl_nm, center, sigma, amplitude)
        T = np.clip(T, 0.0, 1.0)
        R = np.clip(1.0 - T, 0.0, 1.0)

        # Same target for s and p
        target = np.concatenate([R.astype(np.float32), T.astype(np.float32),
                                  R.astype(np.float32), T.astype(np.float32)])
        targets.append(target)

    return np.array(targets)


def generate_double_gaussian_targets(n=10, rng=None):
    """Generate double-Gaussian (bandpass) targets."""
    if rng is None:
        rng = np.random.RandomState()

    n_pts = BRANCH_DIM // 2
    wl_nm = WAVELENGTHS_NM.astype(np.float32)
    targets = []

    for _ in range(n):
        center1 = rng.uniform(450, 650)
        center2 = rng.uniform(750, 1000)
        sigma1 = rng.uniform(20, 100)
        sigma2 = rng.uniform(20, 100)
        amp1 = rng.uniform(0.5, 1.0)
        amp2 = rng.uniform(0.5, 1.0)

        T = _gaussian(wl_nm, center1, sigma1, amp1) + _gaussian(wl_nm, center2, sigma2, amp2)
        T = np.clip(T, 0.0, 1.0)
        R = np.clip(1.0 - T, 0.0, 1.0)

        target = np.concatenate([R.astype(np.float32), T.astype(np.float32),
                                  R.astype(np.float32), T.astype(np.float32)])
        targets.append(target)

    return np.array(targets)


def generate_dbr_targets(n=10, rng=None, nk_dict=None, materials=None, theta=THETA_DEG):
    """
    Generate DBR-like targets: random dielectric structures -> TMM -> physical spectra.
    These are physically realizable targets that should be easier for the model.
    """
    if rng is None:
        rng = np.random.RandomState()

    if materials is None:
        materials = ALLOWED_MATERIALS

    targets = []

    for _ in range(n):
        n_layers = rng.randint(4, 16)
        mats = [materials[rng.randint(0, len(materials))] for _ in range(n_layers)]
        # Simple thickness: random within 10-200nm
        thicks = [rng.randint(10, 201) for _ in range(n_layers)]

        if nk_dict is not None:
            try:
                from optogpt.core.datasets.sim import spectrum
                wl = np.arange(0.4, 1.101, 0.01)

                sim_s = spectrum(mats, thicks, pol='s', theta=theta,
                                  wavelengths=wl, nk_dict=nk_dict,
                                  substrate=SUBSTRATE, substrate_thick=SUBSTRATE_THICK_NM)
                sim_p = spectrum(mats, thicks, pol='p', theta=theta,
                                  wavelengths=wl, nk_dict=nk_dict,
                                  substrate=SUBSTRATE, substrate_thick=SUBSTRATE_THICK_NM)

                target = np.concatenate([sim_s, sim_p]).astype(np.float32)
                # Interpolate to 71+71+71+71 = 284 if needed
                if len(target) != SPEC_DIM:
                    from scipy.interpolate import interp1d
                    wl_orig = np.linspace(0.4, 1.1, len(target) // 4)
                    wl_target = np.arange(0.4, 1.101, 0.01)
                    parts = []
                    for i in range(4):
                        chunk = target[i * (len(target) // 4):(i + 1) * (len(target) // 4)]
                        fn = interp1d(wl_orig, chunk, kind='linear', fill_value='extrapolate')
                        parts.append(fn(wl_target))
                    target = np.concatenate(parts).astype(np.float32)

                targets.append(target)
            except Exception:
                continue

    return np.array(targets) if targets else np.empty((0, SPEC_DIM))


def generate_random_rt_targets(n=10, rng=None):
    """Generate random R/T profiles satisfying R+T<=1."""
    if rng is None:
        rng = np.random.RandomState()

    n_pts = BRANCH_DIM // 2
    targets = []

    for _ in range(n):
        # Smooth random T profiles
        x = np.linspace(0, 1, n_pts)
        # Random control points
        n_ctrl = rng.randint(3, 8)
        ctrl_x = np.sort(rng.uniform(0, 1, n_ctrl))
        ctrl_y = rng.uniform(0.3, 1.0, n_ctrl)

        # Interpolate
        T_s = np.interp(x, np.concatenate([[0], ctrl_x, [1]]),
                         np.concatenate([[ctrl_y[0]], ctrl_y, [ctrl_y[-1]]]))
        T_s = np.clip(T_s, 0.0, 1.0)

        # Different profiles for s and p
        ctrl_y2 = rng.uniform(0.3, 1.0, n_ctrl)
        T_p = np.interp(x, np.concatenate([[0], ctrl_x, [1]]),
                         np.concatenate([[ctrl_y2[0]], ctrl_y2, [ctrl_y2[-1]]]))
        T_p = np.clip(T_p, 0.0, 1.0)

        R_s = np.clip(1.0 - T_s + rng.uniform(-0.05, 0.05, n_pts), 0.0, 1.0)
        R_p = np.clip(1.0 - T_p + rng.uniform(-0.05, 0.05, n_pts), 0.0, 1.0)

        # Ensure R+T <= 1
        R_s = np.clip(R_s, 0.0, 1.0 - T_s)
        R_p = np.clip(R_p, 0.0, 1.0 - T_p)

        target = np.concatenate([
            R_s.astype(np.float32), T_s.astype(np.float32),
            R_p.astype(np.float32), T_p.astype(np.float32),
        ])
        targets.append(target)

    return np.array(targets)


def generate_all_ood_targets(n_broadband=1, n_gaussian=20, n_double_gaussian=10,
                              n_dbr=20, n_random=10, seed=42, nk_dict=None,
                              theta=THETA_DEG):
    """
    Generate a comprehensive set of OOD targets.

    Returns:
        targets: (N, 284) float32 array
        labels: list of target type strings
    """
    rng = np.random.RandomState(seed)
    all_targets = []
    all_labels = []

    # Broadband
    bb = generate_broadband_high_T(n_broadband)
    if n_broadband > 0:
        all_targets.append(bb.reshape(n_broadband, -1) if bb.ndim == 1 else bb)
        all_labels.extend(['broadband_high_T'] * n_broadband)

    # Gaussian
    gauss = generate_gaussian_targets(n_gaussian, rng)
    if len(gauss) > 0:
        all_targets.append(gauss)
        all_labels.extend(['gaussian'] * len(gauss))

    # Double Gaussian
    dg = generate_double_gaussian_targets(n_double_gaussian, rng)
    if len(dg) > 0:
        all_targets.append(dg)
        all_labels.extend(['double_gaussian'] * len(dg))

    # DBR
    dbr = generate_dbr_targets(n_dbr, rng, nk_dict, theta=theta)
    if len(dbr) > 0:
        all_targets.append(dbr)
        all_labels.extend(['dbr'] * len(dbr))

    # Random
    rand = generate_random_rt_targets(n_random, rng)
    if len(rand) > 0:
        all_targets.append(rand)
        all_labels.extend(['random_rt'] * len(rand))

    if all_targets:
        targets = np.concatenate([t.reshape(-1, SPEC_DIM) for t in all_targets if len(t) > 0], axis=0)
    else:
        targets = np.empty((0, SPEC_DIM))

    return targets.astype(np.float32), all_labels


if __name__ == "__main__":
    # Quick test
    targets, labels = generate_all_ood_targets(n_broadband=1, n_gaussian=5,
                                                n_double_gaussian=3, n_dbr=5,
                                                n_random=3, seed=42)
    print(f"Generated {len(targets)} OOD targets")
    for i, (t, l) in enumerate(zip(targets[:5], labels[:5])):
        Rs = t[:71].mean()
        Ts = t[71:142].mean()
        Rp = t[142:213].mean()
        Tp = t[213:284].mean()
        print(f"  {i}: {l:20s} | Rs={Rs:.3f} Ts={Ts:.3f} Rp={Rp:.3f} Tp={Tp:.3f}")
