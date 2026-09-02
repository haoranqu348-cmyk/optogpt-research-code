"""
joint_sp/decoder.py — Joint s+p multi-candidate decoder with TMM re-ranking.

Extends the single-polarization decoder logic to handle 284-dim joint spectra.
Key features:
  - Logits mask: ban non-dielectric materials
  - Greedy / top-k / top-p sampling decode
  - Multi-candidate generation with deduplication
  - TMM re-ranking: evaluates both s and p polarizations simultaneously
"""

import torch
import numpy as np
from torch.autograd import Variable
from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from optogpt.core.models.transformer import subsequent_mask
from joint_sp.constants import (
    SPEC_DIM, BRANCH_DIM, ALLOWED_MATERIALS, BANNED_MATERIALS,
    THETA_DEG, WAVELENGTHS_NM, WAVELENGTHS_UM, SUBSTRATE, SUBSTRATE_THICK_NM,
    UNK_ID, PAD_ID, BOS_ID, EOS_ID, SPECIAL_IDS, MAX_LAYERS,
)


# ============================================================
# Logits Masking
# ============================================================

def build_joint_logits_mask(word_dict, allowed_materials=None, max_layers=MAX_LAYERS):
    """
    Build a boolean mask for joint decoding.

    Allowed tokens:
      - All thickness tokens for allowed materials
      - EOS (to terminate)
      - BOS, PAD, UNK (special, but BOS banned after first token)

    Banned tokens:
      - All tokens belonging to banned materials
      - BOS after first position (handled in decode loop, not mask)

    Args:
        word_dict: {token_str: token_id}
        allowed_materials: list of allowed material names
        max_layers: max structure layers (informational)

    Returns:
        mask: (vocab_size,) bool tensor, True=allowed
        special_ids: dict {name: id}
    """
    if allowed_materials is None:
        allowed_materials = ALLOWED_MATERIALS

    vocab_size = len(word_dict)
    mask = torch.zeros(vocab_size, dtype=torch.bool)

    # Build reverse mapping
    id_to_token = {v: k for k, v in word_dict.items()}

    # Allow special tokens
    special_ids = {}
    for name in ['UNK', 'PAD', 'BOS', 'EOS']:
        tid = word_dict.get(name)
        if tid is not None:
            mask[tid] = True
            special_ids[name] = tid

    # Allow all thickness tokens for allowed materials
    for tid, token in id_to_token.items():
        if tid in SPECIAL_IDS:
            continue
        # Token format: Material_Thickness, e.g., "SiO2_100"
        if '_' in token:
            mat = token.split('_')[0]
            if mat in allowed_materials:
                mask[tid] = True

    n_allowed = mask.sum().item()
    n_banned = vocab_size - n_allowed
    print(f"[build_joint_logits_mask] Vocab={vocab_size}, allowed={n_allowed}, "
          f"banned={n_banned}")

    return mask, special_ids


def apply_logits_mask(logits, mask):
    """
    Apply mask to logits: banned tokens -> -inf.

    Args:
        logits: (batch_size, vocab_size) or (vocab_size,)
        mask: (vocab_size,) bool tensor, True=allowed
    Returns:
        masked logits, same shape as input
    """
    mask = mask.to(logits.device)
    if mask.dim() == 1 and logits.dim() == 2:
        mask = mask.unsqueeze(0).expand_as(logits)
    return logits.masked_fill(~mask, float("-inf"))


# ============================================================
# Sampling Utilities
# ============================================================

def _sample_from_logits_sp(logits, top_k=0, top_p=1.0, temperature=1.0, logits_mask=None):
    """Sample from logits with top-k, top-p, temperature, and mask.
    
    IMPORTANT: Apply mask BEFORE softmax so banned tokens get 0 probability.
    Do NOT use .clamp(min=1e-12) as it gives banned tokens probability.
    """
    if logits_mask is not None:
        logits = apply_logits_mask(logits, logits_mask)

    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)
    # No clamp! Banned tokens already have -inf logits -> 0 softmax probability

    if top_k > 0 and top_k < probs.size(-1):
        topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
        probs = torch.zeros_like(probs).scatter_(-1, topk_idx, topk_vals)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumsum - sorted_probs > top_p
        sorted_probs[cutoff] = 0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)

    denom = probs.sum(dim=-1, keepdim=True)
    # If all zero (should not happen with proper mask), uniform over all
    all_zero = (denom <= 1e-30).squeeze(-1)
    if all_zero.any():
        probs[all_zero] = 1.0 / probs.size(-1)
        denom[all_zero] = 1.0
    probs = probs / denom.clamp(min=1e-30)

    return torch.multinomial(probs, 1).squeeze(-1)


# ============================================================
# Batch Decoding
# ============================================================

def _decode_loop(model, src, bos_id, eos_id, pad_id, unk_id, word_dict,
                 max_len, device, logits_mask, sample_fn=None, sample_kwargs=None):
    """
    Core decode loop shared by greedy and sampling.
    
    Enforces:
      - BOS banned after first token
      - PAD banned for unfinished sequences
      - UNK permanently banned
      - Force EOS at max_layers
    """
    batch_size = src.size(0)
    max_layers = max_len - 2  # excluding BOS and EOS
    
    ys = torch.ones(batch_size, 1, dtype=torch.long, device=device).fill_(bos_id)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    layer_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Build per-step mask that additionally bans BOS (and PAD for unfinished)
    base_mask = logits_mask.clone().to(device) if logits_mask is not None else None
    
    with torch.no_grad():
        for step in range(max_len - 1):
            tgt_mask = subsequent_mask(ys.size(1)).to(device)
            out = model(src, ys, None, tgt_mask)
            logits = model.generator.proj(out[:, -1])
            
            # Apply base mask + per-step bans
            step_mask = base_mask.clone() if base_mask is not None else torch.ones(
                logits.size(-1), dtype=torch.bool, device=device)
            
            # Ban BOS, PAD, UNK after first step (universally)
            step_mask[bos_id] = False
            step_mask[pad_id] = False
            step_mask[unk_id] = False
            
            # Force EOS at max layers (per-item)
            for b in range(batch_size):
                if not finished[b] and layer_counts[b] >= max_layers:
                    logits[b, :] = float('-inf')
                    logits[b, eos_id] = 0.0
            
            logits = apply_logits_mask(logits, step_mask)
            
            if sample_fn is not None:
                next_words = sample_fn(logits, **(sample_kwargs or {}))
            else:
                _, next_words = torch.max(logits, dim=1)
            
            next_words[finished] = pad_id
            
            # Update layer counts (count non-special tokens)
            is_layer = ~torch.isin(next_words, torch.tensor(
                [bos_id, eos_id, pad_id, unk_id], device=device))
            layer_counts += is_layer.long()
            
            ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)
            finished = finished | (next_words == eos_id)
            
            if finished.all():
                break
    
    return ys, finished


def batch_greedy_decode_sp(model, specs, word_dict, index_dict, max_len=22,
                           device=None, decode_batch_size=8, logits_mask=None):
    """Batch greedy decode for joint 284-dim spectra."""
    if device is None:
        device = next(model.parameters()).device

    bos_id = word_dict.get("BOS", BOS_ID)
    eos_id = word_dict.get("EOS", EOS_ID)
    pad_id = word_dict.get("PAD", PAD_ID)
    unk_id = word_dict.get("UNK", UNK_ID)

    n_specs = len(specs)
    all_designs = []

    for start in range(0, n_specs, decode_batch_size):
        end = min(start + decode_batch_size, n_specs)
        batch_specs = specs[start:end]
        src = torch.tensor(np.array(batch_specs), dtype=torch.float32).to(device)

        ys, finished = _decode_loop(
            model, src, bos_id, eos_id, pad_id, unk_id, word_dict,
            max_len, device, logits_mask, sample_fn=None,
        )

        for b in range(len(batch_specs)):
            tokens = []
            for t in range(1, ys.size(1)):
                tid = ys[b, t].item()
                if tid == eos_id:
                    break
                sym = index_dict.get(tid, "UNK")
                if sym not in ("UNK", "EOS", "BOS", "PAD"):
                    tokens.append(sym)
            all_designs.append(tokens)

    return all_designs


def batch_sampling_decode_sp(model, specs, word_dict, index_dict, max_len=22,
                              top_k=10, top_p=0.9, temperature=1.0,
                              device=None, decode_batch_size=8, logits_mask=None):
    """Batch top-k/p sampling decode for joint 284-dim spectra."""
    if device is None:
        device = next(model.parameters()).device

    bos_id = word_dict.get("BOS", BOS_ID)
    eos_id = word_dict.get("EOS", EOS_ID)
    pad_id = word_dict.get("PAD", PAD_ID)
    unk_id = word_dict.get("UNK", UNK_ID)

    n_specs = len(specs)
    all_designs = []

    sample_kwargs = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature}

    for start in range(0, n_specs, decode_batch_size):
        end = min(start + decode_batch_size, n_specs)
        batch_specs = specs[start:end]
        src = torch.tensor(np.array(batch_specs), dtype=torch.float32).to(device)

        def _sample_fn(logits, **kwargs):
            return _sample_from_logits_sp(logits, **kwargs)

        ys, finished = _decode_loop(
            model, src, bos_id, eos_id, pad_id, unk_id, word_dict,
            max_len, device, logits_mask, sample_fn=_sample_fn,
            sample_kwargs=sample_kwargs,
        )

        for b in range(len(batch_specs)):
            tokens = []
            for t in range(1, ys.size(1)):
                tid = ys[b, t].item()
                if tid == eos_id:
                    break
                sym = index_dict.get(tid, "UNK")
                if sym not in ("UNK", "EOS", "BOS", "PAD"):
                    tokens.append(sym)
            all_designs.append(tokens)

    return all_designs


# ============================================================
# Structure Parsing
# ============================================================

def parse_structure(tokens):
    """Parse token list into materials and thicknesses."""
    materials = []
    thicknesses = []
    for t in tokens:
        if '_' in t:
            mat, thick = t.rsplit('_', 1)
            try:
                thicknesses.append(int(thick))
                materials.append(mat)
            except ValueError:
                continue
    return materials, thicknesses


def structure_to_tuple(tokens):
    """Convert token list to hashable tuple."""
    return tuple(tokens)


def is_valid_structure(materials, thicknesses, allowed_materials=None,
                       word_dict=None):
    """
    Strict structure validation.

    Checks:
      - 1-20 layers
      - material and thickness lists same length
      - all materials in ALLOWED_MATERIALS
      - each Material_Thickness token in word_dict (if provided)
      - thickness in valid vocab range (if word_dict provided)
      - no special tokens (BOS/EOS/PAD/UNK)
    """
    if allowed_materials is None:
        allowed_materials = ALLOWED_MATERIALS

    if len(materials) == 0 or len(materials) > MAX_LAYERS:
        return False
    if len(materials) != len(thicknesses):
        return False

    for m, t in zip(materials, thicknesses):
        if m in ('BOS', 'EOS', 'PAD', 'UNK'):
            return False
        if m not in allowed_materials:
            return False
        token = f"{m}_{t}"
        if word_dict is not None and token not in word_dict:
            return False

    return True


# ============================================================
# Multi-Candidate Generation
# ============================================================

def generate_candidates_sp(model, spec_joint, word_dict, index_dict,
                            num_candidates=32, max_len=22,
                            top_k=10, top_p=0.9, temperature=1.0,
                            device=None, logits_mask=None):
    """
    Generate multiple candidates for a single 284-dim joint spectrum.

    Strategy: 1 greedy + (N-1) sampling, dedup by token tuple.

    Args:
        model: TransformerSP
        spec_joint: (284,) numpy array
        word_dict, index_dict: vocabulary
        num_candidates: number of candidates to generate
        max_len: max sequence length
        top_k, top_p, temperature: sampling params
        device: torch device
        logits_mask: optional (vocab_size,) bool mask

    Returns:
        list of dict: [{tokens, materials, thicknesses, n_layers}, ...]
    """
    if device is None:
        device = next(model.parameters()).device

    seen = set()
    candidates = []

    # 1 greedy
    greedy_tokens = batch_greedy_decode_sp(
        model, [spec_joint], word_dict, index_dict,
        max_len=max_len, device=device, decode_batch_size=1,
        logits_mask=logits_mask,
    )[0]

    tup = structure_to_tuple(greedy_tokens)
    if tup and tup not in seen:
        seen.add(tup)
        mats, thicks = parse_structure(greedy_tokens)
        if is_valid_structure(mats, thicks):
            candidates.append({
                'tokens': greedy_tokens,
                'materials': mats,
                'thicknesses': thicks,
                'n_layers': len(mats),
            })

    # Sampling
    specs_batch = np.tile(spec_joint.reshape(1, -1), (num_candidates - 1, 1))
    if len(specs_batch) > 0:
        sampled = batch_sampling_decode_sp(
            model, specs_batch, word_dict, index_dict,
            max_len=max_len, top_k=top_k, top_p=top_p, temperature=temperature,
            device=device, decode_batch_size=min(8, len(specs_batch)),
            logits_mask=logits_mask,
        )
        for tokens in sampled:
            tup = structure_to_tuple(tokens)
            if tup and tup not in seen:
                seen.add(tup)
                mats, thicks = parse_structure(tokens)
                if is_valid_structure(mats, thicks):
                    candidates.append({
                        'tokens': tokens,
                        'materials': mats,
                        'thicknesses': thicks,
                        'n_layers': len(mats),
                    })

    return candidates


# ============================================================
# TMM Re-Ranking
# ============================================================

def tmm_rerank_joint(candidates, spec_joint, nk_dict, wavelengths=None,
                      theta=THETA_DEG, substrate=SUBSTRATE,
                      substrate_thick=SUBSTRATE_THICK_NM,
                      high_T_weight_s=0.35, high_T_weight_p=0.35,
                      tail_weight_s=0.15, tail_weight_p=0.15,
                      objective="joint_error", high_T_objective_weight=1.0):
    """
    Re-rank candidates by joint s+p TMM agreement.

    For each candidate:
      1. Compute s-pol TMM -> sim_Rs, sim_Ts
      2. Compute p-pol TMM -> sim_Rp, sim_Tp
      3. E_s = mean(|sim_Rs - target_Rs| + |sim_Ts - target_Ts|) / 2
      4. E_p = mean(|sim_Rp - target_Rp| + |sim_Tp - target_Tp|) / 2
      5. E_joint = 0.5*E_s + 0.5*E_p
      6. High-T loss: penalize low transmittance

    Args:
        candidates: list of dict from generate_candidates_sp
        spec_joint: (284,) target spectrum [Rs, Ts, Rp, Tp]
        nk_dict: material nk dict
        wavelengths: wavelength array in µm (default: 0.4-1.1 step 0.01)
        theta: incidence angle in degrees
        substrate: substrate name
        substrate_thick: substrate thickness in nm
        high_T_weight_s/p: weight for mean(1-T)
        tail_weight_s/p: weight for p95(1-T)

    Returns:
        list sorted by the selected objective, enriched with metrics
    """
    from optogpt.core.datasets.sim import spectrum

    if wavelengths is None:
        wavelengths = np.arange(0.4, 1.101, 0.01)  # match sim.py default

    if objective not in {"joint_error", "high_transmission"}:
        raise ValueError(f"Unknown ranking objective: {objective}")
    spec_joint = np.asarray(spec_joint, dtype=np.float64)
    if spec_joint.shape != (SPEC_DIM,) or not np.all(np.isfinite(spec_joint)):
        raise ValueError(f"spec_joint must be finite shape ({SPEC_DIM},), got {spec_joint.shape}")

    target_Rs = spec_joint[0:71]
    target_Ts = spec_joint[71:142]
    target_Rp = spec_joint[142:213]
    target_Tp = spec_joint[213:284]

    ranked = []
    failures = {'tmm_error': 0, 'nan_result': 0, 'wrong_shape': 0, 'interp_error': 0}

    for cand in candidates:
        materials = cand['materials']
        thicknesses = cand['thicknesses']

        if len(materials) == 0:
            failures['tmm_error'] += 1
            continue

        try:
            sim_s = spectrum(
                materials, thicknesses, pol='s', theta=theta,
                wavelengths=wavelengths, nk_dict=nk_dict,
                substrate=substrate, substrate_thick=substrate_thick,
            )
            n_pts = len(sim_s) // 2
            sim_Rs = np.array(sim_s[:n_pts], dtype=np.float64)
            sim_Ts = np.array(sim_s[n_pts:], dtype=np.float64)

            sim_p = spectrum(
                materials, thicknesses, pol='p', theta=theta,
                wavelengths=wavelengths, nk_dict=nk_dict,
                substrate=substrate, substrate_thick=substrate_thick,
            )
            sim_Rp = np.array(sim_p[:n_pts], dtype=np.float64)
            sim_Tp = np.array(sim_p[n_pts:], dtype=np.float64)

            # Check for NaN
            if np.any(~np.isfinite(sim_Rs)) or np.any(~np.isfinite(sim_Ts)) or \
               np.any(~np.isfinite(sim_Rp)) or np.any(~np.isfinite(sim_Tp)):
                failures['nan_result'] += 1
                continue

            # Interpolate to 71-point grid if needed
            if n_pts != 71:
                try:
                    from scipy.interpolate import interp1d
                    wl_target = np.linspace(0.4, 1.1, 71)
                    wl_orig = np.linspace(0.4, 1.1, n_pts)
                    sim_Rs = interp1d(wl_orig, sim_Rs, kind='linear', fill_value='extrapolate')(wl_target)
                    sim_Ts = interp1d(wl_orig, sim_Ts, kind='linear', fill_value='extrapolate')(wl_target)
                    sim_Rp = interp1d(wl_orig, sim_Rp, kind='linear', fill_value='extrapolate')(wl_target)
                    sim_Tp = interp1d(wl_orig, sim_Tp, kind='linear', fill_value='extrapolate')(wl_target)
                except Exception:
                    failures['interp_error'] += 1
                    continue

        except Exception as e:
            failures['tmm_error'] += 1
            continue

        # Errors
        E_s = float(np.mean(np.abs(sim_Rs - target_Rs) + np.abs(sim_Ts - target_Ts)) / 2)
        E_p = float(np.mean(np.abs(sim_Rp - target_Rp) + np.abs(sim_Tp - target_Tp)) / 2)
        E_joint = 0.5 * E_s + 0.5 * E_p

        # Transmittance stats
        mean_Ts = float(np.mean(sim_Ts))
        mean_Tp = float(np.mean(sim_Tp))
        p05_Ts = float(np.percentile(sim_Ts, 5))
        p05_Tp = float(np.percentile(sim_Tp, 5))
        min_Ts = float(np.min(sim_Ts))
        min_Tp = float(np.min(sim_Tp))
        mean_unpolarized_T = float(np.mean((sim_Ts + sim_Tp) / 2))
        worst_pol_mean_T = min(mean_Ts, mean_Tp)

        # High-T loss (penalizes low transmittance)
        p95_1mTs = float(np.percentile(1.0 - sim_Ts, 95))
        p95_1mTp = float(np.percentile(1.0 - sim_Tp, 95))
        high_T_loss = (high_T_weight_s * np.mean(1.0 - sim_Ts) +
                       high_T_weight_p * np.mean(1.0 - sim_Tp) +
                       tail_weight_s * p95_1mTs +
                       tail_weight_p * p95_1mTp +
                       0.10 * (1.0 - worst_pol_mean_T) +
                       0.05 * (1.0 - min(min_Ts, min_Tp)))
        ranking_score = E_joint
        if objective == "high_transmission":
            ranking_score += high_T_objective_weight * float(high_T_loss)

        ranked.append({
            **cand,
            'sim_Rs': sim_Rs.tolist(),
            'sim_Ts': sim_Ts.tolist(),
            'sim_Rp': sim_Rp.tolist(),
            'sim_Tp': sim_Tp.tolist(),
            'E_s': E_s,
            'E_p': E_p,
            'E_joint': E_joint,
            'mean_Ts': mean_Ts,
            'mean_Tp': mean_Tp,
            'p05_Ts': p05_Ts,
            'p05_Tp': p05_Tp,
            'min_Ts': min_Ts,
            'min_Tp': min_Tp,
            'mean_unpolarized_T': mean_unpolarized_T,
            'worst_pol_mean_T': worst_pol_mean_T,
            'high_T_loss': high_T_loss,
            'ranking_objective': objective,
            'ranking_score': ranking_score,
        })

    ranked.sort(key=lambda x: (
        x['ranking_score'],
        -x['worst_pol_mean_T'],
        -min(x['p05_Ts'], x['p05_Tp']),
        -min(x['min_Ts'], x['min_Tp']),
        x['E_joint'],
    ))

    return ranked, failures
