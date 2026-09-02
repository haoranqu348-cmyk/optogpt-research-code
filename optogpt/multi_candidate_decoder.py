"""
Multi-Candidate Decoder for OptoGPT.

Supports: greedy, top-k sampling, top-p (nucleus) sampling, temperature.
All methods include validation constraints:
  - First token must be BOS
  - BOS cannot repeat mid-sequence
  - PAD/UNK never emitted as valid layers
  - EOS triggers immediate stop
  - Empty structures filtered
  - Materials must have nk data
  - Thicknesses must match checkpoint vocab
  - Max layer constraint enforced
  - Duplicate candidates deduplicated

For RTX 4070 Laptop (8GB), batch decoding with configurable decode_batch_size.
"""

import torch
import numpy as np
from torch.autograd import Variable
from core.models.transformer import subsequent_mask

PAD_ID = 1
UNK_ID = 0


def batch_greedy_decode(model, specs, word_dict, index_dict, max_len=22,
                         device=None, decode_batch_size=8):
    """
    Batch greedy decode for multiple spectra.
    Returns list of token lists.
    """
    if device is None:
        device = next(model.parameters()).device

    bos_id = word_dict.get("BOS", 2)
    eos_id = word_dict.get("EOS", 3)
    pad_id = word_dict.get("PAD", PAD_ID)

    n_specs = len(specs)
    all_designs = []

    for start in range(0, n_specs, decode_batch_size):
        end = min(start + decode_batch_size, n_specs)
        batch_specs = specs[start:end]
        batch_size = len(batch_specs)
        src = torch.tensor(np.array(batch_specs), dtype=torch.float32).to(device)
        ys = torch.ones(batch_size, 1, dtype=torch.long).fill_(bos_id).to(device)
        finished = torch.zeros(batch_size, dtype=torch.bool)

        with torch.no_grad():
            for _ in range(max_len - 1):
                trg_mask = subsequent_mask(ys.size(1)).to(device)
                out = model(src, ys, None, trg_mask)
                prob = model.generator(out[:, -1])
                _, next_words = torch.max(prob, dim=1)
                next_words[finished] = pad_id
                ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)
                finished = finished | (next_words == eos_id).cpu()
                if finished.all():
                    break

        for b in range(batch_size):
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


def _sample_from_logits(logits, top_k=0, top_p=1.0, temperature=1.0):
    """Sample a token from logits with top-k, top-p, and temperature."""
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)

    if top_k > 0 and top_k < probs.size(-1):
        topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
        probs = torch.zeros_like(probs).scatter_(-1, topk_idx, topk_vals)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)

    # Renormalize
    denom = probs.sum(dim=-1, keepdim=True)
    if (denom > 0).all():
        probs = probs / denom
    else:
        # If all zero, uniform over non-special tokens
        probs = torch.ones_like(probs) / probs.size(-1)

    return torch.multinomial(probs, 1).squeeze(-1)


def batch_sampling_decode(model, specs, word_dict, index_dict, max_len=22,
                           top_k=10, top_p=0.9, temperature=1.0,
                           device=None, decode_batch_size=8):
    """
    Batch top-k/p sampling decode.
    Returns list of token lists.
    """
    if device is None:
        device = next(model.parameters()).device

    bos_id = word_dict.get("BOS", 2)
    eos_id = word_dict.get("EOS", 3)
    pad_id = word_dict.get("PAD", PAD_ID)

    n_specs = len(specs)
    all_designs = []

    for start in range(0, n_specs, decode_batch_size):
        end = min(start + decode_batch_size, n_specs)
        batch_specs = specs[start:end]
        batch_size = len(batch_specs)
        src = torch.tensor(np.array(batch_specs), dtype=torch.float32).to(device)
        ys = torch.ones(batch_size, 1, dtype=torch.long).fill_(bos_id).to(device)
        finished = torch.zeros(batch_size, dtype=torch.bool)

        with torch.no_grad():
            for _ in range(max_len - 1):
                trg_mask = subsequent_mask(ys.size(1)).to(device)
                out = model(src, ys, None, trg_mask)
                # Apply generator projection to get vocab logits
                logits = model.generator.proj(out[:, -1])  # raw logits
                next_words = _sample_from_logits(logits, top_k, top_p, temperature)
                next_words[finished] = pad_id
                ys = torch.cat([ys, next_words.unsqueeze(1).to(device)], dim=1)
                next_cpu = next_words.cpu()
                finished = finished | (next_cpu == eos_id)
                if finished.all():
                    break

        for b in range(batch_size):
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


def parse_structure(tokens):
    """Parse token list like ['SiO2_50', 'TiO2_100'] into materials and thicknesses."""
    materials = []
    thicknesses = []
    for tok in tokens:
        if "_" not in tok:
            continue
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


def is_valid_structure(materials, thicknesses, max_layers=20, nk_dict=None,
                        word_dict=None, index_dict=None):
    """Check structure validity with all constraints."""
    if len(materials) == 0 or len(materials) != len(thicknesses):
        return False, "empty or mismatch"
    if len(materials) > max_layers:
        return False, f"too many layers ({len(materials)} > {max_layers})"
    for m in materials:
        if m in ("EOS", "BOS", "UNK", "PAD", ""):
            return False, f"special token as material: {m}"
    for t in thicknesses:
        if t < 10 or t > 300:
            return False, f"invalid thickness: {t}"
    # Check nk data availability
    if nk_dict is not None:
        for m in materials:
            if m not in nk_dict:
                return False, f"no nk data for: {m}"
    # Check vocabulary consistency
    if word_dict is not None:
        for m, t in zip(materials, thicknesses):
            token = f"{m}_{int(t)}"
            if token not in word_dict:
                return False, f"token not in vocab: {token}"
    return True, "ok"


def structure_to_tuple(tokens):
    """Convert token list to hashable tuple for dedup."""
    return tuple(t for t in tokens if t not in ("BOS", "EOS", "PAD", "UNK", ""))


def generate_candidates(model, spec_target, word_dict, index_dict,
                         num_candidates=32, max_len=22, max_layers=20,
                         top_k=10, top_p=0.9, temperature=1.0,
                         nk_dict=None, device=None, decode_batch_size=8,
                         rng=None):
    """
    Generate multiple candidates for a single target spectrum.

    Uses batch sampling for efficiency. Returns list of valid candidate dicts,
    sorted by... (TMM simulation not done here - call tmm_rerank after).

    Returns list of dicts with keys: tokens, materials, thicknesses, n_layers.
    """
    if device is None:
        device = next(model.parameters()).device
    if rng is None:
        rng = np.random.RandomState()

    # Also generate greedy candidate
    greedy_tokens = batch_greedy_decode(
        model, [spec_target], word_dict, index_dict,
        max_len=max_len, device=device, decode_batch_size=1)[0]

    # Generate sampling candidates in batch
    specs_batch = [spec_target] * num_candidates
    sampling_tokens = batch_sampling_decode(
        model, specs_batch, word_dict, index_dict,
        max_len=max_len, top_k=top_k, top_p=top_p,
        temperature=temperature, device=device,
        decode_batch_size=min(decode_batch_size, num_candidates))

    # Collect all candidates
    all_tokens = [greedy_tokens] + sampling_tokens

    seen = set()
    valid_candidates = []
    for tokens in all_tokens:
        # Deduplicate
        key = structure_to_tuple(tokens)
        if key in seen:
            continue
        seen.add(key)

        mats, thick = parse_structure(tokens)
        valid, reason = is_valid_structure(mats, thick, max_layers, nk_dict, word_dict, index_dict)
        if not valid:
            continue

        valid_candidates.append({
            "tokens": tokens,
            "materials": mats,
            "thicknesses": thick,
            "n_layers": len(mats),
            "total_thickness": sum(thick),
        })

    return valid_candidates


def generate_candidates_batch(model, spec_targets, word_dict, index_dict,
                               num_candidates=32, max_len=22, max_layers=20,
                               top_k=10, top_p=0.9, temperature=1.0,
                               nk_dict=None, device=None, decode_batch_size=8,
                               seed=42):
    """
    Generate candidates for multiple target spectra.
    Returns list of lists (one per target) of candidate dicts.
    """
    results = []
    rng = np.random.RandomState(seed)
    for i, spec in enumerate(spec_targets):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        candidates = generate_candidates(
            model, spec, word_dict, index_dict,
            num_candidates=num_candidates, max_len=max_len,
            max_layers=max_layers, top_k=top_k, top_p=top_p,
            temperature=temperature, nk_dict=nk_dict,
            device=device, decode_batch_size=decode_batch_size,
            rng=rng)
        results.append(candidates)
    return results


# ============================================================
# TMM Re-ranking (added for Phase 2)
# ============================================================

def tmm_rerank(candidates, spec_target, nk_dict, wavelengths=None, n_wl=71,
               pol="s", theta=60, substrate="Glass_Substrate", substrate_thick=500000):
    """Re-rank candidates by TMM simulation Total MAE."""
    from core.datasets.sim import spectrum

    if wavelengths is None:
        wavelengths = np.arange(0.4, 1.1 + 1e-3, 0.01)

    R_target = np.array(spec_target[:n_wl])
    T_target = np.array(spec_target[n_wl:])

    ranked = []
    for cand in candidates:
        try:
            result = spectrum(
                materials=cand["materials"], thickness=cand["thicknesses"],
                pol=pol, theta=theta, wavelengths=wavelengths,
                nk_dict=nk_dict, substrate=substrate,
                substrate_thick=substrate_thick)
            half = len(result) // 2
            R_sim = np.array(result[:half], dtype=np.float64)
            T_sim = np.array(result[half:], dtype=np.float64)
            if np.any(np.isnan(R_sim)) or np.any(np.isnan(T_sim)):
                cand["tmm_error"] = "NaN"
                cand["tmm_success"] = False
                ranked.append(cand)
                continue
            if np.any(np.isinf(R_sim)) or np.any(np.isinf(T_sim)):
                cand["tmm_error"] = "Inf"
                cand["tmm_success"] = False
                ranked.append(cand)
                continue

            mae_R = float(np.mean(np.abs(R_sim - R_target)))
            mae_T = float(np.mean(np.abs(T_sim - T_target)))
            total_mae = float(np.mean(np.abs(
                np.concatenate([R_sim, T_sim]) -
                np.concatenate([R_target, T_target])
            )))

            cand.update({
                "R_sim": R_sim.tolist(),
                "T_sim": T_sim.tolist(),
                "tmm_success": True,
                "mae_R": mae_R,
                "mae_T": mae_T,
                "total_mae": total_mae,
                "max_R_err": float(np.max(np.abs(R_sim - R_target))),
                "max_T_err": float(np.max(np.abs(T_sim - T_target))),
                "avg_absorption": float(np.mean(1 - R_sim - T_sim)),
            })
        except Exception as e:
            cand["tmm_error"] = str(e)
            cand["tmm_success"] = False
        ranked.append(cand)

    ranked.sort(key=lambda c: c.get("total_mae", 1e9))
    return ranked
