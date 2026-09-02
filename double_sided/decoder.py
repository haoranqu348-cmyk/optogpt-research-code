"""Strict SIDE_SEP-aware constrained decoding."""

from dataclasses import dataclass

import numpy as np
import torch

from optogpt.core.models.transformer import subsequent_mask
from .contract import DoubleSidedStructure, SIDE_SEP


@dataclass
class DecodeState:
    front_layers: int = 0
    back_layers: int = 0
    separator_seen: bool = False
    finished: bool = False


def layer_token_ids(word_dict, allowed_materials):
    allowed = set(allowed_materials)
    ids = []
    for token, index in word_dict.items():
        if token in ("UNK", "PAD", "BOS", "EOS", SIDE_SEP) or "_" not in token:
            continue
        material, thickness = token.rsplit("_", 1)
        try:
            value = float(thickness)
        except ValueError:
            continue
        if material in allowed and np.isfinite(value) and value > 0:
            ids.append(int(index))
    if not ids:
        raise ValueError("No valid physical layer tokens in vocabulary")
    return sorted(ids)


def allowed_next_ids(state, word_dict, physical_ids, max_layers_per_side):
    if state.finished:
        return [word_dict["PAD"]]
    if not state.separator_seen:
        if state.front_layers >= max_layers_per_side:
            return [word_dict[SIDE_SEP]]
        ids = list(physical_ids)
        if state.front_layers >= 1:
            ids.append(word_dict[SIDE_SEP])
        return ids
    if state.back_layers >= max_layers_per_side:
        return [word_dict["EOS"]]
    ids = list(physical_ids)
    if state.back_layers >= 1:
        ids.append(word_dict["EOS"])
    return ids


def advance_state(state, token_id, word_dict, physical_id_set):
    if state.finished:
        if token_id != word_dict["PAD"]:
            raise ValueError("Finished sequences may emit PAD only")
        return
    if token_id == word_dict[SIDE_SEP]:
        if state.separator_seen or state.front_layers == 0:
            raise ValueError("Illegal SIDE_SEP transition")
        state.separator_seen = True
    elif token_id == word_dict["EOS"]:
        if not state.separator_seen or state.back_layers == 0:
            raise ValueError("Illegal EOS transition")
        state.finished = True
    elif token_id in physical_id_set:
        if state.separator_seen:
            state.back_layers += 1
        else:
            state.front_layers += 1
    else:
        raise ValueError(f"Illegal decoded token id: {token_id}")


def constrained_decode(model, spectra, word_dict, index_dict, allowed_materials,
                       max_layers_per_side, sample_fn=None, device=None):
    """Decode complete valid structures with a finite per-side technical limit."""
    required = {"UNK", "PAD", "BOS", "EOS", SIDE_SEP}
    if not required.issubset(word_dict):
        raise ValueError(f"Vocabulary missing special tokens: {sorted(required - set(word_dict))}")
    if max_layers_per_side < 1:
        raise ValueError("max_layers_per_side must be positive")
    device = device or next(model.parameters()).device
    source = torch.as_tensor(np.asarray(spectra), dtype=torch.float32, device=device)
    if source.ndim == 1:
        source = source.unsqueeze(0)
    physical_ids = layer_token_ids(word_dict, allowed_materials)
    physical_set = set(physical_ids)
    states = [DecodeState() for _ in range(source.size(0))]
    generated = torch.full(
        (source.size(0), 1), int(word_dict["BOS"]), dtype=torch.long, device=device
    )
    technical_max_tokens = 2 * max_layers_per_side + 3

    with torch.no_grad():
        for _ in range(technical_max_tokens - 1):
            mask = subsequent_mask(generated.size(1)).to(device)
            hidden = model(source, generated, None, mask)
            logits = model.generator.proj(hidden[:, -1]).clone()
            for row, state in enumerate(states):
                allowed = allowed_next_ids(
                    state, word_dict, physical_ids, max_layers_per_side
                )
                constrained = torch.full_like(logits[row], float("-inf"))
                constrained[allowed] = logits[row, allowed]
                logits[row] = constrained
            next_ids = (sample_fn(logits) if sample_fn is not None
                        else torch.argmax(logits, dim=-1))
            for state, token_id in zip(states, next_ids.tolist()):
                advance_state(state, token_id, word_dict, physical_set)
            generated = torch.cat([generated, next_ids[:, None]], dim=1)
            if all(state.finished for state in states):
                break
    if not all(state.finished for state in states):
        raise RuntimeError("Technical maximum reached without complete EOS termination")

    structures = []
    for row in generated.tolist():
        tokens = []
        for token_id in row:
            token = index_dict[int(token_id)]
            if token == "PAD":
                break
            tokens.append(token)
        structures.append(DoubleSidedStructure.from_tokens(
            tokens, allowed_materials, max_layers_per_side
        ))
    return structures
