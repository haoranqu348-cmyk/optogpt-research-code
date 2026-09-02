"""Vocabulary-compatible migration from a joint s+p model to two-sided output."""

import copy
from typing import Iterable

import torch
import torch.nn as nn

from joint_sp.model import make_model_SP
from .contract import SIDE_SEP


VOCAB_STATE_KEYS = {
    "tgt_embed.0.lut.weight",
    "generator.proj.weight",
    "generator.proj.bias",
}


def extend_vocabulary(source_word_dict, extra_layer_tokens: Iterable[str] = ()): 
    """Preserve every existing ID and append SIDE_SEP/new layers deterministically."""
    target = dict(source_word_dict)
    if SIDE_SEP not in target:
        target[SIDE_SEP] = len(target)
    for token in sorted(set(extra_layer_tokens)):
        if token not in target:
            target[token] = len(target)
    if sorted(target.values()) != list(range(len(target))):
        raise ValueError("Vocabulary IDs must be contiguous from zero")
    return target, {value: key for key, value in target.items()}


def migrate_joint_sp_model(source_model, source_word_dict, target_word_dict, model_config):
    """Build a larger-vocabulary model while preserving compatible weights exactly."""
    architecture = model_config.get(
        "architecture_version", getattr(source_model, "architecture_version", None)
    )
    if architecture is None:
        raise ValueError("Source checkpoint must declare architecture_version")
    target_model = make_model_SP(
        tgt_vocab=len(target_word_dict),
        N=int(model_config.get("N", model_config.get("layers", 6))),
        d_model=int(model_config.get("d_model", 1024)),
        d_ff=int(model_config.get("d_ff", 512)),
        h=int(model_config.get("head_num", 8)),
        dropout=float(model_config.get("dropout", 0.1)),
        architecture_version=architecture,
    ).to(next(source_model.parameters()).device)

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    for key, value in source_state.items():
        if key in VOCAB_STATE_KEYS:
            continue
        if key not in target_state or target_state[key].shape != value.shape:
            raise RuntimeError(f"Cannot inherit source parameter {key}: shape/key mismatch")
        target_state[key] = value.detach().clone()

    shared_tokens = sorted(set(source_word_dict).intersection(target_word_dict))
    for token in shared_tokens:
        source_index = int(source_word_dict[token])
        target_index = int(target_word_dict[token])
        target_state["tgt_embed.0.lut.weight"][target_index] = (
            source_state["tgt_embed.0.lut.weight"][source_index].detach().clone()
        )
        target_state["generator.proj.weight"][target_index] = (
            source_state["generator.proj.weight"][source_index].detach().clone()
        )
        target_state["generator.proj.bias"][target_index] = (
            source_state["generator.proj.bias"][source_index].detach().clone()
        )

    target_model.load_state_dict(target_state, strict=True)
    target_model.model_type = "double_sided_joint_sp"
    target_model.vocabulary_transfer = {
        "source_vocab_size": len(source_word_dict),
        "target_vocab_size": len(target_word_dict),
        "inherited_token_rows": len(shared_tokens),
        "new_tokens": sorted(set(target_word_dict) - set(source_word_dict)),
    }
    verify_inherited_weights(source_model, target_model, source_word_dict, target_word_dict)
    return target_model


def verify_inherited_weights(source_model, target_model, source_word_dict, target_word_dict):
    source_state, target_state = source_model.state_dict(), target_model.state_dict()
    for key, value in source_state.items():
        if key not in VOCAB_STATE_KEYS and not torch.equal(value, target_state[key]):
            raise RuntimeError(f"Inherited parameter changed during migration: {key}")
    for token, source_index in source_word_dict.items():
        target_index = target_word_dict[token]
        for key in VOCAB_STATE_KEYS:
            if key.endswith("bias"):
                source_row, target_row = source_state[key][source_index], target_state[key][target_index]
            else:
                source_row, target_row = source_state[key][source_index], target_state[key][target_index]
            if not torch.equal(source_row, target_row):
                raise RuntimeError(f"Inherited vocabulary row changed: {token} in {key}")
    return True


def load_double_sided_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    configs = checkpoint.get("configs", {})
    if configs.get("model_type") != "double_sided_joint_sp":
        raise ValueError("Checkpoint is not a double_sided_joint_sp model")
    word_dict = configs.get("struc_word_dict")
    index_dict = configs.get("struc_index_dict")
    if not word_dict or not index_dict or "SIDE_SEP" not in word_dict:
        raise ValueError("Double-sided checkpoint vocabulary is incomplete")
    model = make_model_SP(
        tgt_vocab=len(word_dict), N=int(configs["N"]),
        d_model=int(configs["d_model"]), d_ff=int(configs["d_ff"]),
        h=int(configs["head_num"]), dropout=float(configs["dropout"]),
        architecture_version=configs["architecture_version"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, word_dict, index_dict, configs, checkpoint
