"""
joint_sp/model.py — TransformerSP: Joint s+p polarization inverse-design model.

Based on core/models/transformer.py base classes.
TransformerSP takes a 284-dim spectrum [Rs, Ts, Rp, Tp] and decodes
a multilayer thin-film structure that simultaneously satisfies both polarizations.

Key components:
  - fc_s: FullyConnectedLayers(142 -> d_model) for s-pol branch
  - fc_p: FullyConnectedLayers(142 -> d_model) for p-pol branch
  - fusion: merges s and p memory into unified decoder memory
  - decoder, tgt_embed, generator: inherited from OptoGPT Decoder architecture

Architecture versions:
  - optogpt_legacy_v1: Original 142-dim optogpt.pt checkpoint semantics
    FC: out = fc2(norm(fc1(x)))         (no ReLU, no extra dropout)
    FFN: out = w2(dropout(w1(x)))        (no ReLU)
  - joint_sp_legacy_v1: Joint s+p model with legacy FC/FFN semantics (default)
  - joint_sp_relu_v0: Joint s+p model using current core ReLU semantics
    (only for loading old experiments trained with ReLU implementation)
"""

import os
import sys
import copy
import random
import hashlib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Path setup
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from optogpt.core.models.transformer import (
    Embeddings, PositionalEncoding, Decoder, DecoderLayer,
    Generator, MultiHeadedAttention, PositionwiseFeedForward,
    LayerNorm, SublayerConnection, subsequent_mask,
    FullyConnectedLayers,
)
from joint_sp.constants import (
    BRANCH_DIM, SPEC_DIM, ALLOWED_MATERIALS, THETA_DEG,
)
from joint_sp.io_utils import atomic_torch_save

# ============================================================
# Architecture version constants
# ============================================================
ARCH_OPTOGPT_LEGACY_V1 = "optogpt_legacy_v1"
ARCH_JOINT_SP_LEGACY_V1 = "joint_sp_legacy_v1"
ARCH_JOINT_SP_RELU_V0 = "joint_sp_relu_v0"
_KNOWN_ARCHITECTURES = {
    ARCH_OPTOGPT_LEGACY_V1,
    ARCH_JOINT_SP_LEGACY_V1,
    ARCH_JOINT_SP_RELU_V0,
}

# Known SHA-256 of optogpt.pt (the 142-dim pretrained checkpoint)
_OPTOGPT_PT_SHA256 = (
    "a7677602ae8dae60dababde9bd3981ad16be61430a22e797ba359b1d01921d85"
)

# Expected fusion keys in joint_sp model
_EXPECTED_FUSION_KEYS = {
    "fusion.0.weight", "fusion.0.bias",
    "fusion.2.a_2", "fusion.2.b_2",
}


def _sha256_file_chunked(path, chunk_size=8 * 1024 * 1024):
    """Compute SHA-256 of a file in chunks (safe for large files)."""
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# Legacy FC/FFN classes — OptoGPT checkpoint-era semantics
# ============================================================

class OptoGPTLegacyFullyConnected(FullyConnectedLayers):
    """OptoGPT checkpoint-era FC semantics.

    Historical forward (no ReLU, no extra dropout):
        return self.fc2(self.norm(self.fc1(x)))

    This matches the computation used when optogpt.pt was trained.
    State-dict keys and tensor shapes are identical to FullyConnectedLayers.
    """

    def forward(self, x):
        return self.fc2(self.norm(self.fc1(x)))


class OptoGPTLegacyFeedForward(PositionwiseFeedForward):
    """OptoGPT checkpoint-era FFN semantics.

    Historical forward (no ReLU):
        return self.w_2(self.dropout(self.w_1(x)))

    State-dict keys and tensor shapes are identical to PositionwiseFeedForward.
    """

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x)))


class TransformerSP(nn.Module):
    """
    Joint s+p Transformer for inverse design.

    Takes a 284-dim spectrum [Rs(71), Ts(71), Rp(71), Tp(71)] and
    decodes a multilayer structure.

    Architecture:
        spec [B, 284] -> split into s [B, 142] and p [B, 142]
        fc_s(s_spec) -> memory_s [B, 1, d_model]
        fc_p(p_spec) -> memory_p [B, 1, d_model]
        fusion(cat(memory_s, memory_p)) -> memory [B, 1, d_model]
        decoder(tgt_embed(tgt), memory, ...) -> hidden [B, L, d_model]
        generator(hidden) -> log_probs [B, L, vocab]
    """

    def __init__(self, fc_s, fc_p, fusion, decoder, tgt_embed, generator):
        super(TransformerSP, self).__init__()
        self.fc_s = fc_s
        self.fc_p = fc_p
        self.fusion = fusion
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Args:
            src: spectrum tensor, shape [B, 284] or [B, 1, 284]
            tgt: target token sequence [B, L]
            src_mask: (unused, kept for API compatibility)
            tgt_mask: decoder causal mask [B, L, L]
        Returns:
            decoder output [B, L, d_model]
        """
        # Input shape normalization
        if src.dim() == 2:
            src = src.unsqueeze(1)  # [B, 284] -> [B, 1, 284]
        elif src.dim() == 3:
            pass  # [B, 1, 284]
        else:
            raise ValueError(f"src must be 2D or 3D, got shape {src.shape}")

        if src.size(-1) != SPEC_DIM:
            raise ValueError(
                f"TransformerSP expects {SPEC_DIM}-dim input (Rs+Ts+Rp+Tp), "
                f"got {src.size(-1)}-dim. Check data format."
            )

        # Split s and p branches
        spec_s = src[..., :BRANCH_DIM]   # [B, 1, 142]
        spec_p = src[..., BRANCH_DIM:]    # [B, 1, 142]

        # Encode each branch
        memory_s = self.fc_s(spec_s)      # [B, 1, d_model]
        memory_p = self.fc_p(spec_p)      # [B, 1, d_model]

        # Fusion
        memory = self.fusion(torch.cat([memory_s, memory_p], dim=-1))  # [B, 1, d_model]

        # Decode
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model_SP(src_vocab=SPEC_DIM, tgt_vocab=904, N=6, d_model=1024,
                  d_ff=512, h=8, dropout=0.1,
                  architecture_version=ARCH_JOINT_SP_LEGACY_V1):
    """
    Factory function for TransformerSP, analogous to make_model_I.

    Args:
        src_vocab: spectrum dimension (284 for joint s+p)
        tgt_vocab: vocabulary size
        N: number of decoder layers
        d_model: model dimension
        d_ff: feed-forward hidden dimension
        h: number of attention heads
        dropout: dropout rate
        architecture_version: one of ARCH_OPTOGPT_LEGACY_V1,
            ARCH_JOINT_SP_LEGACY_V1, ARCH_JOINT_SP_RELU_V0.
            Default is joint_sp_legacy_v1 (historical semantics).

    Returns:
        TransformerSP model
    """
    if architecture_version not in _KNOWN_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture_version '{architecture_version}'. "
            f"Known: {sorted(_KNOWN_ARCHITECTURES)}"
        )

    c = copy.deepcopy

    # Select FC and FFN classes based on architecture version
    if architecture_version in (ARCH_JOINT_SP_LEGACY_V1, ARCH_OPTOGPT_LEGACY_V1):
        FCClass = OptoGPTLegacyFullyConnected
        FFNClass = OptoGPTLegacyFeedForward
    elif architecture_version == ARCH_JOINT_SP_RELU_V0:
        FCClass = FullyConnectedLayers
        FFNClass = PositionwiseFeedForward
    else:
        raise ValueError(f"Unhandled architecture_version: {architecture_version}")

    # Branch encoders: 142 -> d_model
    fc_s = FCClass(BRANCH_DIM, d_model, dropout)
    fc_p = FCClass(BRANCH_DIM, d_model, dropout)

    # Fusion layer: concatenated s+p memory -> unified memory
    fusion = nn.Sequential(
        nn.Linear(2 * d_model, d_model),
        nn.GELU(),
        LayerNorm(d_model),
        nn.Dropout(dropout),
    )

    # Decoder
    attn = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = FFNClass(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    decoder = Decoder(
        DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N
    )

    # Target embedding + generator
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
    generator = Generator(d_model, tgt_vocab)

    model = TransformerSP(fc_s, fc_p, fusion, decoder, tgt_embed, generator)

    # Xavier init for new params (pretrained ones will be overwritten)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Store build-time config on model for save-time validation
    model.architecture_version = architecture_version
    model._build_N = N
    model._build_d_model = d_model
    model._build_d_ff = d_ff
    model._build_h = h
    model._build_dropout = dropout
    model._build_tgt_vocab = tgt_vocab

    return model


def _get_cfg(configs, key, default=None):
    """Handle both dict and SimpleNamespace configs."""
    if isinstance(configs, dict):
        return configs.get(key, default)
    return getattr(configs, key, default)


def _resolve_joint_architecture(saved_arch, architecture_override, context=""):
    """
    Resolve architecture version with strict safety rules.

    Rules:
      1. If checkpoint HAS architecture_version:
         - Validate it's a known version
         - No override: use saved_arch
         - Same override: accept
         - Different override: RAISE ValueError (cannot replace saved version)
      2. If checkpoint HAS NO architecture_version:
         - No override: RAISE ValueError
         - With override: validate, warn, use override

    Args:
        saved_arch: architecture_version from checkpoint configs (or None)
        architecture_override: user-specified override (or None)
        context: human-readable context for error messages

    Returns:
        effective architecture_version string
    """
    if saved_arch is not None:
        # Checkpoint has explicit version
        if saved_arch not in _KNOWN_ARCHITECTURES:
            raise ValueError(
                f"{context}: checkpoint has unknown architecture_version "
                f"'{saved_arch}'. Known: {sorted(_KNOWN_ARCHITECTURES)}"
            )
        if architecture_override is None:
            return saved_arch
        if architecture_override == saved_arch:
            print(f"  (architecture_override matches saved '{saved_arch}')")
            return saved_arch
        # Override conflicts with saved version — REJECT
        raise ValueError(
            f"{context}: checkpoint already has architecture_version "
            f"'{saved_arch}', cannot override to '{architecture_override}'. "
            f"State-dict shapes are identical for both semantics; "
            f"loading with wrong version would produce silently wrong results."
        )
    else:
        # Checkpoint has no version
        if architecture_override is None:
            raise ValueError(
                f"{context}: checkpoint has no architecture_version. "
                f"Unversioned checkpoints require explicit "
                f"architecture_override='joint_sp_legacy_v1' or "
                f"'joint_sp_relu_v0'."
            )
        if architecture_override not in _KNOWN_ARCHITECTURES:
            raise ValueError(
                f"{context}: unknown architecture_override "
                f"'{architecture_override}'. Known: {sorted(_KNOWN_ARCHITECTURES)}"
            )
        print(f"  ⚠ WARNING: using architecture_override='{architecture_override}' "
              f"for unversioned checkpoint")
        return architecture_override


# ============================================================
# Unified loading function (single entry for all inference consumers)
# ============================================================

def load_joint_sp_checkpoint(ckpt_path, device='cpu',
                              architecture_override=None):
    """
    Unified loader for joint_sp checkpoints.

    All inference/validation/deployment consumers MUST use this function.
    Direct torch.load + make_model_SP + load_state_dict is forbidden
    because it bypasses architecture version checks.

    Args:
        ckpt_path: path to a joint_sp checkpoint (.pt)
        device: torch device
        architecture_override: if set, force a specific architecture_version.
            WARNING: only use for loading old unversioned experiments.

    Returns:
        (model, word_dict, index_dict, configs)
    """
    print(f"[load_joint_sp_checkpoint] Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    configs = ckpt.get('configs', {})
    if not configs:
        raise ValueError("Checkpoint missing 'configs' key")

    word_dict = _get_cfg(configs, 'struc_word_dict')
    index_dict = _get_cfg(configs, 'struc_index_dict')
    if word_dict is None or index_dict is None:
        raise ValueError(
            "Checkpoint missing struc_word_dict/struc_index_dict. "
            "Do NOT mix vocab from another checkpoint."
        )

    tgt_vocab = len(word_dict)
    N = _get_cfg(configs, 'N') or _get_cfg(configs, 'layers', 6)
    d_model = _get_cfg(configs, 'd_model', 1024)
    d_ff = _get_cfg(configs, 'd_ff', 512)
    h = _get_cfg(configs, 'head_num', 8)
    dropout = _get_cfg(configs, 'dropout', 0.1)

    # Determine architecture_version
    saved_arch = _get_cfg(configs, 'architecture_version', None)
    effective_arch = _resolve_joint_architecture(
        saved_arch, architecture_override,
        context=f"load_joint_sp_checkpoint({ckpt_path})",
    )

    print(f"  Config: layers={N}, d_model={d_model}, d_ff={d_ff}, "
          f"heads={h}, vocab={tgt_vocab}, arch={effective_arch}")

    model = make_model_SP(
        src_vocab=SPEC_DIM, tgt_vocab=tgt_vocab,
        N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout,
        architecture_version=effective_arch,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    print(f"  ✓ Loaded {len(ckpt['model_state_dict'])} keys (strict)")

    # Ensure configs dict has architecture_version
    if isinstance(configs, dict):
        enriched = dict(configs)
    else:
        enriched = vars(configs).copy()
    enriched['architecture_version'] = effective_arch
    enriched['model_type'] = 'joint_sp'

    return model, word_dict, index_dict, enriched


# ============================================================
# Pretrained weight mapping (optogpt.pt -> joint_sp)
# ============================================================


def load_sp_from_pretrained(pretrained_path, device='cpu',
                            architecture_override=None):
    """
    Load TransformerSP from optogpt.pt or joint_sp checkpoint.

    Detection:
      - If model_type='joint_sp', do direct restore with version check.
      - If known optogpt.pt SHA-256 match, use legacy semantics.
      - Otherwise, error unless architecture_override is provided.

    Handles legacy checkpoints saved with 'core.*' module paths by
    temporarily registering a compatibility alias.

    Returns:
        (model, word_dict, index_dict, configs, is_joint_sp)
    """
    print(f"[load_sp_from_pretrained] Loading: {pretrained_path}")

    # ---- Legacy checkpoint compatibility ----
    _legacy_alias_needed = 'core' not in sys.modules
    if _legacy_alias_needed:
        import optogpt.core as _core_pkg
        sys.modules['core'] = _core_pkg
        for sub in ['models', 'datasets', 'trains']:
            sub_name = f'core.{sub}'
            if sub_name not in sys.modules:
                try:
                    sys.modules[sub_name] = __import__(f'optogpt.core.{sub}', fromlist=[sub])
                except Exception:
                    pass

    try:
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    finally:
        if _legacy_alias_needed:
            for mod_name in list(sys.modules.keys()):
                if mod_name == 'core' or mod_name.startswith('core.'):
                    del sys.modules[mod_name]

    configs = ckpt['configs']

    word_dict = _get_cfg(configs, 'struc_word_dict')
    index_dict = _get_cfg(configs, 'struc_index_dict')
    if word_dict is None or index_dict is None:
        raise ValueError("Checkpoint missing struc_word_dict/struc_index_dict")

    tgt_vocab = len(word_dict)
    N = _get_cfg(configs, 'N') or _get_cfg(configs, 'layers', 6)
    d_model = _get_cfg(configs, 'd_model', 1024)
    d_ff = _get_cfg(configs, 'd_ff', 512)
    h = _get_cfg(configs, 'head_num', 8)
    dropout = _get_cfg(configs, 'dropout', 0.1)
    model_type = _get_cfg(configs, 'model_type', '')
    is_joint_sp = (model_type == 'joint_sp')
    saved_arch = _get_cfg(configs, 'architecture_version', None)

    print(f"  Config: layers={N}, d_model={d_model}, d_ff={d_ff}, "
          f"heads={h}, vocab={tgt_vocab}, type={model_type or 'pretrained'}")

    # ---- Branch: joint_sp checkpoint ----
    if is_joint_sp:
        effective_arch = _resolve_joint_architecture(
            saved_arch, architecture_override,
            context=f"load_sp_from_pretrained({pretrained_path})",
        )

        model = make_model_SP(
            src_vocab=SPEC_DIM, tgt_vocab=tgt_vocab,
            N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout,
            architecture_version=effective_arch,
        ).to(device)

        pretrained_sd = ckpt['model_state_dict']
        model.load_state_dict(pretrained_sd, strict=True)
        print(f"  ✓ Direct restore ({len(pretrained_sd)} keys), arch={effective_arch}")

        if isinstance(configs, dict):
            enriched = dict(configs)
        else:
            enriched = vars(configs).copy()
        enriched['model_type'] = 'joint_sp'
        enriched['architecture_version'] = effective_arch
        enriched['spec_dim'] = SPEC_DIM
        enriched['branch_dim'] = BRANCH_DIM
        return model, word_dict, index_dict, enriched, True

    # ---- Branch: pretrained (optogpt.pt) ----
    # Verify checkpoint identity via SHA-256
    ckpt_full_hash = _sha256_file_chunked(pretrained_path)
    is_known_optogpt = (ckpt_full_hash == _OPTOGPT_PT_SHA256)

    if is_known_optogpt:
        print(f"  ✓ Known optogpt.pt (SHA-256 match)")
        architecture_version = ARCH_JOINT_SP_LEGACY_V1
    elif architecture_override:
        architecture_version = architecture_override
        print(f"  ⚠ Unrecognized checkpoint SHA-256: {ckpt_full_hash[:16]}...")
        print(f"  ⚠ Using architecture_override='{architecture_override}'")
    else:
        raise ValueError(
            f"Unrecognized pretrained checkpoint SHA-256: {ckpt_full_hash}. "
            f"Expected: {_OPTOGPT_PT_SHA256}. "
            f"If this is a legitimate variant, use architecture_override=..."
        )

    # Build model with correct architecture
    model = make_model_SP(
        src_vocab=SPEC_DIM, tgt_vocab=tgt_vocab,
        N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout,
        architecture_version=architecture_version,
    ).to(device)

    pretrained_sd = ckpt['model_state_dict']
    source_keys = set(pretrained_sd.keys())
    new_sd = model.state_dict()
    target_keys = set(new_sd.keys())

    print(f"\n[Weight Transfer Report]")
    print(f"  Source keys: {len(source_keys)}")
    print(f"  Target keys: {len(target_keys)}")

    loaded_keys = []
    missing_keys = []
    new_keys = []

    # Map fc -> fc_s and fc_p (exactly 6 source keys)
    _EXPECTED_FC_SOURCE_KEYS = {
        'fc.fc1.weight', 'fc.fc1.bias', 'fc.fc2.weight', 'fc.fc2.bias',
        'fc.norm.a_2', 'fc.norm.b_2',
    }

    # Strict: all fc.* source keys must be exactly the expected 6
    actual_fc_source = {k for k in source_keys if k.startswith('fc.')}
    if actual_fc_source != _EXPECTED_FC_SOURCE_KEYS:
        missing = _EXPECTED_FC_SOURCE_KEYS - actual_fc_source
        unexpected = actual_fc_source - _EXPECTED_FC_SOURCE_KEYS
        parts = []
        if missing:
            parts.append(f"Missing FC source keys: {sorted(missing)}")
        if unexpected:
            parts.append(f"Unexpected FC source keys: {sorted(unexpected)}")
        raise RuntimeError("FC key set mismatch: " + "; ".join(parts))

    for src_key in sorted(_EXPECTED_FC_SOURCE_KEYS):
        for prefix in ['fc_s', 'fc_p']:
            tgt_key = src_key.replace('fc.', f'{prefix}.')
            if tgt_key in target_keys:
                if pretrained_sd[src_key].shape == new_sd[tgt_key].shape:
                    new_sd[tgt_key] = pretrained_sd[src_key].clone()
                    loaded_keys.append(tgt_key)
                else:
                    raise RuntimeError(
                        f"Shape mismatch: {src_key} {pretrained_sd[src_key].shape} "
                        f"→ {tgt_key} {new_sd[tgt_key].shape}"
                    )
            else:
                raise RuntimeError(f"Target key not found in model: {tgt_key}")

    # Map decoder, tgt_embed, generator directly
    for key in source_keys:
        if key.startswith('fc.'):
            continue
        if key in target_keys:
            if pretrained_sd[key].shape == new_sd[key].shape:
                new_sd[key] = pretrained_sd[key].clone()
                loaded_keys.append(key)
            else:
                raise RuntimeError(
                    f"Shape mismatch: {key} {pretrained_sd[key].shape} "
                    f"vs {new_sd[key].shape}"
                )
        else:
            # Key in source but not in target — must error
            raise RuntimeError(f"Unexpected source key not in target model: {key}")

    # Identify new (fusion) keys
    for key in target_keys:
        if key not in loaded_keys:
            new_keys.append(key)

    # Verify: all new keys must be fusion keys
    non_fusion_new = [k for k in new_keys if k not in _EXPECTED_FUSION_KEYS]
    if non_fusion_new:
        raise RuntimeError(
            f"Non-fusion target keys not loaded from checkpoint: {non_fusion_new}"
        )

    # Verify: all non-loaded target keys are only fusion
    missing_inherited = [k for k in target_keys
                         if k not in loaded_keys and k not in _EXPECTED_FUSION_KEYS]
    if missing_inherited:
        raise RuntimeError(
            f"Expected inherited keys not found in source: {missing_inherited}"
        )

    model.load_state_dict(new_sd, strict=True)

    # For known optogpt.pt: verify exact key counts
    if is_known_optogpt:
        expected = {'source': 168, 'target': 178, 'inherited': 174, 'fusion': 4}
        actual = {
            'source': len(source_keys), 'target': len(target_keys),
            'inherited': len(loaded_keys), 'fusion': len(new_keys),
        }
        for k in expected:
            if actual[k] != expected[k]:
                raise RuntimeError(
                    f"Key count mismatch for known optogpt.pt: "
                    f"{k}={actual[k]}, expected {expected[k]}"
                )
        print(f"  ✓ Strict key counts: source={actual['source']}, "
              f"target={actual['target']}, inherited={actual['inherited']}, "
              f"fusion={actual['fusion']}")

    print(f"  Inherited: {len(loaded_keys)} keys")
    print(f"  New (fusion+): {len(new_keys)} keys")
    for nk in sorted(new_keys):
        print(f"    {nk}")
    print(f"  Architecture: {architecture_version}")

    enriched = {
        "model_type": "joint_sp",
        "architecture_version": architecture_version,
        "spec_dim": SPEC_DIM,
        "branch_dim": BRANCH_DIM,
        "spec_layout": ["Rs", "Ts", "Rp", "Tp"],
        "theta_deg": THETA_DEG,
        "polarizations": ["s", "p"],
        "allowed_materials": ALLOWED_MATERIALS,
        "struc_word_dict": word_dict,
        "struc_index_dict": index_dict,
        "N": N, "d_model": d_model, "d_ff": d_ff,
        "head_num": h, "dropout": dropout,
        "pretrained_source": str(pretrained_path),
        "pretrained_sha256": ckpt_full_hash,
        "n_loaded_keys": len(loaded_keys),
        "n_new_keys": len(new_keys),
    }

    return model, word_dict, index_dict, enriched, False


def save_sp_checkpoint(model, optimizer, epoch, loss_all, path, configs,
                       best_dev_loss=None, best_epoch=None,
                       patience_counter=0, training_phase='B',
                       data_manifest_hash=None, global_step=0,
                       pretrained_sha256=None, scheduler=None, scaler=None,
                       batches_per_epoch=None):
    """
    Save comprehensive TransformerSP checkpoint with RNG states.

    Requirements (raises on violation):
      - architecture_version must exist on model and in configs, and match
      - pretrained_sha256 is MANDATORY (64-char lowercase hex)
      - Model build config (N, d_model, d_ff, h, dropout, tgt_vocab) must
        match configs
    """
    save_dir = Path(path).parent.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizer_state = None
    if optimizer is not None:
        if hasattr(optimizer, 'optimizer'):
            optimizer_state = optimizer.optimizer.state_dict()
        elif hasattr(optimizer, 'state_dict'):
            optimizer_state = optimizer.state_dict()

    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    scaler_state = scaler.state_dict() if scaler is not None else None

    cfg = dict(configs) if not isinstance(configs, dict) else dict(configs)

    # ---- Architecture version enforcement ----
    model_arch = getattr(model, 'architecture_version', None)
    cfg_arch = cfg.get('architecture_version')
    if model_arch is None:
        raise RuntimeError(
            "Refusing to save: model has no architecture_version attribute. "
            "Use make_model_SP(architecture_version=...)"
        )
    if cfg_arch is not None and cfg_arch != model_arch:
        raise RuntimeError(
            f"Architecture version mismatch: model={model_arch}, configs={cfg_arch}"
        )
    cfg['architecture_version'] = model_arch
    cfg['model_type'] = 'joint_sp'

    # ---- pretrained_sha256 enforcement ----
    effective_hash = pretrained_sha256 or cfg.get('pretrained_sha256')
    if effective_hash is None:
        raise RuntimeError(
            "Refusing to save: pretrained_sha256 is required but neither "
            "argument nor configs provides it."
        )
    if not isinstance(effective_hash, str) or len(effective_hash) != 64:
        raise ValueError(
            f"pretrained_sha256 must be 64-char hex string, got: "
            f"{repr(effective_hash)}"
        )
    if not all(c in '0123456789abcdefABCDEF' for c in effective_hash):
        raise ValueError(
            f"pretrained_sha256 contains non-hex characters: {repr(effective_hash)}"
        )
    effective_hash = effective_hash.lower()
    if (pretrained_sha256 is not None and cfg.get('pretrained_sha256') is not None
            and pretrained_sha256.lower() != cfg['pretrained_sha256'].lower()):
        raise RuntimeError(
            f"pretrained_sha256 mismatch: arg={pretrained_sha256.lower()}, "
            f"configs={cfg['pretrained_sha256'].lower()}"
        )
    cfg['pretrained_sha256'] = effective_hash

    # ---- Model config consistency ----
    word_dict = cfg.get('struc_word_dict')
    index_dict = cfg.get('struc_index_dict')
    if word_dict is None or index_dict is None:
        raise RuntimeError("configs missing struc_word_dict/struc_index_dict")

    # Extract from model instance
    model_N = getattr(model, '_build_N', None) or len(model.decoder.layers)
    model_d_model = getattr(model, '_build_d_model', None) or model.generator.proj.in_features
    model_d_ff = getattr(model, '_build_d_ff', None) or model.decoder.layers[0].feed_forward.w_1.out_features
    model_h = getattr(model, '_build_h', None) or model.decoder.layers[0].self_attn.h
    model_dropout = getattr(model, '_build_dropout', None)
    model_vocab = getattr(model, '_build_tgt_vocab', None) or model.generator.proj.out_features

    # Validate against configs
    checks = [
        ('N', model_N, cfg.get('N') or cfg.get('layers')),
        ('d_model', model_d_model, cfg.get('d_model')),
        ('d_ff', model_d_ff, cfg.get('d_ff')),
        ('head_num', model_h, cfg.get('head_num')),
    ]
    for name, model_val, cfg_val in checks:
        if cfg_val is not None and model_val != cfg_val:
            raise RuntimeError(
                f"Config mismatch: model {name}={model_val}, configs {name}={cfg_val}"
            )

    # Populate/verify configs
    cfg['N'] = model_N
    cfg['d_model'] = model_d_model
    cfg['d_ff'] = model_d_ff
    cfg['head_num'] = model_h

    # Dropout consistency — must match if present in configs
    cfg_dropout = cfg.get('dropout')
    if model_dropout is not None:
        if cfg_dropout is not None and cfg_dropout != model_dropout:
            raise RuntimeError(
                f"dropout mismatch: model={model_dropout}, configs={cfg_dropout}"
            )
        cfg['dropout'] = model_dropout
    elif cfg_dropout is not None:
        # Model has no stored dropout, keep configs value
        pass

    # Vocab consistency
    if len(word_dict) != model_vocab:
        raise RuntimeError(
            f"Vocab mismatch: word_dict len={len(word_dict)}, model tgt_vocab={model_vocab}"
        )
    if len(index_dict) != model_vocab:
        raise RuntimeError(
            f"Vocab mismatch: index_dict len={len(index_dict)}, model tgt_vocab={model_vocab}"
        )

    atomic_torch_save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_state,
        'scheduler_state_dict': scheduler_state,
        'scaler_state_dict': scaler_state,
        'loss_all': loss_all,
        'configs': cfg,
        'best_dev_loss': best_dev_loss,
        'best_epoch': best_epoch,
        'patience_counter': patience_counter,
        'training_phase': training_phase,
        'data_manifest_hash': data_manifest_hash,
        'batches_per_epoch': batches_per_epoch,
        'resume_granularity': 'epoch',
        'rng_python': random.getstate(),
        'rng_numpy': np.random.get_state(),
        'rng_torch_cpu': torch.get_rng_state(),
        'rng_torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, str(path))
    print(f"[save_sp_checkpoint] epoch={epoch}, phase={training_phase}, "
          f"arch={model_arch}, sha256={effective_hash[:16]}..., path={path}")
