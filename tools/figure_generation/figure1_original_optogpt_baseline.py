#!/usr/bin/env python3
"""Generate the Figure 1(c) zero-shot baseline from original OptoGPT."""

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


WORKSPACE = Path(__file__).resolve().parent
PROJECT = WORKSPACE / "optogpt_project"
PACKAGE_ROOT = PROJECT / "optogpt"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from joint_sp.constants import (  # noqa: E402
    ALLOWED_MATERIALS,
    BANNED_MATERIALS,
    MAX_LAYERS,
    SUBSTRATE,
    SUBSTRATE_THICK_NM,
    THETA_DEG,
)
from joint_sp.decoder import (  # noqa: E402
    build_joint_logits_mask,
    generate_candidates_sp,
    tmm_rerank_joint,
)
from joint_sp.model import (  # noqa: E402
    OptoGPTLegacyFeedForward,
    OptoGPTLegacyFullyConnected,
)
from optogpt.core.datasets.sim import load_materials  # noqa: E402
from optogpt.core.models.transformer import (  # noqa: E402
    Decoder,
    DecoderLayer,
    Embeddings,
    Generator,
    MultiHeadedAttention,
    PositionalEncoding,
    Transformer_I,
)


EXPECTED_SHA256 = "a7677602ae8dae60dababde9bd3981ad16be61430a22e797ba359b1d01921d85"
WAVELENGTHS_UM = np.arange(0.4, 1.101, 0.01)
WAVELENGTHS_NM = (WAVELENGTHS_UM * 1000).astype(int)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cfg_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def load_checkpoint(path, device):
    # The checkpoint was serialized with the original top-level `core` package.
    import optogpt.core as core_package
    import optogpt.core.models as core_models
    import optogpt.core.models.transformer as core_transformer

    aliases = {
        "core": core_package,
        "core.models": core_models,
        "core.models.transformer": core_transformer,
    }
    previous = {name: sys.modules.get(name) for name in aliases}
    sys.modules.update(aliases)
    try:
        return torch.load(path, map_location=device)
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


def make_legacy_model(spec_dim, vocab_size, layers, d_model, d_ff, heads, dropout):
    """Construct the exact checkpoint-era forward semantics."""
    clone = copy.deepcopy
    attention = MultiHeadedAttention(heads, d_model, dropout=dropout)
    feed_forward = OptoGPTLegacyFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    fc = OptoGPTLegacyFullyConnected(spec_dim, d_model, dropout)
    return Transformer_I(
        fc,
        Decoder(
            DecoderLayer(
                d_model,
                clone(attention),
                clone(attention),
                clone(feed_forward),
                dropout,
            ),
            layers,
        ),
        nn.Sequential(Embeddings(d_model, vocab_size), clone(position)),
        Generator(d_model, vocab_size),
    )


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def rank_for_condition(candidates, target_joint, nk_dict, condition):
    evaluated, failures = tmm_rerank_joint(
        candidates,
        target_joint,
        nk_dict,
        wavelengths=WAVELENGTHS_UM,
        theta=THETA_DEG,
        objective="joint_error",
    )
    error_key = "E_s" if condition == "s" else "E_p"
    evaluated.sort(key=lambda item: (item[error_key], item["E_joint"], item["n_layers"]))
    return evaluated, failures


def compact_candidate(candidate):
    return {
        key: candidate[key]
        for key in (
            "tokens",
            "materials",
            "thicknesses",
            "n_layers",
            "E_s",
            "E_p",
            "E_joint",
            "sim_Rs",
            "sim_Ts",
            "sim_Rp",
            "sim_Tp",
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/Users/quhaoran/lab/20260801_145630_optogpt_complete_paper_archive/"
            "trained_models/model/optogpt.pt"
        ),
    )
    parser.add_argument(
        "--target-json",
        type=Path,
        default=WORKSPACE / "paper_figures" / "data" / "figure4_same_target_candidates.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "paper_figures" / "data",
    )
    args = parser.parse_args()

    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != EXPECTED_SHA256:
        raise RuntimeError(
            f"Unexpected original OptoGPT SHA-256: {checkpoint_hash}; expected {EXPECTED_SHA256}"
        )

    target_record = json.loads(args.target_json.read_text())
    target_data = target_record["shared_target"]
    target_joint = np.asarray(
        target_data["Rs"] + target_data["Ts"] + target_data["Rp"] + target_data["Tp"],
        dtype=np.float32,
    )
    target_s = target_joint[:142]
    target_p = target_joint[142:]

    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["configs"]
    word_dict = cfg_value(config, "struc_word_dict")
    index_dict = cfg_value(config, "struc_index_dict")
    spec_dim = cfg_value(config, "spec_dim", 142)
    if spec_dim != 142:
        raise RuntimeError(f"Original OptoGPT spec_dim is {spec_dim}, expected 142")

    model = make_legacy_model(
        spec_dim=spec_dim,
        vocab_size=len(word_dict),
        layers=cfg_value(config, "layers", 6),
        d_model=cfg_value(config, "d_model", 1024),
        d_ff=cfg_value(config, "d_ff", 512),
        heads=cfg_value(config, "head_num", 8),
        dropout=cfg_value(config, "dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    nk_dict = load_materials(
        all_mats=[SUBSTRATE] + ALLOWED_MATERIALS + list(BANNED_MATERIALS),
        wavelengths=WAVELENGTHS_UM,
        DATABASE=str(PACKAGE_ROOT / "optogpt" / "nk"),
    )
    logits_mask, _ = build_joint_logits_mask(word_dict, ALLOWED_MATERIALS)

    runs = {}
    for offset, (condition, target_branch) in enumerate((('s', target_s), ('p', target_p))):
        run_seed = args.seed + offset
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        candidates = generate_candidates_sp(
            model,
            target_branch,
            word_dict,
            index_dict,
            num_candidates=args.num_candidates,
            max_len=MAX_LAYERS + 2,
            device=device,
            logits_mask=logits_mask,
        )
        ranked, failures = rank_for_condition(candidates, target_joint, nk_dict, condition)
        if not ranked:
            raise RuntimeError(f"No TMM-valid {condition}-conditioned candidate was retained")
        runs[f"{condition}_conditioned"] = {
            "condition_input": "[Rs, Ts]" if condition == "s" else "[Rp, Tp]",
            "reranking_metric": "E_s" if condition == "s" else "E_p",
            "seed": run_seed,
            "requested_candidates": args.num_candidates,
            "generated_unique_candidates": len(candidates),
            "retained_tmm_candidates": len(ranked),
            "tmm_failures": failures,
            "best_candidate": compact_candidate(ranked[0]),
            "candidate_pool": [compact_candidate(candidate) for candidate in ranked],
        }

    output = {
        "run_contract": {
            "baseline": "Original pretrained OptoGPT - zero-shot single-polarization conditioning",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "architecture_version": "optogpt_legacy_v1",
            "checkpoint_spec_dim": spec_dim,
            "evaluation_angle_deg": THETA_DEG,
            "evaluation_layout": ["Rs", "Ts", "Rp", "Tp"],
            "wavelengths_nm": WAVELENGTHS_NM.tolist(),
            "allowed_materials": ALLOWED_MATERIALS,
            "sampling": {"top_k": 10, "top_p": 0.9, "temperature": 1.0},
            "device": str(device),
            "limitation": (
                "Zero-shot out-of-distribution baseline; the original checkpoint was not "
                "fine-tuned for the 60-degree joint-polarization task."
            ),
        },
        "shared_target": {
            "archived_validation_index": target_data["archived_validation_index"],
            "source_target_tokens": target_data["source_target_tokens"],
            "Rs": target_data["Rs"],
            "Ts": target_data["Ts"],
            "Rp": target_data["Rp"],
            "Tp": target_data["Tp"],
        },
        **runs,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "figure1_original_optogpt_zero_shot.json"
    json_path.write_text(json.dumps(output, indent=2))

    csv_path = args.output_dir / "figure1_original_optogpt_zero_shot_spectra.csv"
    s_best = runs["s_conditioned"]["best_candidate"]
    p_best = runs["p_conditioned"]["best_candidate"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "wavelength_nm",
                "target_Rs", "target_Ts", "target_Rp", "target_Tp",
                "s_conditioned_Rs", "s_conditioned_Ts", "s_conditioned_Rp", "s_conditioned_Tp",
                "p_conditioned_Rs", "p_conditioned_Ts", "p_conditioned_Rp", "p_conditioned_Tp",
            ]
        )
        for index, wavelength in enumerate(WAVELENGTHS_NM):
            writer.writerow(
                [
                    wavelength,
                    target_data["Rs"][index], target_data["Ts"][index],
                    target_data["Rp"][index], target_data["Tp"][index],
                    s_best["sim_Rs"][index], s_best["sim_Ts"][index],
                    s_best["sim_Rp"][index], s_best["sim_Tp"][index],
                    p_best["sim_Rs"][index], p_best["sim_Ts"][index],
                    p_best["sim_Rp"][index], p_best["sim_Tp"][index],
                ]
            )

    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "device": str(device),
                "target_index": target_data["archived_validation_index"],
                "s_conditioned": {
                    "generated": runs["s_conditioned"]["generated_unique_candidates"],
                    "retained": runs["s_conditioned"]["retained_tmm_candidates"],
                    "tokens": s_best["tokens"],
                    "E_s": s_best["E_s"],
                    "E_p": s_best["E_p"],
                    "E_joint": s_best["E_joint"],
                },
                "p_conditioned": {
                    "generated": runs["p_conditioned"]["generated_unique_candidates"],
                    "retained": runs["p_conditioned"]["retained_tmm_candidates"],
                    "tokens": p_best["tokens"],
                    "E_s": p_best["E_s"],
                    "E_p": p_best["E_p"],
                    "E_joint": p_best["E_joint"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
