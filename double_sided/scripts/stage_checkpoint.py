"""Safely stage and migrate a supplied joint s+p checkpoint without overwriting it."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from double_sided.model import extend_vocabulary, migrate_joint_sp_model
from joint_sp.model import load_joint_sp_checkpoint


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extra-material", action="append", default=[])
    parser.add_argument("--thickness-min", type=int, default=10)
    parser.add_argument("--thickness-max", type=int, default=500)
    parser.add_argument("--thickness-step", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source = Path(args.checkpoint).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    staged = output / "source_checkpoints"
    staged.mkdir()
    copied = staged / source.name
    shutil.copy2(source, copied)
    if sha256(source) != sha256(copied):
        raise RuntimeError("Checkpoint copy hash mismatch")
    model, word_dict, _, configs = load_joint_sp_checkpoint(copied, device=args.device)
    extra_tokens = [
        f"{material}_{thickness}"
        for material in args.extra_material
        for thickness in range(args.thickness_min, args.thickness_max + 1, args.thickness_step)
    ]
    target_word_dict, target_index_dict = extend_vocabulary(word_dict, extra_tokens)
    migrated = migrate_joint_sp_model(model, word_dict, target_word_dict, configs)
    migrated_configs = dict(configs)
    migrated_configs.update({
        "model_type": "double_sided_joint_sp", "token_contract": "BOS/front/SIDE_SEP/back/EOS",
        "struc_word_dict": target_word_dict, "struc_index_dict": target_index_dict,
        "source_checkpoint": str(copied), "source_checkpoint_sha256": sha256(copied),
        "vocabulary_transfer": migrated.vocabulary_transfer,
        "allowed_materials": sorted(set(configs.get("allowed_materials", []))
                                    | set(args.extra_material)),
    })
    init_path = output / "double_sided_initialized.pt"
    torch.save({"model_state_dict": migrated.state_dict(), "configs": migrated_configs,
                "training_status": "initialized_not_trained"}, init_path)
    metadata = {
        "source_path": str(source), "copied_path": str(copied),
        "source_sha256": sha256(source), "initialized_path": str(init_path),
        "initialized_sha256": sha256(init_path),
        "source_unchanged_after_copy": sha256(source) == sha256(copied),
        "vocabulary_transfer": migrated.vocabulary_transfer,
        "parameter_count": sum(parameter.numel() for parameter in migrated.parameters()),
        "training_status": "initialized_not_trained",
    }
    (output / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
