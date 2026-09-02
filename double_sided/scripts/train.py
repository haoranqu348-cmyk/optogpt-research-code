"""Train the SIDE_SEP model only after checkpoint and material gates pass."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from double_sided.config import DoubleSidedConfig
from double_sided.decoder import constrained_decode
from double_sided.physics import simulate_c, summarize
from double_sided.training import build_output_mask, make_loaders, run_epoch
from optogpt.core.datasets.sim import load_materials


def set_training_phase(model, phase):
    phase = phase.upper()
    if phase not in ("A", "B", "C"):
        raise ValueError("phase must be A, B, or C")
    for name, parameter in model.named_parameters():
        if phase == "A":
            parameter.requires_grad = name.startswith(("decoder.", "tgt_embed.", "generator."))
        elif phase == "B":
            parameter.requires_grad = name.startswith(("fusion.", "decoder.", "tgt_embed.", "generator."))
        else:
            parameter.requires_grad = True
    return {
        "phase": phase,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _top_k_sampler(temperature=0.9, top_k=32):
    def sample(logits):
        values, indices = torch.topk(logits / temperature, k=min(top_k, logits.size(-1)), dim=-1)
        selected = torch.multinomial(torch.softmax(values, dim=-1), 1)
        return indices.gather(1, selected).squeeze(1)
    return sample


def physical_probe(model, word_dict, index_dict, allowed_materials, max_layers, nk_dict, config,
                   device, candidates=32, seed=20260728):
    n = len(config.wavelengths_nm)
    target = np.concatenate([np.zeros(n), np.ones(n), np.zeros(n), np.ones(n)]).astype(np.float32)
    generated = constrained_decode(
        model, [target], word_dict, index_dict, allowed_materials,
        max_layers_per_side=max_layers, device=device,
    )
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        if candidates > 1:
            generated.extend(constrained_decode(
                model, np.repeat(target[None, :], candidates - 1, axis=0),
                word_dict, index_dict, allowed_materials,
                max_layers_per_side=max_layers, sample_fn=_top_k_sampler(), device=device,
            ))
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    unique = {}
    for structure in generated:
        merged = structure.merged()
        unique.setdefault(merged.physical_hash(), merged)
    ranked = []
    for structure in unique.values():
        ranked.append((summarize(simulate_c(structure, nk_dict, config)), structure))
    ranked.sort(key=lambda item: item[0]["objective"])
    metrics, structure = ranked[0]
    objectives = [item[0]["objective"] for item in ranked]
    return {
        "tokens": structure.to_tokens(), "physical_layer_counts": structure.physical_layer_counts,
        "metrics": metrics, "requested_candidates": candidates,
        "valid_candidates": len(generated), "unique_physical_structures": len(unique),
        "median_objective": float(np.median(objectives)), "tmm_calls": 2 * n * len(unique),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialized-checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--material-gate-manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-layers-per-side", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--phase", choices=["A", "B", "C"], default="A")
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--physical-eval-every", type=int, default=1)
    parser.add_argument("--physical-eval-candidates", type=int, default=32)
    parser.add_argument("--resume-from", default=None)
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--allow-base-only", action="store_true",
                        help="Allow a base-material training experiment despite a failed new-material gate")
    args = parser.parse_args()
    if args.allow_base_only:
        gate = {"training_gate_passed": False, "mode": "base_materials_only"}
    else:
        if not args.material_gate_manifest:
            raise ValueError("Expanded-material training requires --material-gate-manifest")
        gate = json.loads(Path(args.material_gate_manifest).read_text())
        if not gate.get("training_gate_passed", False):
            raise RuntimeError("Material Pareto gate did not pass; refusing formal expanded-material training")
    if args.max_layers_per_side < 1:
        raise ValueError("max-layers-per-side must be positive")
    if not 0 <= args.smoothing < 1:
        raise ValueError("smoothing must be in [0, 1)")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.initialized_checkpoint, map_location=device, weights_only=False)
    configs = checkpoint["configs"]
    if configs.get("model_type") != "double_sided_joint_sp":
        raise ValueError("Expected a staged double-sided initialized checkpoint")
    from joint_sp.model import make_model_SP
    word_dict, index_dict = configs["struc_word_dict"], configs["struc_index_dict"]
    model = make_model_SP(
        tgt_vocab=len(word_dict), N=configs["N"], d_model=configs["d_model"],
        d_ff=configs["d_ff"], h=configs["head_num"], dropout=configs["dropout"],
        architecture_version=configs["architecture_version"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    allowed_materials = tuple(configs.get("allowed_materials", []))
    loaders = make_loaders(
        args.data_dir, word_dict, allowed_materials, args.max_layers_per_side,
        args.batch_size, args.seed,
    )
    phase_metadata = set_training_phase(model, args.phase)
    criterion = nn.NLLLoss(ignore_index=word_dict["PAD"], reduction="sum")
    output_mask = build_output_mask(word_dict, allowed_materials)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    output = Path(args.output_dir)
    if args.resume_from:
        output.mkdir(parents=True, exist_ok=True)
        resume = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        if resume.get("phase") != args.phase:
            raise RuntimeError("Resume phase mismatch")
        history = list(resume.get("history", []))
        start_epoch = int(resume["epoch"]) + 1
        best_token = min((row["dev_token_loss"] for row in history), default=float("inf"))
        best_physical = min((row.get("physical_probe", {}).get("metrics", {}).get("objective", float("inf"))
                             for row in history), default=float("inf"))
        if resume.get("scaler_state_dict"):
            scaler.load_state_dict(resume["scaler_state_dict"])
    else:
        output.mkdir(parents=True, exist_ok=False)
        history, start_epoch, best_token, best_physical = [], 1, float("inf"), float("inf")
    config = DoubleSidedConfig(technical_max_layers_per_side=max(32, args.max_layers_per_side)).validate()
    nk_dict = load_materials(
        all_mats=[config.substrate, *allowed_materials], wavelengths=config.wavelengths_nm / 1000.0,
        DATABASE=str(Path(__file__).resolve().parents[2] / "optogpt" / "nk"),
    )
    run_metadata = {
        **phase_metadata, "device": str(device), "amp": amp_enabled,
        "batch_size": args.batch_size, "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "lr": args.lr, "smoothing": args.smoothing,
        "allowed_output_tokens": int(output_mask.sum()), "forbidden_output_tokens": int((~output_mask).sum()),
    }
    (output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            model, loaders["train"], criterion, device, optimizer,
            output_mask=output_mask, smoothing=args.smoothing, scaler=scaler,
            amp_enabled=amp_enabled, grad_accum_steps=args.grad_accum_steps,
        )
        with torch.no_grad():
            dev_loss = run_epoch(
                model, loaders["dev"], criterion, device,
                output_mask=output_mask, smoothing=args.smoothing,
                amp_enabled=amp_enabled,
            )
        row = {"epoch": epoch, "phase": args.phase, "train_token_loss": train_loss,
               "dev_token_loss": dev_loss}
        if args.physical_eval_every > 0 and epoch % args.physical_eval_every == 0:
            row["physical_probe"] = physical_probe(
                model, word_dict, index_dict, allowed_materials,
                args.max_layers_per_side, nk_dict, config, device,
                candidates=args.physical_eval_candidates, seed=args.seed + epoch,
            )
        history.append(row)
        payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "configs": dict(configs, max_layers_per_side=args.max_layers_per_side),
                   "history": history, "epoch": epoch, "phase": args.phase,
                   "scaler_state_dict": scaler.state_dict() if amp_enabled else None,
                   "run_metadata": run_metadata,
                   "validation_note": "token loss is diagnostic; physical TMM evaluation is required"}
        atomic_torch_save(payload, output / "latest.pt")
        if dev_loss < best_token:
            best_token = dev_loss; atomic_torch_save(payload, output / "best_token_loss.pt")
        physical_objective = row.get("physical_probe", {}).get("metrics", {}).get("objective")
        if physical_objective is not None and physical_objective < best_physical:
            best_physical = physical_objective
            atomic_torch_save(payload, output / "best_physical.pt")
        (output / "training_log.json").write_text(json.dumps(history, indent=2))
        print(row, flush=True)


if __name__ == "__main__":
    main()
