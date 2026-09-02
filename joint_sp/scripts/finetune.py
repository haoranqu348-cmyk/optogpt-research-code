"""
joint_sp/scripts/finetune.py — Fine-tune TransformerSP on joint s+p data.

Two-phase training:
  Phase A (warmup): Freeze all except fusion layer
  Phase B: Unfreeze all, differential LR

Usage:
    python joint_sp/scripts/finetune.py \
        --data_dir data_60deg_sp_joint --pretrained model/optogpt.pt \
        --epochs 10 --batch_size 16 --lr 3e-5
"""

import os
import sys
import json
import time
import random
import hashlib
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# Path setup
_SCRIPT_DIR = Path(__file__).resolve().parent
_JOINT_SP = _SCRIPT_DIR.parent
_PKG_ROOT = _JOINT_SP.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from optogpt.core.datasets.datasets import PrepareDataAug, PAD, UNK
from optogpt.core.trains.train import LabelSmoothing
from joint_sp.model import (
    make_model_SP, load_sp_from_pretrained, load_joint_sp_checkpoint,
    save_sp_checkpoint, _get_cfg,
)
from joint_sp.constants import (
    SPEC_DIM, BRANCH_DIM, ALLOWED_MATERIALS, BANNED_MATERIALS,
    THETA_DEG, PAD_ID, UNK_ID, MAX_LAYERS,
    normalize_structure_tokens, structure_hash_from_tokens,
    validate_disk_structure_tokens,
)
from joint_sp.model import _OPTOGPT_PT_SHA256 as KNOWN_OPTOGPT_SHA256
from joint_sp.io_utils import atomic_json_dump

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(data_iter, model, criterion, device, desc="Train", optimizer=None,
              scaler=None, amp_enabled=False, global_step=0):
    """Run one epoch and return (average loss, optimizer step count)."""
    total_tokens = 0.0
    total_loss = 0.0
    start = time.time()

    for i, batch in enumerate(data_iter):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        ntokens = batch.ntokens.to(device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            out = model(
                batch.src.to(device),
                batch.trg.to(device),
                batch.src_mask,
                batch.trg_mask.to(device) if batch.trg_mask is not None else None,
            )
            generated = model.generator(out)
            normalized_loss = criterion(
                generated.contiguous().view(-1, generated.size(-1)),
                batch.trg_y.to(device).contiguous().view(-1),
            ) / ntokens

        if optimizer is not None:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(normalized_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                normalized_loss.backward()
                optimizer.step()
            global_step += 1

        loss_sum = normalized_loss.detach().float() * ntokens.float()
        total_loss += float(loss_sum.cpu())
        total_tokens += float(ntokens.detach().cpu())

        if i % 20 == 0:
            elapsed = time.time() - start
            rate = total_tokens / elapsed if elapsed > 0 else 0
            print(f"  {desc} Batch {i:4d}: loss={float(normalized_loss.detach()):.4f}, "
                  f"tokens/sec={rate:.1f}")

    return total_loss / total_tokens, global_step


def validate_data_strict(data_dir, word_dict):
    """Validate data compatibility — raises on any issue."""
    data_dir = Path(data_dir)
    import pickle
    
    issues = []
    complete_path = data_dir / "BUILD_COMPLETE.json"
    config_path = data_dir / "generation_config.json"
    if not complete_path.exists():
        issues.append(f"Missing successful build marker: {complete_path}")
    if not config_path.exists():
        issues.append(f"Missing generation contract: {config_path}")
    else:
        with open(config_path, encoding="utf-8") as f:
            contract = json.load(f)
        if contract.get("spec_layout") != ["Rs", "Ts", "Rp", "Tp"]:
            issues.append("generation_config spec_layout must be [Rs, Ts, Rp, Tp]")
        if contract.get("spec_dim") != SPEC_DIM:
            issues.append(f"generation_config spec_dim must be {SPEC_DIM}")
        if float(contract.get("theta_deg", float("nan"))) != float(THETA_DEG):
            issues.append(f"generation_config theta_deg must be {THETA_DEG}")

    split_hashes = {}
    
    # Check spectrum shape
    for split_name in ['train', 'dev']:
        spec_path = data_dir / f"Spectrum_{split_name}.pkl"
        struct_path = data_dir / f"Structure_{split_name}.pkl"
        if not spec_path.exists():
            issues.append(f"Missing {spec_path}")
            continue
        with open(spec_path, 'rb') as f:
            specs = pickle.load(f)
        with open(struct_path, 'rb') as f:
            structs = pickle.load(f)
        
        if specs.shape[1] != SPEC_DIM:
            issues.append(f"{split_name} spectrum dim={specs.shape[1]}, expected {SPEC_DIM}")
        if len(specs) != len(structs):
            issues.append(f"{split_name} count mismatch: {len(structs)} structs vs {len(specs)} specs")
        if not np.all(np.isfinite(specs)):
            issues.append(f"{split_name} contains NaN/Inf spectra")
        
    # Check structure validation (ALL, not just first 100)
        for i, s in enumerate(structs):
            try:
                cleaned = validate_disk_structure_tokens(s, word_dict, ALLOWED_MATERIALS)
            except ValueError as e:
                issues.append(f"{split_name}[{i}]: {e}")
                if len(issues) > 50:
                    break
        split_hashes[split_name] = {
            structure_hash_from_tokens(s) for s in structs
            if isinstance(s, (list, tuple))
        }
        
        # Check for double BOS/EOS (ALL structures)
        for i, s in enumerate(structs):
            bos_count = sum(1 for t in s if t == 'BOS')
            eos_count = sum(1 for t in s if t == 'EOS')
            if bos_count > 0 or eos_count > 0:
                issues.append(f"{split_name}[{i}] contains BOS/EOS in disk data (should not)")
                break  # one is enough to fail

    test_struct_path = data_dir / "Structure_test.pkl"
    if test_struct_path.exists():
        with open(test_struct_path, "rb") as f:
            test_structs = pickle.load(f)
        split_hashes["test"] = {structure_hash_from_tokens(s) for s in test_structs}
    if split_hashes.get("train", set()) & split_hashes.get("dev", set()):
        issues.append("Structure hash leakage: train intersects dev")
    if split_hashes.get("train", set()) & split_hashes.get("test", set()):
        issues.append("Structure hash leakage: train intersects test")
    if split_hashes.get("dev", set()) & split_hashes.get("test", set()):
        issues.append("Structure hash leakage: dev intersects test")
    
    if issues:
        raise ValueError("Data validation FAILED:\n  " + "\n  ".join(issues))
    
    print("  ✓ Data validation passed (strict)")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TransformerSP on joint data")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to optogpt.pt (default: model/optogpt.pt)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=1)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--fusion_warmup_epochs", type=int, default=2,
                        help="Epochs to train only fusion layer")
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from latest checkpoint")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--architecture_override", type=str, default=None,
                        help="Force architecture_version for legacy/unversioned checkpoints "
                             "(joint_sp_legacy_v1 or joint_sp_relu_v0)")
    parser.add_argument("--output_name", type=str, default="optogpt_60deg_sp")
    parser.add_argument("--output_dir", type=str, default=None)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    args = parser.parse_args()

    if args.scheduler_factor <= 0 or args.scheduler_factor >= 1:
        raise ValueError("scheduler_factor must be in (0, 1)")
    if args.scheduler_patience < 0 or args.min_lr < 0:
        raise ValueError("scheduler_patience and min_lr must be non-negative")

    amp_enabled = bool(args.amp and DEVICE.type == "cuda")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Paths
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _JOINT_SP / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained_path = args.pretrained or str(_PKG_ROOT / "model" / "optogpt.pt")
    resume_ckpt_path = None
    if args.resume_from:
        resume_ckpt_path = args.resume_from
    elif args.resume:
        resume_ckpt_path = str(output_dir / f"{args.output_name}_latest.pt")

    print("=" * 70)
    print(f"Fine-tuning TransformerSP: {THETA_DEG}° Joint s+p")
    print(f"  Device: {DEVICE}")
    print(f"  Data: {args.data_dir}")
    print(f"  Pretrained: {pretrained_path}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  AMP: {amp_enabled}, Scheduler: ReduceLROnPlateau")
    print(f"  Fusion warmup: {args.fusion_warmup_epochs} epochs")
    print(f"  Output: {output_dir / args.output_name}_*.pt")
    print("=" * 70)

    # ---- Load model ----
    print(f"\n[1/6] Loading model checkpoint...")
    if resume_ckpt_path:
        if not Path(resume_ckpt_path).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt_path}")
        model, word_dict, index_dict, configs = load_joint_sp_checkpoint(
            resume_ckpt_path, device=DEVICE,
            architecture_override=args.architecture_override,
        )
        is_joint = True
    else:
        model, word_dict, index_dict, configs, is_joint = load_sp_from_pretrained(
            pretrained_path, device=DEVICE,
            architecture_override=args.architecture_override,
        )
    if is_joint and not resume_ckpt_path:
        print("  Detected joint_sp checkpoint — setting fusion_warmup_epochs=0")
        args.fusion_warmup_epochs = 0
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {n_params:,}, Trainable: {trainable:,}")

    # ---- Load data ----
    print(f"\n[2/6] Loading data from {args.data_dir}")
    data_dir = Path(args.data_dir)

    required = ["Structure_train.pkl", "Spectrum_train.pkl",
                 "Structure_dev.pkl", "Spectrum_dev.pkl"]
    for f in required:
        if not (data_dir / f).exists():
            raise FileNotFoundError(f"Missing: {data_dir / f}")

    # Strict validation — raises on any issue
    validate_data_strict(data_dir, word_dict)

    data = PrepareDataAug(
        train_file=str(data_dir / "Structure_train.pkl"),
        train_spec_file=str(data_dir / "Spectrum_train.pkl"),
        train_ratio=100,
        dev_file=str(data_dir / "Structure_dev.pkl"),
        dev_spec_file=str(data_dir / "Spectrum_dev.pkl"),
        BATCH_SIZE=args.batch_size,
        spec_type="R_T",
        if_inverse="Inverse",
        struct_word_dict=word_dict,
        struct_index_dict=index_dict,
        shuffle_train=True,  # Enable per-epoch shuffling
    )
    print(f"  Train batches: {len(data.train_data)}, Dev batches: {len(data.dev_data)}")

    # Check for empty batches
    if len(data.train_data) == 0:
        raise ValueError("Training data produces 0 batches")
    if len(data.dev_data) == 0:
        raise ValueError("Dev data produces 0 batches")

    # Compute data manifest hash for provenance
    data_hashes = {}
    for fname in required:
        data_hashes[fname] = hashlib.sha256(
            (data_dir / fname).read_bytes()
        ).hexdigest()[:16]
    data_manifest_hash = hashlib.sha256(
        json.dumps(data_hashes, sort_keys=True).encode()
    ).hexdigest()[:16]
    print(f"  Data manifest hash: {data_manifest_hash}")

    # ---- Handle resume ----
    start_phase = 'A'
    start_epoch_offset = 0  # epochs already completed
    global_step = 0

    if resume_ckpt_path:
        print(f"\n[Resume] Loading checkpoint: {resume_ckpt_path}")
        resume_ckpt = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)

        # Verify architecture_version compatibility
        resume_cfg = resume_ckpt.get('configs', {})
        resume_arch = _get_cfg(resume_cfg, 'architecture_version')
        model_arch = getattr(model, 'architecture_version', None)
        if resume_arch and model_arch and resume_arch != model_arch:
            raise RuntimeError(
                f"Resume checkpoint architecture ({resume_arch}) "
                f"≠ current model ({model_arch})"
            )
        if resume_arch is None:
            raise RuntimeError(
                "Resume checkpoint has no architecture_version — "
                "refusing to resume. Unversioned checkpoints are not accepted."
            )

        saved_training = _get_cfg(resume_cfg, 'training_config')
        if not isinstance(saved_training, dict):
            raise RuntimeError("Resume checkpoint is missing training_config")
        requested_training = {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'fusion_warmup_epochs': args.fusion_warmup_epochs,
            'smoothing': args.smoothing,
            'seed': args.seed,
            'amp': amp_enabled,
            'scheduler_factor': args.scheduler_factor,
            'scheduler_patience': args.scheduler_patience,
            'min_lr': args.min_lr,
        }
        mismatches = {
            key: (saved_training.get(key), value)
            for key, value in requested_training.items()
            if saved_training.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Resume training configuration mismatch: {mismatches}")

        start_phase = resume_ckpt.get('training_phase', 'B')
        start_epoch_offset = resume_ckpt.get('epoch', 0)
        global_step = resume_ckpt.get('global_step', 0)
        best_dev_loss = resume_ckpt.get('best_dev_loss', float('inf'))
        best_epoch = resume_ckpt.get('best_epoch', 0)
        patience_counter = resume_ckpt.get('patience_counter', 0)
        loss_all = resume_ckpt.get('loss_all', {'train_loss': [], 'dev_loss': []})
        resume_optimizer_state = resume_ckpt.get('optimizer_state_dict')
        resume_scheduler_state = resume_ckpt.get('scheduler_state_dict')
        resume_scaler_state = resume_ckpt.get('scaler_state_dict')
        saved_batches_per_epoch = resume_ckpt.get('batches_per_epoch')
        if saved_batches_per_epoch != len(data.train_data):
            raise RuntimeError(
                f"Training batch count changed since checkpoint: "
                f"saved={saved_batches_per_epoch}, current={len(data.train_data)}"
            )
        if resume_scheduler_state is None or resume_scaler_state is None:
            raise RuntimeError(
                "Resume checkpoint lacks scheduler/scaler state; "
                "create a new production-format checkpoint first"
            )

        # Restore RNG states
        if 'rng_python' in resume_ckpt and resume_ckpt['rng_python'] is not None:
            random.setstate(resume_ckpt['rng_python'])
        if 'rng_numpy' in resume_ckpt and resume_ckpt['rng_numpy'] is not None:
            np.random.set_state(resume_ckpt['rng_numpy'])
        if 'rng_torch_cpu' in resume_ckpt and resume_ckpt['rng_torch_cpu'] is not None:
            torch.set_rng_state(resume_ckpt['rng_torch_cpu'])
        if resume_ckpt.get('rng_torch_cuda') and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume_ckpt['rng_torch_cuda'])

        # Restore optimizer if available
        resume_optimizer_state = resume_ckpt.get('optimizer_state_dict')

        # Check data manifest consistency
        saved_hash = resume_ckpt.get('data_manifest_hash')
        if saved_hash:
            current_hash = hashlib.sha256(
                json.dumps(data_hashes, sort_keys=True).encode()
            ).hexdigest()[:16]
            if saved_hash != current_hash:
                raise RuntimeError(
                    f"Data manifest changed since checkpoint! "
                    f"Saved: {saved_hash}, Current: {current_hash}"
                )

        print(f"  Resumed: epoch_offset={start_epoch_offset}, phase={start_phase}, "
              f"best_dev_loss={best_dev_loss:.4f}, global_step={global_step}")

        # Set model to correct phase
        if start_phase == 'A':
            for name, param in model.named_parameters():
                if 'fusion' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
    else:
        if args.resume or args.resume_from:
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt_path}")
        loss_all = {'train_loss': [], 'dev_loss': []}
        best_dev_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        resume_optimizer_state = None
        resume_scheduler_state = None
        resume_scaler_state = None

    configs = dict(configs)
    configs['training_config'] = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'fusion_warmup_epochs': args.fusion_warmup_epochs,
        'smoothing': args.smoothing,
        'seed': args.seed,
        'amp': amp_enabled,
        'scheduler_factor': args.scheduler_factor,
        'scheduler_patience': args.scheduler_patience,
        'min_lr': args.min_lr,
        'resume_granularity': 'epoch',
    }

    # ---- Training setup ----
    print(f"\n[3/6] Setting up training...")
    criterion = LabelSmoothing(len(word_dict), padding_idx=PAD, smoothing=args.smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if resume_scaler_state is not None:
        scaler.load_state_dict(resume_scaler_state)

    # Phase A: freeze all except fusion (unless resuming from Phase B)
    if start_phase == 'A':
        fusion_params = []
        for name, param in model.named_parameters():
            if 'fusion' in name:
                fusion_params.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Assert: ONLY fusion parameters are trainable
        trainable_non_fusion = [
            n for n, p in model.named_parameters()
            if p.requires_grad and 'fusion' not in n
        ]
        frozen_fusion = [
            n for n, p in model.named_parameters()
            if not p.requires_grad and 'fusion' in n
        ]
        if trainable_non_fusion:
            raise RuntimeError(
                f"Phase A: non-fusion params are trainable: {trainable_non_fusion}"
            )
        if frozen_fusion:
            raise RuntimeError(
                f"Phase A: fusion params are frozen: {frozen_fusion}"
            )
        print(f"  Phase A: {len(fusion_params)} trainable params (fusion only) ✓")
    else:
        fusion_params = [p for n, p in model.named_parameters() if 'fusion' in n]

    optimizer_a = torch.optim.Adam(fusion_params, lr=args.lr,
                                    betas=(0.9, 0.98), eps=1e-9)
    scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_a, mode='min', factor=args.scheduler_factor,
        patience=args.scheduler_patience, min_lr=args.min_lr,
    )
    # Restore optimizer if resuming into Phase A
    if start_phase == 'A' and resume_optimizer_state is not None:
        optimizer_a.load_state_dict(resume_optimizer_state)
        scheduler_a.load_state_dict(resume_scheduler_state)

    # ---- Phase A: Fusion warmup ----
    if start_phase == 'A' and args.fusion_warmup_epochs > 0:
        # Skip already-completed epochs
        remaining_a = args.fusion_warmup_epochs - start_epoch_offset
        if remaining_a <= 0:
            print(f"  Phase A already completed ({start_epoch_offset}/{args.fusion_warmup_epochs}), skipping")
        else:
            print(f"\n[4/6] Phase A: Fusion warmup (epochs {start_epoch_offset+1}-{args.fusion_warmup_epochs})...")
            for epoch in range(start_epoch_offset, args.fusion_warmup_epochs):
                data.reshuffle_train()
                model.train()
                train_loss, global_step = run_epoch(
                    data.train_data, model, criterion, DEVICE,
                    f"PhaseA Train E{epoch+1}", optimizer=optimizer_a,
                    scaler=scaler, amp_enabled=amp_enabled, global_step=global_step,
                )

                model.eval()
                with torch.no_grad():
                    dev_loss, _ = run_epoch(
                        data.dev_data, model, criterion, DEVICE,
                        f"PhaseA Dev  E{epoch+1}", amp_enabled=amp_enabled,
                        global_step=global_step,
                    )

                scheduler_a.step(dev_loss)

                loss_all['train_loss'].append(float(train_loss))
                loss_all['dev_loss'].append(float(dev_loss))
                print(f"  PhaseA Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                      f"dev_loss={dev_loss:.4f}")

                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_path = output_dir / f"{args.output_name}_best.pt"
                    save_sp_checkpoint(model, optimizer_a, epoch + 1, loss_all,
                                       str(best_path), configs,
                                       best_dev_loss=best_dev_loss,
                                       best_epoch=best_epoch,
                                       patience_counter=patience_counter,
                                       training_phase='A',
                                       data_manifest_hash=data_manifest_hash,
                                       global_step=global_step,
                                       scheduler=scheduler_a, scaler=scaler,
                                       batches_per_epoch=len(data.train_data))
                else:
                    patience_counter += 1

                # Save latest
                latest_path = output_dir / f"{args.output_name}_latest.pt"
                save_sp_checkpoint(model, optimizer_a, epoch + 1, loss_all,
                                   str(latest_path), configs,
                                   best_dev_loss=best_dev_loss,
                                   best_epoch=best_epoch,
                                   patience_counter=patience_counter,
                                   training_phase='A',
                                   data_manifest_hash=data_manifest_hash,
                                   global_step=global_step,
                                   scheduler=scheduler_a, scaler=scaler,
                                   batches_per_epoch=len(data.train_data))

    # ---- Phase B: Full fine-tuning ----
    # If we skipped Phase A entirely, ensure we have loss_all initialized
    if start_phase == 'B' and not loss_all.get('train_loss'):
        loss_all = {'train_loss': [], 'dev_loss': []}

    print(f"\n[5/6] Phase B: Full fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True

    fusion_lr = args.lr * 5
    pretrained_lr = args.lr * 0.2
    param_groups = []
    for name, param in model.named_parameters():
        if 'fusion' in name:
            param_groups.append({'params': param, 'lr': fusion_lr})
        else:
            param_groups.append({'params': param, 'lr': pretrained_lr})

    optimizer_b = torch.optim.Adam(param_groups, lr=args.lr,
                                    betas=(0.9, 0.98), eps=1e-9)
    scheduler_b = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_b, mode='min', factor=args.scheduler_factor,
        patience=args.scheduler_patience, min_lr=args.min_lr,
    )
    # Restore optimizer if resuming into Phase B
    if start_phase == 'B' and resume_optimizer_state is not None:
        optimizer_b.load_state_dict(resume_optimizer_state)
        scheduler_b.load_state_dict(resume_scheduler_state)

    # Calculate epoch offsets for Phase B resume
    if start_phase == 'B':
        phase_b_offset = max(0, start_epoch_offset - args.fusion_warmup_epochs)
    else:
        phase_b_offset = 0
    total_phaseb_epochs = args.epochs - args.fusion_warmup_epochs

    if total_phaseb_epochs <= 0:
        print(f"  epochs ({args.epochs}) <= fusion_warmup_epochs "
              f"({args.fusion_warmup_epochs}), skipping Phase B")
    else:
        for epoch in range(phase_b_offset, total_phaseb_epochs):
            global_epoch = args.fusion_warmup_epochs + epoch + 1
            data.reshuffle_train()

            model.train()
            train_loss, global_step = run_epoch(
                data.train_data, model, criterion, DEVICE,
                f"PhaseB Train E{global_epoch}", optimizer=optimizer_b,
                scaler=scaler, amp_enabled=amp_enabled, global_step=global_step,
            )

            model.eval()
            with torch.no_grad():
                dev_loss, _ = run_epoch(
                    data.dev_data, model, criterion, DEVICE,
                    f"PhaseB Dev  E{global_epoch}", amp_enabled=amp_enabled,
                    global_step=global_step,
                )

            scheduler_b.step(dev_loss)

            loss_all['train_loss'].append(float(train_loss))
            loss_all['dev_loss'].append(float(dev_loss))
            print(f"  PhaseB Epoch {global_epoch}: train_loss={train_loss:.4f}, "
                  f"dev_loss={dev_loss:.4f}")

            # Update best/patience FIRST
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_epoch = global_epoch
                patience_counter = 0
                best_path = output_dir / f"{args.output_name}_best.pt"
                save_sp_checkpoint(model, optimizer_b, global_epoch, loss_all,
                                   str(best_path), configs,
                                   best_dev_loss=best_dev_loss,
                                   best_epoch=best_epoch,
                                   patience_counter=patience_counter,
                                   training_phase='B',
                                   data_manifest_hash=data_manifest_hash,
                                   global_step=global_step,
                                   scheduler=scheduler_b, scaler=scaler,
                                   batches_per_epoch=len(data.train_data))
                print(f"  ✓ New best model! dev_loss={dev_loss:.4f}")
            else:
                patience_counter += 1

            # Save latest AFTER updating best/patience
            latest_path = output_dir / f"{args.output_name}_latest.pt"
            save_sp_checkpoint(model, optimizer_b, global_epoch, loss_all,
                               str(latest_path), configs,
                               best_dev_loss=best_dev_loss,
                               best_epoch=best_epoch,
                               patience_counter=patience_counter,
                               training_phase='B',
                               data_manifest_hash=data_manifest_hash,
                               global_step=global_step,
                               scheduler=scheduler_b, scaler=scaler,
                               batches_per_epoch=len(data.train_data))

            if args.early_stopping and patience_counter >= args.patience:
                print(f"  Early stopping at epoch {global_epoch}")
                break

    # ---- Save final ----
    print(f"\n[6/6] Saving final outputs...")

    # Loss history
    loss_path = output_dir / f"{args.output_name}_loss_history.json"
    atomic_json_dump(loss_all, loss_path)

    # Config
    config_path = output_dir / f"{args.output_name}_config.json"
    atomic_json_dump(configs, config_path)

    print(f"\nDone! Best epoch: {best_epoch}, Best dev_loss: {best_dev_loss:.4f}")
    print(f"  Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
