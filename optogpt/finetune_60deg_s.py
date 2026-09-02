"""
Fine-tune OptoGPT on fixed 60° s-polarization data.

This script loads the pretrained OptoGPT checkpoint (model/optogpt.pt),
reuses its vocabulary, and fine-tunes on the 60° s-pol dataset.

Usage:
    python finetune_60deg_s.py --data_dir ../data_60deg_s --epochs 5 --batch_size 16 --lr 3e-5
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent  # optogpt/optogpt/
sys.path.insert(0, str(PROJECT_ROOT))

from core.datasets.datasets import PrepareDataAug, PAD, UNK
from core.models.transformer import make_model_I
from core.trains.train import SimpleLossCompute, LabelSmoothing, NoamOpt, save_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint paths
PRETRAINED_CKPT = str((PROJECT_ROOT.parent / "model" / "optogpt.pt").resolve())
# Save fine-tuned models alongside original checkpoint (absolute path, avoid CWD dependency)
OUTPUT_DIR = PROJECT_ROOT.parent / "model"

# Fixed conditions
THETA_DEG = 60
POLARIZATION = "s"


class FinetuneOpt:
    """Simple optimizer wrapper with constant learning rate for fine-tuning."""
    def __init__(self, optimizer, lr=3e-5):
        self.optimizer = optimizer
        self._step = 0
        self._lr = lr

    def step(self):
        self._step += 1
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self):
        return self._lr


def run_epoch(data_iter, model, loss_compute, device, desc="Train"):
    """Run one epoch. Returns average loss."""
    total_tokens = 0.0
    total_loss = 0.0
    start = time.time()

    for i, batch in enumerate(data_iter):
        out = model(
            batch.src.to(device),
            batch.trg.to(device),
            batch.src_mask,
            batch.trg_mask.to(device) if batch.trg_mask is not None else None,
        )
        loss = loss_compute(out, batch.trg_y.to(device), batch.ntokens.to(device))
        total_loss += loss
        total_tokens += batch.ntokens

        if i % 20 == 0:
            elapsed = time.time() - start
            rate = total_tokens.float() / elapsed if elapsed > 0 else 0
            print(f"  {desc} Batch {i:4d}: loss={loss / batch.ntokens:.4f}, "
                  f"tokens/sec={rate:.1f}")

    avg_loss = total_loss / total_tokens
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Fine-tune OptoGPT on 60° s-pol data")
    parser.add_argument("--data_dir", type=str, default="../data_60deg_s",
                        help="Directory with generated 60° s-pol data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (adjust for GPU memory)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_name", type=str, default="optogpt_60deg_s",
                        help="Base name for saved checkpoints")
    parser.add_argument("--early_stopping", action="store_true", default=False,
                        help="Enable early stopping based on dev loss")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping (epochs without improvement)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from _latest.pt checkpoint")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained checkpoint (default: model/optogpt.pt)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Data dir: {args.data_dir}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    # ---- Determine if resuming ----
    latest_path = str(OUTPUT_DIR / f"{args.output_name}_latest.pt")
    start_epoch = 0
    loss_all = {"train_loss": [], "dev_loss": []}
    best_dev_loss = float("inf")
    patience_counter = 0

    if args.resume:
        if not os.path.exists(latest_path):
            print(f"ERROR: Resume checkpoint not found: {latest_path}")
            print(f"  Run without --resume to start fresh.")
            sys.exit(1)

        print(f"\nResuming from: {latest_path}")
        resume_ckpt = torch.load(latest_path, map_location=DEVICE, weights_only=False)
        resume_configs = resume_ckpt["configs"]
        word_dict = resume_configs.get("struc_word_dict", {})
        index_dict = resume_configs.get("struc_index_dict", {})

        if not word_dict:
            print("ERROR: Resume checkpoint missing vocabulary. Cannot resume.")
            sys.exit(1)

        start_epoch = resume_ckpt.get("epoch", 0)
        loss_all = resume_ckpt.get("loss_all", {"train_loss": [], "dev_loss": []})
        best_dev_loss = resume_configs.get("best_dev_loss", float("inf"))

        # Recalculate patience_counter from loss history
        patience_counter = 0
        if args.early_stopping and len(loss_all["dev_loss"]) > 0:
            best_so_far = min(loss_all["dev_loss"])
            for dl in reversed(loss_all["dev_loss"]):
                if dl > best_so_far:
                    patience_counter += 1
                else:
                    break

        print(f"  Resuming from epoch {start_epoch} (target: {args.epochs})")
        print(f"  Train loss history: {len(loss_all['train_loss'])} epochs")
        print(f"  Best dev loss so far: {best_dev_loss:.4f}")
        print(f"  Patience counter: {patience_counter}/{args.patience}")
        print(f"  Vocab size: {len(word_dict)}")

        # Build model from resume config
        model = make_model_I(
            src_vocab=resume_configs.get("spec_dim", 142),
            tgt_vocab=resume_configs.get("struc_dim", len(word_dict)),
            N=resume_configs.get("layers", 6),
            d_model=resume_configs.get("d_model", 1024),
            d_ff=resume_configs.get("d_ff", 512),
            h=resume_configs.get("head_num", 8),
            dropout=resume_configs.get("dropout", 0.1),
        ).to(DEVICE)

        model.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Rebuild optimizer and load state
        base_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                           betas=(0.9, 0.98), eps=1e-9)
        optimizer = FinetuneOpt(base_optimizer, lr=args.lr)
        if resume_ckpt.get("optimizer_state_dict") is not None:
            optimizer.optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            print("  Optimizer state restored.")

        train_config = dict(resume_configs)
        train_config["epochs"] = args.epochs
        train_config["resumed_at"] = datetime.now().isoformat()
    else:
        # ---- Load checkpoint (fresh start) ----
        pretrained_source = args.pretrained if args.pretrained else PRETRAINED_CKPT
        print(f"\nLoading pretrained model: {pretrained_source}")
        ckpt = torch.load(str(pretrained_source), map_location=DEVICE, weights_only=False)
        configs = ckpt["configs"]
        word_dict = configs.struc_word_dict if hasattr(configs, 'struc_word_dict') else configs.get('struc_word_dict', {})
        index_dict = configs.struc_index_dict if hasattr(configs, 'struc_index_dict') else configs.get('struc_index_dict', {})

        print(f"  Vocab size: {len(word_dict)}")
        # Helper for dict/SimpleNamespace compatibility
        def _cfg(key, default=None):
            if isinstance(configs, dict):
                return configs.get(key, default)
            return getattr(configs, key, default)

        print(f"  PAD={word_dict.get('PAD')}, UNK={word_dict.get('UNK')}, "
              f"BOS={word_dict.get('BOS')}, EOS={word_dict.get('EOS')}")
        print(f"  Model: layers={_cfg('layers', 6)}, d_model={_cfg('d_model', 1024)}, "
              f"d_ff={_cfg('d_ff', 512)}, heads={_cfg('head_num', 8)}")

        # ---- Build model from checkpoint config ----
        model = make_model_I(
            src_vocab=_cfg('spec_dim', 142),
            tgt_vocab=_cfg('struc_dim', len(word_dict)),
            N=_cfg('layers', 6),
            d_model=_cfg('d_model', 1024),
            d_ff=_cfg('d_ff', 512),
            h=_cfg('head_num', 8),
            dropout=_cfg('dropout', 0.1),
        ).to(DEVICE)

        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # ---- Setup optimizer (fresh) ----
        base_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                           betas=(0.9, 0.98), eps=1e-9)
        optimizer = FinetuneOpt(base_optimizer, lr=args.lr)

        # ---- Build training config ----
        train_config = {
            "description": f"OptoGPT fine-tuned on {THETA_DEG}° {POLARIZATION}-polarization data",
            "pretrained_checkpoint": str(pretrained_source),
            "data_dir": str(Path(args.data_dir).resolve()),
            "theta_deg": THETA_DEG,
            "polarization": POLARIZATION,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "label_smoothing": args.smoothing,
            "seed": args.seed,
            "device": str(DEVICE),
            "started_at": datetime.now().isoformat(),
            "early_stopping": args.early_stopping,
            "patience": args.patience,
            "spec_dim": _cfg('spec_dim', 142),
            "struc_dim": _cfg('struc_dim', 904),
            "layers": _cfg('layers', 6),
            "d_model": _cfg('d_model', 1024),
            "d_ff": _cfg('d_ff', 512),
            "head_num": _cfg('head_num', 8),
            "dropout": _cfg('dropout', 0.1),
            "struc_word_dict": word_dict,
            "struc_index_dict": index_dict,
        }

    # ---- Load 60° s-pol data ----
    data_dir = Path(args.data_dir)
    print(f"\nLoading data from {data_dir}")

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
    )
    print(f"  Train batches: {len(data.train_data)}, Dev batches: {len(data.dev_data)}")

    # ---- Setup loss criterion ----
    criterion = LabelSmoothing(len(word_dict), padding_idx=PAD, smoothing=args.smoothing)

    # ---- Training loop ----
    best_epoch = start_epoch if start_epoch > 0 else 0

    print(f"\n{'='*60}")
    if args.resume:
        print(f"Resuming fine-tuning: epoch {start_epoch+1} to {args.epochs}")
    else:
        print(f"Starting fine-tuning: {args.epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        train_loss = run_epoch(
            data.train_data,
            model,
            SimpleLossCompute(model.generator, criterion, optimizer),
            DEVICE,
            desc=f"Train E{epoch+1}",
        )
        loss_all["train_loss"].append(float(train_loss))

        # ---- Validate (no gradients!) ----
        model.eval()
        with torch.no_grad():
            dev_loss = run_epoch(
                data.dev_data,
                model,
                SimpleLossCompute(model.generator, criterion, opt=None),
                DEVICE,
                desc=f"Val  E{epoch+1}",
            )
        loss_all["dev_loss"].append(float(dev_loss))

        elapsed = time.time() - epoch_start
        print(f"\n  Epoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, dev_loss={dev_loss:.4f}, "
              f"time={elapsed:.1f}s")

        # ---- Save checkpoints ----
        train_config["epoch"] = epoch + 1
        train_config["train_loss"] = float(train_loss)
        train_config["dev_loss"] = float(dev_loss)
        train_config["best_dev_loss"] = float(best_dev_loss)

        # Always save latest
        latest_path = str(OUTPUT_DIR / f"{args.output_name}_latest.pt")
        save_checkpoint(model, optimizer, epoch + 1, loss_all, latest_path, train_config)
        print(f"  Saved latest: {latest_path}")

        # Save best
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_path = str(OUTPUT_DIR / f"{args.output_name}_best.pt")
            save_checkpoint(model, optimizer, epoch + 1, loss_all, best_path, train_config)
            print(f"  >>> New best! Saved: {best_path}")
        else:
            patience_counter += 1

        # Early stopping
        if args.early_stopping and patience_counter >= args.patience:
            print(f"\n  Early stopping triggered after {args.patience} epochs without improvement.")
            print(f"  Best dev loss: {best_dev_loss:.4f} at epoch {best_epoch}")
            break

    train_config["finished_at"] = datetime.now().isoformat()
    train_config["final_train_loss"] = float(loss_all["train_loss"][-1])
    train_config["final_dev_loss"] = float(loss_all["dev_loss"][-1])
    train_config["best_dev_loss"] = float(best_dev_loss)
    train_config["best_epoch"] = best_epoch
    train_config["total_epochs_run"] = len(loss_all["train_loss"])

    # Save final config
    config_path = str(OUTPUT_DIR / f"{args.output_name}_config.json")
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=2, default=str)

    # Save loss history as standalone JSON for easy plotting
    loss_path = str(OUTPUT_DIR / f"{args.output_name}_loss_history.json")
    with open(loss_path, "w") as f:
        json.dump({
            "train_loss": loss_all["train_loss"],
            "dev_loss": loss_all["dev_loss"],
            "best_epoch": best_epoch,
            "best_dev_loss": float(best_dev_loss),
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Fine-tuning complete!")
    print(f"  Best dev loss: {best_dev_loss:.4f} (epoch {best_epoch})")
    print(f"  Final train loss: {loss_all['train_loss'][-1]:.4f}")
    print(f"  Final dev loss: {loss_all['dev_loss'][-1]:.4f}")
    print(f"  Best model: {OUTPUT_DIR / f'{args.output_name}_best.pt'}")
    print(f"  Config: {config_path}")
    print(f"  Loss history: {loss_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
