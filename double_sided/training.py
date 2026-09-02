"""Training utilities that preserve the explicit double-sided token contract."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from optogpt.core.models.transformer import subsequent_mask
from .contract import DoubleSidedStructure


class DoubleSidedDataset(Dataset):
    def __init__(self, data_dir, split, word_dict, allowed_materials, max_layers_per_side):
        root = Path(data_dir)
        spectra = np.load(root / f"spectra_ABC_{split}.npz")["C"]
        if spectra.ndim != 2 or spectra.shape[1] != 284:
            raise ValueError(f"Invalid {split} spectrum contract")
        self.spectra, self.sequences = spectra.astype(np.float32), []
        with (root / f"structures_{split}.jsonl").open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                tokens = json.loads(line)["tokens"]
                DoubleSidedStructure.from_tokens(tokens, allowed_materials, max_layers_per_side)
                missing = [token for token in tokens if token not in word_dict]
                if missing:
                    raise ValueError(f"{split}[{index}] contains out-of-vocabulary tokens: {missing}")
                sequence = [int(word_dict[token]) for token in tokens]
                if word_dict["UNK"] in sequence:
                    raise ValueError(f"{split}[{index}] maps to UNK")
                self.sequences.append(sequence)
        if len(self.sequences) != len(self.spectra):
            raise ValueError(f"Invalid {split} structure/spectrum count")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.spectra[index], self.sequences[index]


def collate_double_sided(batch, pad_id):
    spectra, sequences = zip(*batch)
    maximum = max(len(sequence) for sequence in sequences)
    padded = np.full((len(sequences), maximum), pad_id, dtype=np.int64)
    for row, sequence in enumerate(sequences):
        padded[row, :len(sequence)] = sequence
    target = torch.from_numpy(padded)
    decoder_input, target_y = target[:, :-1], target[:, 1:]
    target_mask = (decoder_input != pad_id).unsqueeze(-2)
    target_mask = target_mask & subsequent_mask(decoder_input.size(1)).type_as(target_mask)
    return {
        "spectrum": torch.from_numpy(np.asarray(spectra, dtype=np.float32)),
        "decoder_input": decoder_input, "target_y": target_y,
        "target_mask": target_mask,
        "ntokens": (target_y != pad_id).sum(),
    }


def make_loaders(data_dir, word_dict, allowed_materials, max_layers_per_side,
                 batch_size, seed, num_workers=0):
    generator = torch.Generator().manual_seed(seed)
    datasets = {
        split: DoubleSidedDataset(
            data_dir, split, word_dict, allowed_materials, max_layers_per_side
        ) for split in ("train", "dev")
    }
    if not datasets["train"] or not datasets["dev"]:
        raise ValueError("Train and dev datasets must both be non-empty")
    collate = lambda batch: collate_double_sided(batch, word_dict["PAD"])
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            generator=generator, num_workers=num_workers, collate_fn=collate),
        "dev": DataLoader(datasets["dev"], batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=collate),
    }


def build_output_mask(word_dict, allowed_materials):
    """Allow only physical base-material layers, EOS, and SIDE_SEP as targets."""
    allowed_materials = set(allowed_materials)
    mask = torch.zeros(len(word_dict), dtype=torch.bool)
    for token, index in word_dict.items():
        if token in ("EOS", "SIDE_SEP"):
            mask[index] = True
            continue
        if "_" not in token:
            continue
        material, thickness = token.rsplit("_", 1)
        try:
            value = float(thickness)
        except ValueError:
            continue
        if material in allowed_materials and np.isfinite(value) and value > 0:
            mask[index] = True
    if int(mask.sum()) != len(allowed_materials) * 50 + 2:
        raise ValueError("Allowed output vocabulary does not match 10 materials x 50 thicknesses + EOS/SEP")
    return mask


def masked_label_smoothed_loss(log_probabilities, targets, pad_id, smoothing, output_mask):
    flat_log = log_probabilities.reshape(-1, log_probabilities.size(-1))
    flat_target = targets.reshape(-1)
    active = flat_target != pad_id
    if not bool(active.any()):
        return flat_log.sum() * 0.0
    selected_log = flat_log[active]
    selected_target = flat_target[active]
    nll = -selected_log.gather(1, selected_target.unsqueeze(1)).squeeze(1)
    smooth = -selected_log[:, output_mask].mean(dim=1)
    return ((1.0 - smoothing) * nll + smoothing * smooth).sum()


def run_epoch(model, loader, criterion, device, optimizer=None, output_mask=None,
              smoothing=0.0, scaler=None, amp_enabled=False, grad_accum_steps=1):
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive")
    if training:
        optimizer.zero_grad(set_to_none=True)
    output_mask_device = output_mask.to(device) if output_mask is not None else None
    for batch_index, batch in enumerate(loader):
        if training and batch_index % grad_accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        spectrum = batch["spectrum"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        target_y = batch["target_y"].to(device)
        target_mask = batch["target_mask"].to(device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            hidden = model(spectrum, decoder_input, None, target_mask)
            if output_mask_device is None:
                log_probabilities = model.generator(hidden)
            else:
                raw_logits = model.generator.proj(hidden)
                raw_logits = raw_logits.masked_fill(~output_mask_device, float("-inf"))
                log_probabilities = F.log_softmax(raw_logits, dim=-1)
            if smoothing > 0 and output_mask_device is not None:
                loss_sum = masked_label_smoothed_loss(
                    log_probabilities, target_y, criterion.ignore_index,
                    smoothing, output_mask_device,
                )
            else:
                loss_sum = criterion(
                    log_probabilities.reshape(-1, log_probabilities.size(-1)), target_y.reshape(-1)
                )
        if training:
            normalized = loss_sum / batch["ntokens"].to(device) / grad_accum_steps
            if scaler is not None and scaler.is_enabled():
                scaler.scale(normalized).backward()
            else:
                normalized.backward()
            should_step = ((batch_index + 1) % grad_accum_steps == 0 or
                           batch_index + 1 == len(loader))
            if should_step:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=1.0,
                )
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
        total_loss += float(loss_sum.detach().cpu())
        total_tokens += int(batch["ntokens"])
    return total_loss / max(1, total_tokens)
