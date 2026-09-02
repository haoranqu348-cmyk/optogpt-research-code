"""Debug script to test CPU inference."""
import sys, os
import numpy as np
import torch

# Use the directory where this script is as base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # optogpt/
sys.path.insert(0, SCRIPT_DIR)  # optogpt/optogpt/

from core.models.transformer import make_model_I, subsequent_mask

DEVICE = torch.device("cpu")

# Load fine-tuned checkpoint
ft_path = os.path.join(PROJECT_DIR, "model", "optogpt_60deg_s_best.pt")
pt_path = os.path.join(PROJECT_DIR, "model", "optogpt.pt")

print(f"Loading fine-tuned: {ft_path}")
ckpt = torch.load(ft_path, map_location="cpu", weights_only=False)
print(f"Loading pretrained: {pt_path}")
p = torch.load(pt_path, map_location="cpu", weights_only=False)

wd = p["configs"].struc_word_dict
idict = p["configs"].struc_index_dict
print(f"Vocab size: {len(wd)}")
print(f"BOS={wd['BOS']}, EOS={wd['EOS']}, PAD={wd['PAD']}, UNK={wd['UNK']}")

model = make_model_I(142, len(wd), 6, 1024, 512, 8, 0.1).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Test inference
spec = np.random.RandomState(0).rand(142).astype(np.float32)
src = torch.tensor([spec], dtype=torch.float32).to(DEVICE)
bos_id = wd["BOS"]
eos_id = wd["EOS"]
pad_id = wd["PAD"]

ys = torch.ones(1, 1, dtype=torch.long).fill_(bos_id).to(DEVICE)
print(f"ys shape: {ys.shape}, dtype: {ys.dtype}")

max_len = 22
design = []
with torch.no_grad():
    for step in range(max_len - 1):
        trg_mask = subsequent_mask(ys.size(1))
        out = model(src, ys, None, trg_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        nw = next_word.item()
        print(f"  Step {step}: next_word={nw}, sym={idict.get(nw, '?')}")
        if nw == eos_id:
            break
        ys = torch.cat([ys, torch.tensor([[nw]], device=DEVICE)], dim=1)
        sym = idict.get(nw, "UNK")
        if sym not in ("UNK", "EOS", "BOS", "PAD"):
            design.append(sym)

print(f"\nDesign: {design}")
print("SUCCESS!")
