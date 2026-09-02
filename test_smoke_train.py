"""Quick smoke test: 1 epoch train + val + save on 60° s-pol data."""
import sys, os, torch, numpy as np, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'optogpt'))
from core.datasets.datasets import PrepareDataAug, PAD
from core.models.transformer import make_model_I
from core.trains.train import SimpleLossCompute, LabelSmoothing, save_checkpoint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = Path(__file__).resolve().parent
print(f'Device: {DEVICE}')

# Load ckpt
ckpt = torch.load(str(ROOT / 'model' / 'optogpt.pt'), map_location=DEVICE, weights_only=False)
configs = ckpt['configs']
word_dict = configs.struc_word_dict
index_dict = configs.struc_index_dict
PAD_ID = word_dict['PAD']
print(f'Vocab: {len(word_dict)}, PAD={PAD_ID}')

# Build model
model = make_model_I(configs.spec_dim, configs.struc_dim, configs.layers,
                     configs.d_model, configs.d_ff, configs.head_num,
                     configs.dropout).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
n = sum(p.numel() for p in model.parameters())
print(f'Params: {n:,}')

# Load data
data = PrepareDataAug(
    str(ROOT / 'data_60deg_s' / 'Structure_train.pkl'),
    str(ROOT / 'data_60deg_s' / 'Spectrum_train.pkl'),
    100,
    str(ROOT / 'data_60deg_s' / 'Structure_dev.pkl'),
    str(ROOT / 'data_60deg_s' / 'Spectrum_dev.pkl'),
    BATCH_SIZE=8, spec_type='R_T', if_inverse='Inverse',
    struct_word_dict=word_dict, struct_index_dict=index_dict)
print(f'Train batches: {len(data.train_data)}, Dev batches: {len(data.dev_data)}')

# Setup optimizer
criterion = LabelSmoothing(len(word_dict), padding_idx=PAD, smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, betas=(0.9, 0.98), eps=1e-9)

class SimpleOpt:
    def __init__(self, opt): self.opt = opt
    def step(self): self.opt.step()
    def zero_grad(self): self.opt.zero_grad()

opt = SimpleOpt(optimizer)

# ---- 1 epoch train ----
model.train()
t0 = time.time()
total_loss = 0.0
total_tok = 0.0
train_lc = SimpleLossCompute(model.generator, criterion, opt)
for i, batch in enumerate(data.train_data):
    out = model(batch.src.to(DEVICE), batch.trg.to(DEVICE),
                batch.src_mask, batch.trg_mask.to(DEVICE))
    loss = train_lc(out, batch.trg_y.to(DEVICE), batch.ntokens.to(DEVICE))
    total_loss += loss
    total_tok += batch.ntokens
    if i % 200 == 0:
        print(f'  Batch {i}: loss={loss/batch.ntokens:.4f}')
train_avg = total_loss / total_tok
print(f'Train done: avg_loss={train_avg:.4f}, time={time.time()-t0:.1f}s')

# ---- Validation ----
model.eval()
with torch.no_grad():
    val_loss = 0.0
    val_tok = 0.0
    val_lc = SimpleLossCompute(model.generator, criterion, opt=None)
    for batch in data.dev_data:
        out = model(batch.src.to(DEVICE), batch.trg.to(DEVICE),
                    batch.src_mask, batch.trg_mask.to(DEVICE))
        loss = val_lc(out, batch.trg_y.to(DEVICE), batch.ntokens.to(DEVICE))
        val_loss += loss
        val_tok += batch.ntokens
val_avg = val_loss / val_tok
print(f'Val: avg_loss={val_avg:.4f}')

# ---- Save checkpoint ----
os.makedirs(str(ROOT / 'model'), exist_ok=True)
train_config = {'theta_deg': 60, 'polarization': 's', 'lr': 3e-5,
                'batch_size': 8, 'pretrained': 'model/optogpt.pt'}
loss_all = {'train_loss': [float(train_avg)], 'dev_loss': [float(val_avg)]}
save_checkpoint(model, opt, 1, loss_all,
                str(ROOT / 'model' / 'optogpt_60deg_s_smoke.pt'), train_config)
print('Saved smoke checkpoint')
print('DONE')
