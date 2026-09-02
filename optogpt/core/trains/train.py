import os
import math
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from pathlib import Path
from torch.autograd import Variable
import pickle as pkl
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist



class LabelSmoothing(nn.Module):
    "Implement label smoothing with optional allowed-token mask."
    def __init__(self, size, padding_idx, smoothing=0.0, allowed_mask=None):
        """
        Args:
            size: vocab size
            padding_idx: PAD token id (always zeroed)
            smoothing: label smoothing factor in [0, 1]
            allowed_mask: (size,) boolean tensor or None.
                If provided, smoothing mass is ONLY distributed to True positions.
                This prevents banned/thickness-violating tokens from receiving probability.
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        if allowed_mask is not None:
            self.register_buffer('allowed_mask', allowed_mask.to(torch.bool))
        else:
            self.allowed_mask = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()

        if self.allowed_mask is not None:
            # Only distribute smoothing to allowed tokens
            n_allowed = self.allowed_mask.sum().item()
            if n_allowed > 0:
                true_dist.fill_(0)
                true_dist[:, self.allowed_mask] = self.smoothing / n_allowed
                # Zero PAD (already excluded if not in allowed_mask, but be safe)
                true_dist[:, self.padding_idx] = 0
            else:
                true_dist.fill_(0)
        else:
            true_dist.fill_(self.smoothing / (self.size - 2))

        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        if self.opt is not None:
            # Support both wrapped (NoamOpt, FinetuneOpt) and raw optimizers
            if hasattr(self.opt, 'zero_grad'):
                self.opt.zero_grad()
            elif hasattr(self.opt, 'optimizer') and hasattr(self.opt.optimizer, 'zero_grad'):
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()  # fallback for raw torch.optim
            loss.backward()
            self.opt.step()
        return loss.data.item() * norm.float()

# We used factor=2, warmup-step = 4000
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def run_epoch(data, model, criterion, optimizer, epoch, DEVICE):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    for i , batch in enumerate(data):
        out = model(batch.src.to(DEVICE),  batch.src_mask.to(DEVICE))
        # print(out.size(), batch.trg.size())
        loss = criterion(out, batch.trg.to(DEVICE))

        if optimizer is not None:
            if hasattr(optimizer, 'zero_grad'):
                optimizer.zero_grad()
            elif hasattr(optimizer, 'optimizer') and hasattr(optimizer.optimizer, 'zero_grad'):
                optimizer.optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch {:d} Batch: {:d} Loss: {:.4f} Tokens per Sec: {:.2f}s".format(epoch, i - 1, loss, (tokens.float() / elapsed)))
            start = time.time()
            tokens = 0
        del out, loss
    print(total_loss, i)
    return total_loss/i

def count_params(model):

    return sum([np.prod(layer.size()) for layer in model.parameters() if layer.requires_grad])

def save_checkpoint(model, optimizer, epoch, loss_all, path, configs):
    # save the checkpoint safely and create directories if needed
    save_dir = Path(path).parent.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    optimizer_state = None
    if optimizer is not None:
        if hasattr(optimizer, 'optimizer'):
            optimizer_state = optimizer.optimizer.state_dict()
        elif hasattr(optimizer, 'state_dict'):
            optimizer_state = optimizer.state_dict()
        else:
            optimizer_state = optimizer
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_state,
            'loss_all':loss_all,
            'configs':configs,
            # 'seed':seed,
        }, path)


def train(data, model, criterion, optimizer, configs, DEVICE):
    """
    Train and Save the model.
    """
    # init loss as a large value
    best_dev_loss = 1e5
    loss_all = {'train_loss':[], 'dev_loss':[]}

    save_folder = configs.save_folder
    save_name = configs.save_name
    EPOCHS = configs.epochs

    for epoch in range(EPOCHS):
        # Train model 
        model.train()
        train_loss = run_epoch(data.train_data, model, criterion, optimizer, epoch, DEVICE)

        # validate model on dev dataset

        model.eval()
        print('>>>>> Evaluate')
        with torch.no_grad():
            dev_loss = run_epoch(data.dev_data, model, criterion, None, epoch, DEVICE)
        print('<<<<< Evaluate loss: {:.8f}'.format(dev_loss))
        loss_all['train_loss'].append(train_loss.detach())
        loss_all['dev_loss'].append(dev_loss.detach())

        # save the model with best-dev-loss
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_checkpoint(model, optimizer, epoch, loss_all, 'saved_models/ol_transformer/'+save_folder+'/'+save_name+'_best.pt',  configs)
            print('Saved')
        if epoch%2 == 1:
            best_dev_loss = dev_loss
            save_checkpoint(model, optimizer, epoch, loss_all, 'saved_models/ol_transformer/'+save_folder+'/'+save_name+'_recent.pt',  configs)
            
        print(f">>>>> current best loss: ", best_dev_loss)


def run_epoch_I(data, model, loss_compute, epoch, DEVICE):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    for i , batch in enumerate(data):
        out = model(batch.src.to(DEVICE), batch.trg.to(DEVICE), batch.src_mask, batch.trg_mask.to(DEVICE))
        loss = loss_compute(out, batch.trg_y.to(DEVICE), batch.ntokens.to(DEVICE))
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch {:d} Batch: {:d} Loss: {:.4f} Tokens per Sec: {:.2f}s".format(epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed )))
            start = time.time()
            tokens = 0
        del out, loss

    return total_loss / total_tokens

    
def train_I(data, model, criterion, optimizer, configs, DEVICE):
    """
    Train and Save the model.
    """
    # init loss as a large value
    best_dev_loss = 1e5
    loss_all = {'train_loss':[], 'dev_loss':[]}

    save_folder = configs.save_folder
    save_name = configs.save_name
    EPOCHS = configs.epochs

    for epoch in range(EPOCHS):
        # Train model 
        model.train()
        train_loss = run_epoch_I(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch, DEVICE)
        model.eval()

        # validate model on dev dataset
        print('>>>>> Evaluate')
        with torch.no_grad():
            dev_loss = run_epoch_I(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch, DEVICE)
        print('<<<<< Evaluate loss: {:.2f}'.format(dev_loss))
        loss_all['train_loss'].append(train_loss.detach())
        loss_all['dev_loss'].append(dev_loss.detach())
        
        # save the model with best-dev-loss

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_checkpoint(model, optimizer, epoch, loss_all, 'saved_models/optogpt/'+save_folder+'/'+save_name+'_best.pt',  configs)

        save_checkpoint(model, optimizer, epoch, loss_all, 'saved_models/optogpt/'+save_folder+'/'+save_name+'_recent.pt',  configs)
            
        print(f">>>>> current best loss: {best_dev_loss}")
        
