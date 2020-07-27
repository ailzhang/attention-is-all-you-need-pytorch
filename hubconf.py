from argparse import Namespace
import math
import time
import dill as pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from train import prepare_dataloaders, cal_performance, patch_src, patch_trg
import random
import numpy as np

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def skipIfNotImplemented(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except NotImplementedError:
            print('skipped since {} is not implemented'.format(func.__name__))
    return wrapper

class Model:
    def __init__(self, device=None, jit=False):
        self.device = device
        self.jit = jit
        self.opt = Namespace(**{
            'batch_size': 128,
            'd_inner_hid': 2048,
            'd_k': 64,
            'd_model': 512,
            'd_word_vec': 512,
            'd_v': 64,
            'data_pkl': 'm30k_deen_shr.pkl',
            'debug': '',
            'dropout': 0.1,
            'embs_share_weight': False,
            'epoch': 1,
            'label_smoothing': False,
            'log': None,
            'n_head': 8,
            'n_layers': 6,
            'n_warmup_steps': 128,
            'cuda': True,
            'proj_share_weight': False,
            'save_mode': 'best',
            'save_model': None,
            'script': False,
            'train_path': None,
            'val_path': None,
        })

    def get_module(self):
        _, validation_data = prepare_dataloaders(self.opt, self.device)
        transformer = Transformer(
            self.opt.src_vocab_size,
            self.opt.trg_vocab_size,
            src_pad_idx=self.opt.src_pad_idx,
            trg_pad_idx=self.opt.trg_pad_idx,
            trg_emb_prj_weight_sharing=self.opt.proj_share_weight,
            emb_src_trg_weight_sharing=self.opt.embs_share_weight,
            d_k=self.opt.d_k,
            d_v=self.opt.d_v,
            d_model=self.opt.d_model,
            d_word_vec=self.opt.d_word_vec,
            d_inner=self.opt.d_inner_hid,
            n_layers=self.opt.n_layers,
            n_head=self.opt.n_head,
            dropout=self.opt.dropout).to(self.device)

        if self.jit:
            transformer = torch.jit.script(transformer)
        batch = list(validation_data)[0]
        src_seq = patch_src(batch.src, self.opt.src_pad_idx).to(self.device)
        trg_seq, _ = map(lambda x: x.to(self.device), patch_trg(batch.trg, self.opt.trg_pad_idx))

        # We use validation_data for training as well so that it can finish fast enough.
        return transformer, (src_seq, trg_seq)

    @skipIfNotImplemented
    def eval(self, niter=1):
        m, example_inputs = self.get_module()
        m.eval()
        for _ in range(niter):
            m(*example_inputs)

    @skipIfNotImplemented
    def train(self, niter=1):
        m, _ = self.get_module()
        optimizer = ScheduledOptim(
            optim.Adam(m.parameters(), betas=(0.9, 0.98), eps=1e-09),
            2.0, self.opt.d_model, self.opt.n_warmup_steps)
        training_data, _ = prepare_dataloaders(self.opt, self.device)
        batch = list(training_data)[0]
        src_seq = patch_src(batch.src, self.opt.src_pad_idx).to(self.device)
        trg_seq, gold = map(lambda x: x.to(self.device), patch_trg(batch.trg, self.opt.trg_pad_idx))
        for _ in range(niter):
            optimizer.zero_grad()
            pred = m(src_seq, trg_seq)

            loss, n_correct, n_word = cal_performance(
                pred, gold, self.opt.trg_pad_idx, smoothing=self.opt.label_smoothing)
            loss.backward()
            optimizer.step_and_update_lr()


if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    model, example_inputs = m.get_module()
    model(*example_inputs)
    m.train()
    m.eval()
