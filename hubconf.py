import argparse
import math
import time
import pytest
import sys
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

    def _prepare_opt(self, args):
        parser = argparse.ArgumentParser()

        parser.add_argument('-data_pkl', default='m30k_deen_shr.pkl')     # all-in-1 data pickle or bpe field

        parser.add_argument('-train_path', default=None)   # bpe encoded data
        parser.add_argument('-val_path', default=None)     # bpe encoded data

        parser.add_argument('-epoch', type=int, default=1)
        parser.add_argument('-b', '--batch_size', type=int, default=128)

        parser.add_argument('-d_model', type=int, default=512)
        parser.add_argument('-d_inner_hid', type=int, default=2048)
        parser.add_argument('-d_k', type=int, default=64)
        parser.add_argument('-d_v', type=int, default=64)

        parser.add_argument('-n_head', type=int, default=8)
        parser.add_argument('-n_layers', type=int, default=6)
        parser.add_argument('-warmup','--n_warmup_steps', type=int, default=128)

        parser.add_argument('-dropout', type=float, default=0.1)
        parser.add_argument('-embs_share_weight', action='store_true')
        parser.add_argument('-proj_share_weight', action='store_true')

        parser.add_argument('-log', default=None)
        parser.add_argument('-save_model', default=None)
        parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

        parser.add_argument('-no_cuda', action='store_true')
        parser.add_argument('-label_smoothing', action='store_true')
        parser.add_argument('--debug', metavar='fn', default="", help="Dump outputs into file")
        parser.add_argument('--script', default=False, help="Script the model")

        self.opt = parser.parse_args(args)
        self.opt.cuda = not self.opt.no_cuda
        self.opt.d_word_vec = self.opt.d_model

    def get_module(self, args=None):
        self._prepare_opt(args)
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
    def eval(self, niter=1, args=None):
        m, example_inputs = self.get_module(args)
        m.eval()
        for _ in range(niter):
            m(*example_inputs)

    @skipIfNotImplemented
    def train(self, niter=1, args=None):
        m, _ = self.get_module(args)
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


@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=1,
    disable_gc=True,
    max_time=0.1,
    min_rounds=3,
)
class TestBenchNetwork:
    m = Model(device='cuda', jit=False)
    model, example_inputs = m.get_module(args=[])

    def test_train(self, benchmark):
        benchmark(self.m.train, args=[])

    def test_eval(self, benchmark):
        benchmark(self.m.eval, args=[])

if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    model, example_inputs = m.get_module()
    model(*example_inputs)
    m.train()
    m.eval()
