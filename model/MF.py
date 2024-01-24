import pdb

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import random

from utils import cal_global_nov


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class RateDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor, i_num):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        self.i_num = i_num

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], torch.randint(0, self.i_num, ())

    def __len__(self):
        return self.user_tensor.size(0)


class BPRMF(nn.Module):
    def __init__(self, params, sys_params):
        super(BPRMF, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.device = params['device']
        self.dataset = params['dataset']
        self.latent_dim = 64

        self.user_embedding = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.latent_dim)
        self.f = nn.Sigmoid()
        self.sys_params = sys_params
        if 'ncf' in sys_params.model:
            self.mlp = nn.Sequential(
                nn.Linear(self.latent_dim * 2, 4 * self.latent_dim),
                nn.ReLU(),
                nn.Linear(4 * self.latent_dim, 2 * self.latent_dim),
                nn.ReLU(),
                nn.Linear(2 * self.latent_dim, 1)
            )
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, user_indices, pos_item_indices, neg_item_indices):
        user_vec = self.user_embedding(user_indices)
        pos_item_vec = self.item_embedding(pos_item_indices)
        neg_item_vec = self.item_embedding(neg_item_indices)
        if 'mf' in self.sys_params.model:
            pos_scores = self.f(torch.mul(user_vec, pos_item_vec).sum(dim=1))
            neg_scores = self.f(torch.mul(user_vec, neg_item_vec).sum(dim=1))
        else:
            pos_scores = self.mlp(torch.cat([user_vec, pos_item_vec], dim=1))
            neg_scores = self.mlp(torch.cat([user_vec, neg_item_vec], dim=1))
        cf_loss = torch.mean((-1.0) * F.logsigmoid(pos_scores - neg_scores))
        reg_loss = _L2_loss_mean(user_vec) + _L2_loss_mean(pos_item_vec) + _L2_loss_mean(neg_item_vec)
        return cf_loss + 1e-4 * reg_loss

    def get_user_ratings(self, user_indices):
        if 'ncf' in self.sys_params.model:
            user_emb = self.user_embedding(user_indices).repeat_interleave(self.num_items, dim=0)
            item_emb = self.item_embedding.weight.repeat(user_indices.shape[0], 1)
            return self.mlp(torch.cat([user_emb, item_emb], dim=1)).reshape(-1, self.num_items)
        return torch.matmul(self.user_embedding(user_indices), self.item_embedding.weight.T)

