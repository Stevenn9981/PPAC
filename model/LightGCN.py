import torch
from dgl.nn.pytorch import GraphConv
from torch import nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, params, sys_params):
        super(LightGCN, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.device = params['device']
        self.dataset = params['dataset']
        self.latent_dim = 64
        self.n_layers = 3
        self.f = nn.Sigmoid()
        if sys_params.rAdj:
            self.conv = GraphConv(64, 64, weight=False, bias=False, norm='right', allow_zero_in_degree=True)
            if sys_params.gamma == 100:
                self.conv = GraphConv(64, 64, weight=False, bias=False, norm='left', allow_zero_in_degree=True)
        else:
            self.conv = GraphConv(64, 64, weight=False, bias=False, allow_zero_in_degree=True)
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def computer(self, graph):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        layer_emb = torch.cat([users_emb, items_emb])
        embs = [layer_emb]
        for layer in range(self.n_layers):
            layer_emb = self.conv(graph, layer_emb)
            embs.append(layer_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def get_user_ratings(self, users, graph):
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users.long()]
        rating = self.f(torch.matmul(users_emb, all_items.T))
        return rating

    def getEmbedding(self, graph, users, pos_items=None, neg_items=None):
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users]

        if neg_items is None:
            return users_emb, all_items, self.user_embedding(users), self.item_embedding.weight

        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.user_embedding(users)
        pos_emb_ego = self.item_embedding(pos_items)
        neg_emb_ego = self.item_embedding(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, graph, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(graph, users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        return loss, reg_loss
