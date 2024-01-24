import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.LightGCN import LightGCN
from model.MF import BPRMF
from utils import cal_global_nov, cal_local_nov


def cal_l2_loss(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def create_train_graph(train_records, params):
    src = []
    dst = []
    u_i_pairs = set()
    for uid in train_records:
        iids = train_records[uid]
        for iid in iids:
            if (uid, iid) not in u_i_pairs:
                src.append(int(uid))
                dst.append(int(iid))
                u_i_pairs.add((uid, iid))
    u_num, i_num = params['num_users'], params['num_items']
    src_ids = torch.tensor(src)
    dst_ids = torch.tensor(dst) + u_num
    g = dgl.graph((src_ids, dst_ids), num_nodes=u_num + i_num)
    g = dgl.to_bidirected(g)
    return g


def batch(x, bs):
    x = list(range(x))
    return [x[i:i + bs] for i in range(0, len(x), bs)]


class PPAC_MF(BPRMF):
    def __init__(self, train_records, params, sys_params, sim_users=None):
        super(PPAC_MF, self).__init__(params, sys_params)

        self.ablation = sys_params.ablation
        self.global_pop = cal_global_nov(train_records, self.num_items)[1].cpu()
        self.local_pop = cal_local_nov(self.dataset, sim_users, train_records, self.num_items)[1].cpu()
        self.global_pop = F.normalize(self.global_pop.float(), dim=0)
        self.local_pop = F.normalize(self.local_pop.float(), dim=1)
        self.gamma = sys_params.gamma
        self.beta = sys_params.beta
        self.local_pred = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim // 2, self.latent_dim)
        )
        self.global_pred = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim // 2, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)
        )
        self.reg_loss_fn = nn.MSELoss()
        self.f = nn.Sigmoid()
        self.l2_coe = 1e-4
        self.reg_coe = 1e-3

    def forward(self, user_indices, pos_item_indices, neg_item_indices):
        user_vec = self.user_embedding(user_indices)
        pos_item_vec = self.item_embedding(pos_item_indices)
        neg_item_vec = self.item_embedding(neg_item_indices)
        if 'mf' in self.sys_params.model:
            pos_scores = torch.mul(user_vec, pos_item_vec).sum(dim=1)
            neg_scores = torch.mul(user_vec, neg_item_vec).sum(dim=1)
        else:
            pos_scores = self.mlp(torch.cat([user_vec, pos_item_vec], dim=1))
            neg_scores = self.mlp(torch.cat([user_vec, neg_item_vec], dim=1))
        usr_ci_emb = self.local_pred(user_vec)
        pos_ci_emb = self.local_pred(pos_item_vec)
        neg_ci_emb = self.local_pred(neg_item_vec)

        pos_ci_local = self.f(torch.mul(usr_ci_emb, pos_ci_emb).sum(1))
        neg_ci_local = self.f(torch.mul(usr_ci_emb, neg_ci_emb).sum(1))
        pos_ci_global = self.global_pred(pos_item_vec)
        neg_ci_global = self.global_pred(neg_item_vec)

        pos_scores = pos_scores * (pos_ci_local * pos_ci_global)
        neg_scores = neg_scores * (neg_ci_local * neg_ci_global)

        cf_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        local_reg_loss = (self.reg_loss_fn(pos_ci_local,
                                           self.local_pop[user_indices.cpu(), pos_item_indices.cpu()].to(self.device)) +
                          self.reg_loss_fn(neg_ci_local,
                                           self.local_pop[user_indices.cpu(), neg_item_indices.cpu()].to(
                                               self.device))) / 2

        global_reg_loss = (self.reg_loss_fn(pos_ci_global, self.global_pop[pos_item_indices.cpu()].to(self.device)) +
                           self.reg_loss_fn(neg_ci_global, self.global_pop[neg_item_indices.cpu()].to(self.device))) / 2

        reg_loss = local_reg_loss + global_reg_loss

        l2_loss = cal_l2_loss(user_vec) + cal_l2_loss(pos_item_vec) + cal_l2_loss(neg_item_vec)

        return cf_loss + reg_loss * self.reg_coe + 1e-4 * l2_loss

    def get_user_ratings(self, user_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding.weight
        scores = super().get_user_ratings(user_indices)

        user_ci_emb = self.local_pred(user_vec)
        item_ci_emb = self.local_pred(item_vec)

        pred_local = self.f(torch.matmul(user_ci_emb, item_ci_emb.T))
        pred_global = self.global_pred(item_vec).expand(scores.shape)
        # scores = scores * (pred_local * pred_global) - self.gamma * (pred_local + pred_global)

        real_local = self.local_pop[user_indices.cpu()].to(self.device)
        real_global = self.global_pop.expand(scores.shape).to(self.device)

        scores = scores * (pred_local * pred_global) + self.gamma * real_local + self.beta * real_global

        return scores


class PPAC_LG(LightGCN):
    def __init__(self, train_records, params, sys_params, sim_users=None):
        super(PPAC_LG, self).__init__(params, sys_params)
        self.sys_params = sys_params
        self.gamma = sys_params.gamma
        self.beta = sys_params.beta
        self.ablation = sys_params.ablation
        self.local_pred = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim)
        )
        self.global_pred = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 2, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)
        )
        # self.trans_w = torch.randn((self.latent_dim, self.latent_dim))
        # nn.init.normal_(self.trans_w, std=0.1)
        self.global_pop = cal_global_nov(train_records, self.num_items)[1].cpu()
        self.local_pop = cal_local_nov(self.dataset, sim_users, train_records, self.num_items)[1].cpu()
        self.global_pop = F.normalize(self.global_pop.float(), dim=0)
        self.local_pop = F.normalize(self.local_pop.float(), dim=1)

        self.reg_loss_fn = nn.MSELoss()
        self.f = nn.Sigmoid()
        self.l2_coe = 1e-4
        self.reg_coe = 0.001

    def bpr_loss(self, graph, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(graph, users.long(), pos.long(), neg.long())
        l2_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                             posEmb0.norm(2).pow(2) +
                             negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)

        usr_ci_emb = self.local_pred(userEmb0)
        pos_ci_emb = self.local_pred(posEmb0)
        neg_ci_emb = self.local_pred(negEmb0)

        pos_ci_local = self.f(torch.mul(usr_ci_emb, pos_ci_emb).sum(1))
        neg_ci_local = self.f(torch.mul(usr_ci_emb, neg_ci_emb).sum(1))
        pos_ci_global = self.global_pred(posEmb0)
        neg_ci_global = self.global_pred(negEmb0)

        pos_scores = pos_scores * (pos_ci_local * pos_ci_global)
        neg_scores = neg_scores * (neg_ci_local * neg_ci_global)

        cf_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        local_reg_loss = (self.reg_loss_fn(pos_ci_local, self.local_pop[users.cpu(), pos.cpu()].to(self.device)) +
                          self.reg_loss_fn(neg_ci_local, self.local_pop[users.cpu(), neg.cpu()].to(self.device))) / 2
        global_reg_loss = (self.reg_loss_fn(pos_ci_global, self.global_pop[pos.cpu()].to(self.device)) +
                           self.reg_loss_fn(neg_ci_global, self.global_pop[neg.cpu()].to(self.device))) / 2

        linreg_loss = local_reg_loss + global_reg_loss

        # l2_loss = cal_l2_loss(user_vec) + cal_l2_loss(pos_item_vec) + cal_l2_loss(neg_item_vec)

        return cf_loss + linreg_loss * self.reg_coe, l2_loss

    def get_user_ratings(self, users, graph):
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users.long()]
        scores = torch.matmul(users_emb, all_items.T)

        user_vec = self.user_embedding(users)
        item_vec = self.item_embedding.weight

        user_ci_emb = self.local_pred(user_vec)
        item_ci_emb = self.local_pred(item_vec)

        pred_local = self.f(torch.matmul(user_ci_emb, item_ci_emb.T))
        pred_global = self.global_pred(item_vec).expand(scores.shape)

        # scores = scores * (pair_local * pair_global) - self.gamma * (pair_local + pair_global)

        real_local = self.local_pop[users.cpu()].to(self.device)
        real_global = self.global_pop.expand(scores.shape).to(self.device)

        scores = scores * (pred_local * pred_global) + self.gamma * real_local + self.beta * real_global
        return scores