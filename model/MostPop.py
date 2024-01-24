import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import cal_global_nov, cal_local_nov


class MostPop(nn.Module):
    def __init__(self, train_records, params, sys_params, sim_users=None):
        super(MostPop, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.device = params['device']
        self.dataset = params['dataset']
        self.sys_params = sys_params
        self.global_pop = cal_global_nov(train_records, self.num_items)[1].to(self.device)
        self.local_pop = cal_local_nov(self.dataset, sim_users, train_records, self.num_items)[1].to(self.device)
        self.global_pop = F.normalize(self.global_pop.float(), dim=0)
        self.local_pop = F.normalize(self.local_pop.float(), dim=1)

    def get_user_ratings(self, user_indices):
        if self.sys_params.model == 'mostpop':
            return self.global_pop.repeat(user_indices.shape[0], 1)
        else:
            return self.local_pop[user_indices]
