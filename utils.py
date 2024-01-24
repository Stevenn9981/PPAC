import collections
import math
import os
import pdb

import numpy as np
import scipy
import torch
from scipy import stats
from tqdm import tqdm
import torch.nn.functional as F

TOP_Ks = [20, 50, 100]


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def RecallPrecision_ATk_cointer(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """

    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = right_pred / recall_n
    precis = right_pred / precis_n
    return {'recall': recall, 'precision': precis}


def NDCGatK_r_cointer(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return ndcg


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 2048)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def test_one_batch(X, topks, global_pop=None):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, arp = [], [], [], []
    for k in topks:
        ret = RecallPrecision_ATk_cointer(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r_cointer(groundTrue, r, k))
        if global_pop is not None:
            arp.append(global_pop[sorted_items[:, :k]].mean(1).tolist())
    ret = {'recall': recall,
           'precision': pre,
           'ndcg': ndcg}
    if global_pop is not None:
        ret['arp'] = arp
    return ret


def test_one_batch_for_global_novelty(rating, topks, global_nov, global_pop):
    nov = dict()
    pru = dict()
    for k in topks:
        topk_items = rating[:, :k]
        nov[k] = global_nov[topk_items].mean(1).tolist()
        pru[k] = [-stats.spearmanr(global_pop[topk_items][i], torch.arange(k))[0] for i in range(len(topk_items))]
    return nov, pru


def test_one_batch_local_novelty(rating, topks, local_nov, batch_users, local_pop):
    nov = dict()
    pru = dict()
    for k in topks:
        topk_items = rating[:, :k]
        nov[k] = torch.gather(local_nov[batch_users], 1, topk_items).mean(1).tolist()
        topk_pop = torch.gather(local_pop[batch_users], 1, topk_items)
        pru[k] = []
        for i in range(len(topk_items)):
            src = -stats.spearmanr(topk_pop[i], torch.arange(k))[0]
            if math.isnan(src):
                src = 0
            pru[k].append(src)
    return nov, pru


def cal_global_nov(train_records, num_items):
    pop = [0] * num_items
    nov = [1] * num_items
    for user in train_records:
        for item in train_records[user]:
            pop[item] += 1
    u_num = len(train_records)
    for i in range(num_items):
        if pop[i] > 0:
            nov[i] = -(1 / np.log2(u_num)) * np.log2(pop[i] / u_num)
    return torch.tensor(nov), torch.tensor(pop)


def cal_local_nov(dataset, sim_users, train_records, num_items):
    sim_coe = len(max(sim_users.values(), key=len))
    local_nov, local_pop = None, None
    if os.path.exists(f'co_items/local_nov-{dataset}-{sim_coe}.npy'):
        local_nov = np.load(f'co_items/local_nov-{dataset}-{sim_coe}.npy', allow_pickle=True)
    if os.path.exists(f'co_items/local_pop-{dataset}-{sim_coe}.npy'):
        local_pop = np.load(f'co_items/local_pop-{dataset}-{sim_coe}.npy', allow_pickle=True)
    if local_nov is None or local_pop is None:
        u_num = max(train_records.keys()) + 1
        local_nov = [[1] * num_items for _ in range(u_num)]
        local_pop = [[0] * num_items for _ in range(u_num)]
        train_item_records = collections.defaultdict(set)
        for user in train_records:
            for item in train_records[user]:
                train_item_records[item].add(user)
        for item in tqdm(train_item_records):
            for user in range(u_num):
                di = len(train_item_records[item].intersection(sim_users[user]))
                if di != 0:
                    local_pop[user][item] = di
                    local_nov[user][item] = -(1 / np.log2(len(sim_users[user]))) * np.log2(di / len(sim_users[user]))
        np.save(f'co_items/local_nov-{dataset}-{sim_coe}.npy', local_nov)
        np.save(f'co_items/local_pop-{dataset}-{sim_coe}.npy', local_pop)
    return torch.tensor(local_nov), torch.tensor(local_pop)


def test_model(model, train_records, test_records, global_pop, graph=None):
    model.eval()
    max_K = max(TOP_Ks)
    results = {'precision': dict(),
               'recall': dict(),
               'ndcg': dict(),
               'arp': dict()}
    for topk in TOP_Ks:
        results['precision'][topk] = []
        results['recall'][topk] = []
        results['ndcg'][topk] = []
        results['arp'][topk] = []
    with torch.no_grad():
        users = list(test_records.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        u_batch_size = 128 if hasattr(model, 'sys_params') and 'ncf' in model.sys_params.model else 8192
        total_batch = len(users) // u_batch_size + 1
        for batch_users in minibatch(users, batch_size=u_batch_size):
            users_list.append(batch_users)
            allPos = [train_records[u] for u in batch_users]
            groundTrue = [test_records[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(model.device)
            if graph:
                rating = model.get_user_ratings(batch_users_gpu, graph)
            else:
                rating = model.get_user_ratings(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -float('inf')
            _, rating_K = torch.topk(rating, k=max_K)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, TOP_Ks, global_pop))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            for idx, topk in enumerate(TOP_Ks):
                results['recall'][topk].extend(result['recall'][idx])
                results['precision'][topk].extend(result['precision'][idx])
                results['ndcg'][topk].extend(result['ndcg'][idx])
                results['arp'][topk].extend(result['arp'][idx])
        results['users'] = [user for user_lst in users_list for user in user_lst]
        return results


def test_cointer_model(model, train_records, sim_users, test_records, i_num, graph=None):
    model.eval()
    max_K = max(TOP_Ks)
    with torch.no_grad():
        users = list(test_records.keys())
        users_list = []
        rating_list = []
        batch_users_list = []
        u_batch_size = 128  # 8192
        total_batch = len(users) // u_batch_size + 1
        for batch_users in minibatch(users, batch_size=u_batch_size):
            users_list.append(batch_users)
            allPos = [train_records[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(model.device)
            if graph:
                rating = model.get_user_ratings(batch_users_gpu, graph)
            else:
                rating = model.get_user_ratings(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = 0
            _, rating_K = torch.topk(rating, k=max_K)
            rating_list.append(rating_K.cpu())
            batch_users_list.append(batch_users)
        assert total_batch == len(users_list)
        X = zip(rating_list, batch_users_list)
        GNov_results = collections.defaultdict(list)
        LNov_results = collections.defaultdict(list)
        GPRU_results = collections.defaultdict(list)
        LPRU_results = collections.defaultdict(list)
        global_nov, global_pop = cal_global_nov(train_records, i_num)
        local_nov, local_pop = cal_local_nov(model.dataset, sim_users, train_records, i_num)
        for rating, batch_users in X:
            glo_nov, glo_pru = test_one_batch_for_global_novelty(rating, TOP_Ks, global_nov, global_pop)
            loc_nov, loc_pru = test_one_batch_local_novelty(rating, TOP_Ks, local_nov, batch_users, local_pop)
            for topk in TOP_Ks:
                GNov_results[topk].extend(glo_nov[topk])
                LNov_results[topk].extend(loc_nov[topk])
                GPRU_results[topk].extend(glo_pru[topk])
                LPRU_results[topk].extend(loc_pru[topk])
        return GNov_results, LNov_results, GPRU_results, LPRU_results, global_pop, local_pop


def clear_test_file():
    # dataset = 'TaobaoAd'
    # domains = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6']
    dataset = 'ml-1M'
    domains = ['Arts', 'Inst', 'Music', 'Pantry', 'Video']
    for domain in domains:
        file = open(f'data/{dataset}/{domain}_train.csv', 'r')
        u_s = set()
        i_s = set()
        for line in file.readlines():
            ele = line.strip().split(',')
            u_s.add(ele[1])
            i_s.add(ele[0])
        file.close()
        for mode in ['val', 'test']:
            val_file = open(f'data/{dataset}/{domain}_{mode}.csv', 'r')
            val_lines = []
            for line in val_file.readlines():
                ele = line.strip().split(',')
                if ele[1] in u_s and ele[0] in i_s:
                    val_lines.append(line)
            val_file.close()
            val_file = open(f'data/{dataset}/{domain}_{mode}.csv', 'w')
            for line in val_lines:
                val_file.write(line)
            val_file.close()


def cal_p_val(array, popmean):
    t, p = scipy.stats.ttest_1samp(array, popmean)
    return p / 2
