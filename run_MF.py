import argparse
import json

import time

import matplotlib.pyplot as plt

from PPAC import PPAC_MF
from model.MF import *
from utils import *
from utils import test_cointer_model

TOP_K = 30


def set_random_seed(seed):
    r"""Set the random seed for reproducibility.

    Args:
        seed (int, default=0): The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args(parser):
    parser.add_argument('--dataset', default='ml-1M', type=str, help='dataset')
    parser.add_argument('--model', default='ppa', type=str, help='model name')
    parser.add_argument('--sim_coe', default=30, type=int, help='#similar users')
    parser.add_argument('--gamma', default=0, type=float, help='coefficient of local popularity')
    parser.add_argument('--beta', default=0, type=float, help='coefficient of global popularity')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--imp_n_layers', default=5, type=int, help='#layers in IMPGCN')
    parser.add_argument('--imp_n_classes', default=3, type=int, help='#classes in IMPGCN')
    parser.add_argument('--bs', default=8192, type=int, help='batch size')
    parser.add_argument('--rAdj', action='store_true', help='whether use r-AdjNorm')
    parser.add_argument('--test_pru', action='store_true', help='if test pop bias')
    parser.add_argument('--ablation', default='none', type=str, help='local or global')
    parser.add_argument('--var', default='none', type=str, help='all_pred or all_real')
    parser.add_argument('--cpr', action='store_true', help='whether use cpr loss')
    parser.add_argument('--draw_aly', default='intro', type=str, help='for paper figure only')
    parser.add_argument('--ncf', action='store_true', help='whether use NCF for MF model')
    parser.add_argument('--debias_data', default=True, type=bool, help='whether use debiased test')
    parser.add_argument('--train', action='store_true', help='whether we train the model')

    sys_paras = parser.parse_args()

    return sys_paras


def run_init_exp(sys_paras):
    dataset = sys_paras.dataset
    similarity_coe = sys_paras.sim_coe
    test_records, train_records, device, i_num, params = read_dataset(dataset, sys_paras.debias_data)
    print(f'#Users: {len(train_records)}')
    print(f'avg #interactions per usr: {sum([len(train_records[u]) for u in train_records]) / len(train_records):.4f}')

    global_pop = F.normalize(cal_global_nov(train_records, i_num)[1].float(), dim=0).numpy()
    model = BPRMF(params, sys_paras).to(device)
    if sys_paras.train:
        train_model(dataset, device, i_num, test_records, train_records, model, sys_paras, global_pop)

    if os.path.exists(f'checkpoints/{sys_paras.sim_coe}-{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt'):
        model.load_state_dict(
            torch.load(f'checkpoints/{sys_paras.sim_coe}-{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt',
                       map_location=device),
            strict=False)
    elif os.path.exists(f'checkpoints/{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt'):
        model.load_state_dict(
            torch.load(f'checkpoints/{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt', map_location=device),
            strict=False)
    else:
        model.load_state_dict(
            torch.load(f'checkpoints/{sys_paras.model}-{dataset}.pt', map_location=device),
            strict=False)

    res = test_model(model, train_records, test_records, global_pop)
    sim_users = cal_sim_users(dataset, similarity_coe, train_records)
    GNov_results, LNov_results, GPRU_results, LPRU_results, global_pop, local_pop = test_cointer_model(model,
                                                                                                       train_records,
                                                                                                       sim_users,
                                                                                                       test_records,
                                                                                                       i_num)

    res['GTop'] = dict()
    res['LTop'] = dict()
    res['GCorrI'] = collections.defaultdict(list)
    res['LCorrI'] = collections.defaultdict(list)
    res['GCorrJ'] = collections.defaultdict(list)
    res['LCorrJ'] = collections.defaultdict(list)
    for i, topk in enumerate(TOP_Ks):
        print(f'Precision@{topk}: {sum(res["precision"][topk]) / len(res["precision"][topk]):.4f}; '
              f'Recall@{topk}: {sum(res["recall"][topk]) / len(res["recall"][topk]):.4f};  '
              f'NDCG@{topk}: {sum(res["ndcg"][topk]) / len(res["ndcg"][topk]):.4f};  '
              f'ARP@{topk}: {sum(res["arp"][topk]) / len(res["arp"][topk]):.4f};  '
              f'GNov@{topk}: {sum(GNov_results[topk]) / len(GNov_results[topk]):.4f}; '
              f'LNov@{topk}: {sum(LNov_results[topk]) / len(LNov_results[topk]):.4f}; '
              f'GPRU@{topk}: {sum(GPRU_results[topk]) / len(GPRU_results[topk]):.4f}; '
              f'LPRU@{topk}: {sum(LPRU_results[topk]) / len(LPRU_results[topk]):.4f} ')
        _, glo_topk_items = torch.topk(global_pop, k=topk)
        _, loc_topk_items = torch.topk(local_pop[res['users']], k=topk)
        res['GTop'][topk] = glo_topk_items.tolist()
        res['LTop'][topk] = loc_topk_items.tolist()
        for idx, test_u in enumerate(res['users']):
            inter_items = set(train_records[test_u])
            glo_items = set(res['GTop'][topk])
            loc_items = set(res['LTop'][topk][idx])
            res['GCorrI'][topk].append(len(inter_items.intersection(glo_items)) / len(inter_items))
            res['LCorrI'][topk].append(len(inter_items.intersection(loc_items)) / len(inter_items))
            res['GCorrJ'][topk].append(len(inter_items.intersection(glo_items)) / len(inter_items.union(glo_items)))
            res['LCorrJ'][topk].append(len(inter_items.intersection(loc_items)) / len(inter_items.union(loc_items)))

    if not os.path.exists('res_dict/'):
        os.mkdir('res_dict')
    res_file_path = f'res_dict/res_{sys_paras.dataset}_{sys_paras.model}_{sys_paras.sim_coe}_{sys_paras.ablation}_{sys_paras.debias_data}_2.json'
    with open(res_file_path, 'w') as convert_file:
        convert_file.write(json.dumps(res))


def draw_boxplot(cointer_rs, ori_co_sims, random_rs, sys_paras):
    plt.figure(3)
    box_data = [random_rs[TOP_K], cointer_rs[TOP_K], ori_co_sims]
    plt.boxplot(box_data)
    plt.xticks([1, 2, 3], ['random', 'co-interacted', 'test data'])
    plt.xlabel('Experiments')
    plt.ylabel('Hit Ratio@' + str(TOP_K))
    plt.title('Box plot of each dataset')
    plt.savefig(f'figures/{sys_paras.model}_boxplot_{sys_paras.sim_coe}_{sys_paras.gamma}.jpg')
    # plt.show()


def cal_sim_users(dataset, similarity_coe, train_records):
    if os.path.exists(f'co_items/sim_users-{dataset}-{similarity_coe}.npy'):
        similar_users = np.load(f'co_items/sim_users-{dataset}-{similarity_coe}.npy', allow_pickle=True).item()
        for u in list(similar_users.keys()):
            if len(similar_users[u]) == 0:
                del similar_users[u]
    else:
        similar_users = collections.defaultdict(list)
        us = list(train_records.keys())

        sim_dict = collections.defaultdict(list)

        for m in tqdm(range(len(us))):
            for n in range(m + 1, len(us)):
                i = us[m]
                j = us[n]
                jar_sim = len(set(train_records[i]).intersection(set(train_records[j]))) / len(
                    set(train_records[i]).union(set(train_records[j])))
                sim_dict[i].append((jar_sim, j))
                sim_dict[j].append((jar_sim, i))

        for u in sim_dict:
            sim_dict[u] = sorted(sim_dict[u])
            for _, i in sim_dict[u][-similarity_coe:]:
                similar_users[u].append(i)

        for u in list(similar_users.keys()):
            if len(similar_users[u]) == 0:
                del similar_users[u]

        print(f'len(similar_users): {len(similar_users)}, len(train_records): {len(train_records)}')
        print(
            f'avg #similar users per usr: {sum([len(similar_users[u]) for u in similar_users]) / len(similar_users)}:.4f')

        np.save(f'co_items/sim_users-{dataset}-{similarity_coe}.npy', similar_users)
    return similar_users


def cal_ori_cosim(test_records, cointer_records):
    co_sims = []
    users = set(test_records.keys()).intersection(set(cointer_records.keys()))
    for usr in users:
        hs_items = set(test_records[usr])
        co_items = set(cointer_records[usr])
        co_sims.append(len(hs_items.intersection(co_items)) / len(hs_items))
    return co_sims


def draw_ori_co_sim(co_sims, sys_paras):
    plt.figure(1)
    ws = np.ones_like(co_sims) / len(co_sims)
    den, cum, bar = plt.hist(sorted(co_sims), bins=10, weights=ws, range=(0, 1))
    # pdb.set_trace()
    bar_as_ticklabel = [f"{100 * a:.1f}%" for a in bar.datavalues]
    plt.bar_label(bar, labels=bar_as_ticklabel, color='navy', fontsize=8)
    plt.xticks(cum)
    plt.xlabel('Percentage of co-interacted items in historical records')
    plt.ylabel('Percentage of Users')
    plt.title('Count co-interacted items in testing data')
    plt.savefig(f'figures/ori_sims_{sys_paras.sim_coe}.jpg')
    # plt.show()


def draw_fig(fake_rs, real_rs, sys_paras):
    plt.figure(2, figsize=(10, 4))
    plt.subplot(1, 2, 1)
    # plt.figure(1)
    ws = np.ones_like(real_rs[TOP_K]) / len(real_rs[TOP_K])
    den, cum, bar = plt.hist(sorted(real_rs[TOP_K]), bins=10, weights=ws, range=(0, 1))
    bar_as_ticklabel = [f"{100 * a:.1f}%" for a in bar.datavalues]
    plt.bar_label(bar, labels=bar_as_ticklabel, color='navy', fontsize=8)
    plt.xticks(cum)
    plt.xlabel('Hit Ratio@' + str(TOP_K))
    plt.ylabel('Percentage of Users')
    plt.title('Recommendation based on co-interaction')
    # plt.savefig('figures/co-interation.jpg')
    # plt.show()
    # plt.figure(2)
    plt.subplot(1, 2, 2)
    den, cum, bar = plt.hist(sorted(fake_rs[TOP_K]), bins=10, weights=ws, range=(0, 1))
    bar_as_ticklabel = [f"{100 * a:.1f}%" for a in bar.datavalues]
    plt.bar_label(bar, labels=bar_as_ticklabel, color='navy', fontsize=8)
    plt.xticks(cum)
    plt.xlabel('Hit Ratio@' + str(TOP_K))
    plt.ylabel('Percentage of Users')
    plt.title('Recommendation based on random')
    # plt.savefig('figures/random.jpg')
    plt.savefig(f'figures/{sys_paras.model}_rec_{sys_paras.sim_coe}_{sys_paras.gamma}.jpg')
    # plt.show()


def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def demo_sample(i_num, train_records):
    users = [u for u in train_records for _ in train_records[u]]
    pos_items = [pos_i for u in train_records for pos_i in train_records[u]]
    play_num = sum(len(train_records[x]) for x in train_records)
    neg_items = np.random.randint(0, i_num, play_num)

    return torch.LongTensor(users), torch.LongTensor(pos_items), torch.LongTensor(neg_items)

def train_model(dataset, device, i_num, test_records, train_records, model, params, global_pop):
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    total_epoch = 2000
    best_epoch = 0
    best_res = 0.005
    for epoch in range(total_epoch):
        model.train()
        tim1 = time.time()
        total_loss = 0
        runs = 0
        # if params.cpr:
        #     users, posItems, negItems = demo_cpr_sample(train_records)
        # else:
        users, posItems, negItems = demo_sample(i_num, train_records)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        for user, pos, neg in minibatch(users, posItems, negItems, batch_size=params.bs):
            optimizer.zero_grad()
            u, i, n = user.to(device), pos.to(device), neg.to(device)
            # forward pass
            loss = model(u, i, n)
            # backward and optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            runs += 1
        model.eval()
        results = test_model(model, train_records, test_records, global_pop)
        ndcg = sum(results['ndcg'][TOP_Ks[1]]) / len(results['ndcg'][TOP_Ks[1]])
        if ndcg > best_res:
            best_res = ndcg
            best_epoch = epoch
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(model.state_dict(),
                       f'checkpoints/{params.sim_coe}-{params.model}-{dataset}-{params.ablation}.pt')
        if epoch - best_epoch > 50:
            break

        print('Epoch [{}/{}], Loss: {:.4f}, NDCG@{}: {:.4f}, Time: {:.2f}s'.format(epoch + 1, total_epoch,
                                                                                   total_loss / runs,
                                                                                   TOP_Ks[1],
                                                                                   ndcg, time.time() - tim1))
    print(f'Best NDCG@{TOP_Ks[1]}: {best_res:.4f}, Best epoch: {best_epoch}')


def read_dataset(dataset, debias=False):
    if 'ml' in dataset:
        return read_ml_dataset(dataset, debias)
    else:
        return read_lg_dataset(dataset, debias)


def read_lg_dataset(dataset, debias):
    u_num = 0
    i_num = 0
    train_records = collections.defaultdict(list)
    test_records = collections.defaultdict(list)
    file = open(f'dataset/{dataset}/train.txt', 'r')
    for line in file.readlines():
        ele = line.strip().split(' ')
        user, items = ele[0], ele[1:]
        u_num = max(u_num, int(user))
        for item in items:
            i_num = max(i_num, int(item))
            train_records[int(user)].append(int(item))
    file.close()
    if debias:
        file = open(f'dataset/{dataset}/balance_test.txt', 'r')
    else:
        file = open(f'dataset/{dataset}/test.txt', 'r')
    for line in file.readlines():
        ele = line.strip().split(' ')
        user, items = ele[0], ele[1:]
        # make sure all the users are appeared in training dataset.
        if int(user) in train_records:
            for item in items:
                i_num = max(i_num, int(item))
                test_records[int(user)].append(int(item))
    file.close()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = {'num_users': u_num + 1,
              'num_items': i_num + 1,
              'device': device,
              'dataset': dataset}
    return test_records, train_records, device, i_num + 1, params


def read_ml_dataset(dataset, debias):
    train_records = collections.defaultdict(list)
    test_records = collections.defaultdict(list)

    file_m = open(f'dataset/{dataset}/movies.dat', 'r', encoding='utf-8')
    mid = 0
    uid = 0
    item_dict = dict()
    train_item_set = set()
    for line in file_m.readlines():
        ele = line.strip().split('::')
        item_dict[ele[0]] = mid
        mid += 1
    file_m.close()
    file = open(f'dataset/{dataset}/ratings.train', 'r')
    for line in file.readlines():
        ele = line.strip().split('::')
        train_records[int(ele[0]) - 1].append(item_dict[ele[1]])
        train_item_set.add(item_dict[ele[1]])
        if int(ele[0]) > uid:
            uid = int(ele[0])
    file.close()
    if debias:
        file = open(f'dataset/{dataset}/balance_ratings.test', 'r')
    else:
        file = open(f'dataset/{dataset}/ratings.test', 'r')
    for line in file.readlines():
        ele = line.strip().split('::')
        # make sure all the users and items are appeared in training dataset.
        if int(ele[0]) - 1 in train_records and item_dict[ele[1]] in train_item_set:
            test_records[int(ele[0]) - 1].append(item_dict[ele[1]])
    file.close()
    u_num = uid
    i_num = mid
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = {'num_users': u_num,
              'num_items': i_num,
              'device': device,
              'dataset': dataset}
    return test_records, train_records, device, i_num, params


def create_train_and_test(dataset):
    rating_all = dict()
    all_records = collections.defaultdict(list)
    train_records = collections.defaultdict(list)
    test_records = collections.defaultdict(list)

    file = open(f'dataset/{dataset}/ratings.dat', 'r')
    for line in file.readlines():
        ele = line.strip().split('::')
        rating_all[(int(ele[0]), ele[1])] = float(ele[2])
        if float(ele[2]) > 3:
            all_records[int(ele[0])].append(ele[1])
    file.close()

    u_num = 0
    for u in all_records:
        item_list = all_records[u]
        if len(all_records[u]) > TOP_K:
            u_num += 1
            random.shuffle(item_list)
            test_records[u].extend(item_list[:TOP_K])
            train_records[u].extend(item_list[TOP_K:])
        else:
            train_records[u].extend(item_list)

    print(f'#{TOP_K}-core users: {u_num}, #total users: {len(all_records)}, avg: {u_num / len(all_records)}:.4f')

    train_str = []
    for u in train_records:
        for i in train_records[u]:
            train_str.append(f'{u}::{i}::{rating_all[u, i]}::100000')
    random.shuffle(train_str)

    test_str = []
    for u in test_records:
        for i in test_records[u]:
            test_str.append(f'{u}::{i}::{int(rating_all[u, i])}::100000')
    random.shuffle(test_str)

    print(f'#lines of train: {len(train_str)}, #lines of test: {len(test_str)}')

    file = open(f'dataset/{dataset}/ratings.train', 'w')
    for line in train_str:
        file.write(line + '\n')
    file.close()

    file = open(f'dataset/{dataset}/ratings.test', 'w')
    for line in test_str:
        file.write(line + '\n')
    file.close()


def using_new_model(sys_paras):
    dataset = sys_paras.dataset
    sim_k = sys_paras.sim_coe
    test_records, train_records, device, i_num, params = read_dataset(dataset, sys_paras.debias_data)
    sim_users = cal_sim_users(dataset, sim_k, train_records)
    model = PPAC_MF(train_records, params, sys_paras, sim_users).to(device)

    global_pop = F.normalize(cal_global_nov(train_records, i_num)[1].float(), dim=0).numpy()
    if sys_paras.train:
        train_model(dataset, device, i_num, test_records, train_records, model, sys_paras, global_pop)

    if os.path.exists(f'checkpoints/{sys_paras.sim_coe}-{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt'):
        model.load_state_dict(
            torch.load(f'checkpoints/{sys_paras.sim_coe}-{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt',
                       map_location=device),
            strict=False)
    elif os.path.exists(f'checkpoints/{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt'):
        model.load_state_dict(
            torch.load(f'checkpoints/{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt', map_location=device),
            strict=False)
    else:
        model.load_state_dict(
            torch.load(f'checkpoints/{sys_paras.model}-{dataset}.pt', map_location=device),
            strict=False)
    time1 = time.time()
    res = test_model(model, train_records, test_records, global_pop)
    time2 = time.time()
    print(
        f"Inference time: {time2 - time1:.2f}s; #Interactions: {params['num_users'] * params['num_items']};"
        f" Avg time/interaction: {(time2 - time1) / (params['num_users'] * params['num_items']) * 1e9:.2f}ns.")
    # pdb.set_trace()
    if sys_paras.test_pru:
        GNov_results, LNov_results, GPRU_results, LPRU_results, global_pop, local_pop = test_cointer_model(model,
                                                                                                           train_records,
                                                                                                           sim_users,
                                                                                                           test_records,
                                                                                                           i_num)
        res['GTop'] = dict()
        res['LTop'] = dict()
        res['GCorrI'] = collections.defaultdict(list)
        res['LCorrI'] = collections.defaultdict(list)
        res['GCorrJ'] = collections.defaultdict(list)
        res['LCorrJ'] = collections.defaultdict(list)
        for i, topk in enumerate(TOP_Ks):
            print(f'Precision@{topk}: {sum(res["precision"][topk]) / len(res["precision"][topk]):.4f}  '
                  f'Recall@{topk}: {sum(res["recall"][topk]) / len(res["recall"][topk]):.4f}  '
                  f'NDCG@{topk}: {sum(res["ndcg"][topk]) / len(res["ndcg"][topk]):.4f} '
                  f'ARP@{topk}: {sum(res["arp"][topk]) / len(res["arp"][topk]):.4f};  '
                  f'GNov@{topk}: {sum(GNov_results[topk]) / len(GNov_results[topk]):.4f}  '
                  f'LNov@{topk}: {sum(LNov_results[topk]) / len(LNov_results[topk]):.4f}  '
                  f'GPRU@{topk}: {sum(GPRU_results[topk]) / len(GPRU_results[topk]):.4f}  '
                  f'LPRU@{topk}: {sum(LPRU_results[topk]) / len(LPRU_results[topk]):.4f} ')
            _, glo_topk_items = torch.topk(global_pop, k=topk)
            _, loc_topk_items = torch.topk(local_pop[res['users']], k=topk)
            res['GTop'][topk] = glo_topk_items.tolist()
            res['LTop'][topk] = loc_topk_items.tolist()
            for idx, test_u in enumerate(res['users']):
                inter_items = set(train_records[test_u])
                glo_items = set(res['GTop'][topk])
                loc_items = set(res['LTop'][topk][idx])
                res['GCorrI'][topk].append(len(inter_items.intersection(glo_items)) / len(inter_items))
                res['LCorrI'][topk].append(len(inter_items.intersection(loc_items)) / len(inter_items))
                res['GCorrJ'][topk].append(len(inter_items.intersection(glo_items)) / len(inter_items.union(glo_items)))
                res['LCorrJ'][topk].append(len(inter_items.intersection(loc_items)) / len(inter_items.union(loc_items)))

        if not os.path.exists('res_dict/'):
            os.mkdir('res_dict')
        res_file_path = f'res_dict/res_{sys_paras.dataset}_{sys_paras.model}_{sys_paras.sim_coe}_{sys_paras.ablation}_{sys_paras.debias_data}_2.json'
        with open(res_file_path, 'w') as convert_file:
            convert_file.write(json.dumps(res))
    else:
        for i, topk in enumerate(TOP_Ks):
            print(f'Precision@{topk}: {sum(res["precision"][topk]) / len(res["precision"][topk]):.4f}  '
                  f'Recall@{topk}: {sum(res["recall"][topk]) / len(res["recall"][topk]):.4f}  '
                  f'NDCG@{topk}: {sum(res["ndcg"][topk]) / len(res["ndcg"][topk]):.4f} '
                  f'ARP@{topk}: {sum(res["arp"][topk]) / len(res["arp"][topk]):.4f};  ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration for debias')
    sys_paras = parse_args(parser)
    set_random_seed(sys_paras.seed)
    print(f'Gamma: {sys_paras.gamma}; Beta: {sys_paras.beta}')

    if sys_paras.model == 'mf' or sys_paras.model == 'ncf':
        run_init_exp(sys_paras)
    else:
        using_new_model(sys_paras)
