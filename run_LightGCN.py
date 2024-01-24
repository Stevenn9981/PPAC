import argparse
import json

import time

import dgl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from PPAC import PPAC_LG
from model.LightGCN import LightGCN
from model.MF import *
from run_MF import parse_args, cal_sim_users, read_dataset, demo_sample, shuffle
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


def run_init_exp(sys_paras):
    dataset = sys_paras.dataset
    similarity_coe = sys_paras.sim_coe
    test_records, train_records, device, i_num, params = read_dataset(dataset, sys_paras.debias_data)
    print(f'#Users: {len(train_records)}')
    print(f'avg #interactions per usr: {sum([len(train_records[u]) for u in train_records]) / len(train_records)}')

    if sys_paras.model == 'lightgcn':
        model = LightGCN(params, sys_paras).to(device)
    graph = create_train_graph(train_records, params).to(device)
    global_pop = F.normalize(cal_global_nov(train_records, i_num)[1].float(), dim=0).numpy()
    if sys_paras.train:
        train_model(dataset, device, i_num, test_records, train_records, model, sys_paras, graph, global_pop)

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

    res = test_model(model, train_records, test_records, global_pop, graph)
    sim_users = cal_sim_users(dataset, similarity_coe, train_records)
    GNov_results, LNov_results, GPRU_results, LPRU_results, global_pop, local_pop = test_cointer_model(model,
                                                                                                       train_records,
                                                                                                       sim_users,
                                                                                                       test_records,
                                                                                                       i_num, graph)

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
    res_file_path = f'res_dict/res_{sys_paras.dataset}_{sys_paras.model}_{sys_paras.sim_coe}_2.json'
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


def train_model(dataset, device, i_num, test_records, train_records, model, params, graph, global_pop):
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    total_epoch = 2000
    best_epoch = 0
    best_res = 0.002
    for epoch in range(total_epoch):
        model.train()
        tim1 = time.time()
        total_loss = 0
        runs = 0
        users, posItems, negItems = demo_sample(i_num, train_records)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        for user, pos, neg in minibatch(users, posItems, negItems, batch_size=params.bs):
            optimizer.zero_grad()
            u, i, n = user.to(device), pos.to(device), neg.to(device)
            # forward pass
            loss, reg_loss = model.bpr_loss(graph, u, i, n)
            loss = loss + reg_loss * 1e-4
            # backward and optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            runs += 1
        model.eval()
        results = test_model(model, train_records, test_records, global_pop, graph)
        ndcg = sum(results['ndcg'][TOP_Ks[1]]) / len(results['ndcg'][TOP_Ks[1]])
        if ndcg > best_res:
            best_res = ndcg
            best_epoch = epoch
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(model.state_dict(), f'checkpoints/{params.sim_coe}-{params.model}-{dataset}-{params.ablation}.pt')
        if epoch - best_epoch > 50:
            break

        print('Epoch [{}/{}], Loss: {:.4f}, NDCG@{}: {:.4f}, Time: {:.2f}s'.format(epoch + 1, total_epoch,
                                                                                   total_loss / runs,
                                                                                   TOP_Ks[1],
                                                                                   ndcg, time.time() - tim1))
    print(f'Best NDCG@{TOP_Ks[1]}: {best_res:.4f}, Best epoch: {best_epoch}')


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

    print(f'#{TOP_K}-core users: {u_num}, #total users: {len(all_records)}, avg: {u_num / len(all_records)}')

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
    params['rAdj'] = sys_paras.rAdj
    graph = create_train_graph(train_records, params).to(device)
    sim_users = cal_sim_users(dataset, sim_k, train_records)

    model = PPAC_LG(train_records, params, sys_paras, sim_users).to(device)

    global_pop = F.normalize(cal_global_nov(train_records, i_num)[1].float(), dim=0).numpy()
    if sys_paras.train:
        train_model(dataset, device, i_num, test_records, train_records, model, sys_paras, graph, global_pop)

    if os.path.exists(f'checkpoints/{sys_paras.sim_coe}-{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt'):
        model.load_state_dict(
            torch.load(f'checkpoints/{sys_paras.sim_coe}-{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt', map_location=device),
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
    res = test_model(model, train_records, test_records, global_pop, graph)
    time2 = time.time()
    print(
        f"Inference time: {time2 - time1:.2f}s; #Interactions: {params['num_users'] * params['num_items']};"
        f" Avg time/interaction: {(time2 - time1) / (params['num_users'] * params['num_items']) * 1e9:.2f}ns.")
    if sys_paras.test_pru:
        GNov_results, LNov_results, GPRU_results, LPRU_results, global_pop, local_pop = test_cointer_model(model,
                                                                                                           train_records,
                                                                                                           sim_users,
                                                                                                           test_records,
                                                                                                           i_num, graph)

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
        res_file_path = f'res_dict/res_{sys_paras.dataset}_{sys_paras.model}_{sys_paras.sim_coe}_2.json'
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

    if sys_paras.model == 'lightgcn' or sys_paras.model == 'lg':
        run_init_exp(sys_paras)
    else:
        using_new_model(sys_paras)
