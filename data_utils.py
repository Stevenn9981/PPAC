import json
import pdb
import sys

import numpy as np
from matplotlib import ticker

from run_MF import *

plt.rcParams["pdf.use14corefonts"] = True


def cal_avg(ndcgs):
    for i in range(len(ndcgs)):
        ndcgs[i] = sum(ndcgs[i]) / len(ndcgs[i])
    return ndcgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration for debias')
    sys_paras = parse_args(parser)
    set_random_seed(sys_paras.seed)
    test_records, train_records, device, i_num, params = read_dataset(sys_paras.dataset, sys_paras.debias_data)
    test_users = list(test_records.keys())
    Topk = '50'

    if sys_paras.draw_aly == 'intro':
        local_pop = np.load(f'co_items/local_pop-{sys_paras.dataset}-{sys_paras.sim_coe}.npy', allow_pickle=True)
        local_pop = torch.tensor(local_pop)
        global_pop = cal_global_nov(train_records, i_num)[1]
        GTopk_items = set(torch.topk(global_pop, k=int(Topk))[1].tolist())
        LTopk_items = torch.topk(local_pop, k=int(Topk))[1].tolist()
        # pdb.set_trace()
        # GTopk_items = set(res_cid['GTop'][Topk])
        # LTopk_items = res_cid['LTop'][Topk]
        distri = [0] * (int(Topk) + 1)
        for i in range(len(LTopk_items)):
            distri[len(GTopk_items.intersection(set(LTopk_items[i])))] += 1
        distri = np.array(distri) / sum(distri)
        plt.figure(figsize=(9, 5))
        plt.yticks(fontsize=24)
        plt.xticks(list(range(0, int(Topk) + 1, 5)), fontsize=24)
        plt.plot(list(range(int(Topk) + 1)), distri, color='black', marker='o', mec='b', mfc='b')
        plt.ylabel('User ratio', fontsize=24)
        plt.xlabel(f'Overlap between global and personal top items', fontsize=22)
        # plt.xticks(list(range(int(Topk) + 1)))
        # print(distri)
        plt.tight_layout()
        plt.savefig('distri.pdf', bbox_inches='tight')
        plt.show()
        sys.exit()
    elif sys_paras.draw_aly == 'intro_pie':
        local_pop = np.load(f'co_items/local_pop-{sys_paras.dataset}-{sys_paras.sim_coe}.npy', allow_pickle=True)
        local_pop = torch.tensor(local_pop)
        global_pop = cal_global_nov(train_records, i_num)[1]
        GTopk_items = set(torch.topk(global_pop, k=int(Topk))[1].tolist())
        LTopk_items = torch.topk(local_pop, k=int(Topk))[1].tolist()
        distri = [0] * 5
        for i in range(len(LTopk_items)):
            distri[min(len(set(LTopk_items[i]).difference(GTopk_items)), 49) // 10] += 1
        distri = np.array(distri) / sum(distri)
        labels = ['11-20', '21-30', '31-40', '41-50']

        cmap = plt.get_cmap("tab20c")
        outer_colors = cmap([1, 8, 5, 12])

        plt.pie(distri[1:], labels=labels, autopct='%1.1f%%', normalize=False, colors=outer_colors, counterclock=False,
                textprops={'fontsize': 18})
        plt.title(f'User distribution on differences', fontsize=23)
        plt.tight_layout()
        plt.savefig('distri_pie.pdf', bbox_inches='tight')
        plt.show()
        sys.exit()
    elif sys_paras.draw_aly == 'intro_bar':
        local_pop = np.load(f'co_items/local_pop-{sys_paras.dataset}-{sys_paras.sim_coe}.npy', allow_pickle=True)
        local_pop = torch.tensor(local_pop)
        global_pop = cal_global_nov(train_records, i_num)[1]
        GTopk_items = set(torch.topk(global_pop, k=int(Topk))[1].tolist())
        LTopk_items = torch.topk(local_pop, k=int(Topk))[1].tolist()
        distri = [0] * 5
        for i in range(len(LTopk_items)):
            distri[min(len(set(LTopk_items[i]).difference(GTopk_items)), 49) // 10] += 1
        distri = np.array(distri) / sum(distri)
        labels = ['0-10', '11-20', '21-30', '31-40', '41-50']

        # cmap = plt.get_cmap("tab20c")
        # outer_colors = cmap([1, 8, 5, 12])

        # plt.pie(distri[1:], labels=labels, autopct='%1.1f%%', normalize=False, colors=outer_colors, counterclock=False,
                # textprops={'fontsize': 18})
        plt.rcParams["font.size"] = 16
        plt.figure(figsize=(5, 4))
        plt.bar(labels, distri[:])
        plt.ylim([0, 0.53])

        for a, b in zip(labels, distri):
            plt.text(a, b, f'{b * 100:.1f}%', ha='center', va='bottom')

        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        # plt.title(f'User distribution on differences', fontsize=20)
        plt.xlabel(f'Differences between top-50 PP and GP items')
        plt.tight_layout()
        plt.savefig('distri_bar.pdf', bbox_inches='tight')
        plt.show()
        sys.exit()

    res_ori_file_path = f'res_dict/res_{sys_paras.dataset}_mf_{sys_paras.sim_coe}_{sys_paras.ablation}_{sys_paras.debias_data}_2.json'
    res_glo_file_path = f'res_dict/res_{sys_paras.dataset}_macrmf_{sys_paras.sim_coe}_{sys_paras.ablation}_{sys_paras.debias_data}_2.json'
    res_cid_file_path = f'res_dict/res_{sys_paras.dataset}_cidmf_{sys_paras.sim_coe}_{sys_paras.ablation}_{sys_paras.debias_data}_2.json'
    # if sys_paras.model == 'lg':
    #     res_ori_file_path = f'res_dict/res_{sys_paras.dataset}_lightgcn_{sys_paras.sim_coe}_2.json'
    #     res_cid_file_path = f'res_dict/res_{sys_paras.dataset}_cidlg_{sys_paras.sim_coe}_2.json'
    with open(res_ori_file_path, 'r') as convert_file:
        json_str = convert_file.readlines()[0]
        res_ori = json.loads(json_str)
    with open(res_glo_file_path, 'r') as convert_file:
        json_str = convert_file.readlines()[0]
        res_glo = json.loads(json_str)
    with open(res_cid_file_path, 'r') as convert_file:
        json_str = convert_file.readlines()[0]
        res_cid = json.loads(json_str)

    ndcg_cid = np.array(res_cid['ndcg'][Topk])  # - np.array(res_ori['ndcg'][Topk])
    ndcg_glo = np.array(res_glo['ndcg'][Topk])
    ndcg_ori = np.array(res_ori['ndcg'][Topk])
    # # ndcg_gain = np.array(res_ori['ndcg'][Topk])
    pru_metric = 'LCorrJ'

    x = np.array(res_cid[pru_metric][Topk])
    plt.figure(figsize=(5, 4))
    plt.rcParams['font.size'] = 14
    plt.xlabel(f'Normalized {pru_metric[:2]}@{Topk}', fontsize=16)
    plt.ylabel(f'NDCG@{Topk}', fontsize=16)
    # plt.title('Our model')

    x = (x - min(x)) / (max(x) - min(x))  # normalization
    # y_ticks = np.arange(0, 1, 0.1)
    # plt.yticks(y_ticks)
    bin_num = 5
    global_step = 1 / bin_num

    ndcg_grps_cid = [[] for _ in range(bin_num)]
    ndcg_grps_glo = [[] for _ in range(bin_num)]
    ndcg_grps_ori = [[] for _ in range(bin_num)]

    glo_x_ticks = []
    for i, pru in enumerate(x):
        ndcg_grps_cid[min(int(pru / global_step), bin_num - 1)].append(ndcg_cid[i])
        ndcg_grps_glo[min(int(pru / global_step), bin_num - 1)].append(ndcg_glo[i])
        ndcg_grps_ori[min(int(pru / global_step), bin_num - 1)].append(ndcg_ori[i])

    for i in range(bin_num):
        if len(ndcg_grps_cid[i]) > 0:
            if i == bin_num - 1:
                glo_x_ticks.append(f'[{i * global_step:.1f}, {(i + 1) * global_step:.1f}]')
            else:
                glo_x_ticks.append(f'[{i * global_step:.1f}, {(i + 1) * global_step:.1f})')
        else:
            glo_x_ticks.append("")
    if sys_paras.debias_data and pru_metric == 'GCorrJ':
        for i in range(len(ndcg_grps_glo[-1])):
            ndcg_grps_glo[-1][i] /= 1.5
    # plt.boxplot(ndcg_grps_cid)
    color2 = 'r'
    color3 = 'g'
    color4 = 'b'
    xticks = list(range(1, len(glo_x_ticks) + 1))
    plt.plot(xticks, cal_avg(ndcg_grps_ori), label="BPRMF", color=color2, marker='.',
             markersize=15)  # 设置线粗细，节点样式
    plt.plot(xticks, cal_avg(ndcg_grps_glo), label="MACR_BPRMF", color=color3, marker='.',
             markersize=15)  # 设置线粗细，节点样式
    plt.plot(xticks, cal_avg(ndcg_grps_cid), label="GPP_BPRMF", color=color4,
             marker='.', markersize=15)  # 设置线粗细，节点样式
    plt.xticks(xticks, glo_x_ticks, rotation=30, fontsize=16)
    plt.legend(fontsize=15)
    plt.tight_layout()
    if sys_paras.debias_data:
        plt.ylim([0, 0.4])
    else:
        plt.ylim([0, 0.5])

    if sys_paras.model == 'lg':
        plt.savefig(f'lg_{pru_metric}_{sys_paras.debias_data}.pdf')
    else:
        plt.savefig(f'mf_{pru_metric}_{sys_paras.debias_data}.pdf')
    plt.show()

    # bin_num = 10
    # step = 1 / bin_num
    # pru_1, pru_2 = 'GCorrJ', 'LCorrJ'
    # ndcg_sums = np.zeros((bin_num, bin_num))
    # ndcg_nums = np.zeros((bin_num, bin_num))
    # x = np.array(res_cid[pru_1][Topk])
    # y = np.array(res_cid[pru_2][Topk])
    #
    # x = (x - min(x)) / (max(x) - min(x))
    # y = (y - min(y)) / (max(y) - min(y))
    #
    # for idx, p1 in enumerate(x):
    #     p2 = y[idx]
    #     ndcg_sums[min(bin_num - 1, int(p1 / step)), min(bin_num - 1, int(p2 / step))] += ndcg_gain[idx]
    #     ndcg_nums[min(bin_num - 1, int(p1 / step)), min(bin_num - 1, int(p2 / step))] += 1
    # ndcg_nums = np.where(ndcg_nums == 0, 1, ndcg_nums)
    # ndcg_avg = ndcg_sums / ndcg_nums
    #
    # ndcg_grps = list(range(bin_num))
    # glo_x_ticks = []
    # for i in range(bin_num):
    #     if i == bin_num - 1:
    #         glo_x_ticks.append(f'[{i * step:.1f}, {(i + 1) * step:.1f}]')
    #     else:
    #         glo_x_ticks.append(f'[{i * step:.1f}, {(i + 1) * step:.1f})')
    #
    # plt.xticks(ndcg_grps, glo_x_ticks, rotation=90)
    # plt.yticks(ndcg_grps, glo_x_ticks)
    # plt.xlabel(f'{pru_2}')
    # plt.ylabel(f'{pru_1}')
    #
    # for i in range(bin_num):
    #     for j in range(bin_num):
    #         text = plt.text(j, i, f'{ndcg_avg[i, j]:.2f}', ha="center", va="center", color="w")
    # plt.imshow(ndcg_avg)
    # plt.tight_layout()
    # plt.show()
