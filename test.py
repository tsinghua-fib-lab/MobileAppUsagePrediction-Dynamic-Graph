import os
import argparse
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm
import setproctitle
import ast
import json
import scipy.sparse as sp
import pdb

from graph import Graph


setproctitle.setproctitle("dim8@hzh")
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def Data_Loading(mode):
    if mode == 'train':
        data = pd.read_csv('data/active/tracenum/test1.txt', sep='\t')
    else:
        data = pd.read_csv('data/active/tracenum/test1.txt', sep='\t')


    data['session_seq'] = data['session_seq'].apply(ast.literal_eval)
    data['time_seq'] = data['time_seq'].apply(ast.literal_eval)
    data['loc_seq'] = data['loc_seq'].apply(ast.literal_eval)

    return data


class Data():
    def __init__(self, data):
        user = data['user']
        app_seqs = data['session_seq']
        time_seqs = data['time_seq']
        loc_seqs = data['loc_seq']
        self.user = np.asarray(user)
        self.app_seqs = np.asarray(app_seqs)
        self.time_seqs = np.asarray(time_seqs)
        self.loc_seqs = np.asarray(loc_seqs)
        self.targets = np.asarray(data['app'])
        self.length = len(app_seqs)


    def generate_batch(self, batch_size, shuffle=False):
        if shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.user = self.user[shuffled_arg]
            self.app_seqs = self.app_seqs[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            self.time_seqs = self.time_seqs[shuffled_arg]
            self.loc_seqs = self.loc_seqs[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]

        return slices


    def get_slice(self, iList):
        user = self.user[iList]
        session_seqs, time_seqs, = self.app_seqs[iList], self.time_seqs[iList]
        loc_seqs, targets = self.loc_seqs[iList], self.targets[iList]

        return user, session_seqs, time_seqs, loc_seqs, targets

def get_slice(data, loc_seq, window):

    inputs = np.asarray(data)
    items, n_node, HT, alias_inputs, node_masks, node_dic = [], [], [], [], [], []
    num_edge, edge_mask, edge_inputs = [], [], []
    locs = []


    # in every batch:
    for i in range(len(inputs)):
        temp_s = inputs[i]  # app sequence

        temp_l = list(set(temp_s))  # app set
        temp_dic = {temp_l[i]: i for i in range(len(temp_l))}
        n_node.append(temp_l)
        alias_inputs.append([temp_dic[i] for i in temp_s])
        node_dic.append(temp_dic)


        min_s = min(window, len(temp_s))
        num_edge.append(int((1 + min_s) * len(temp_s) - (1 + min_s) * min_s / 2))

    max_n_node = np.max([len(i) for i in n_node])  # batch中最大结点数

    max_n_edge = max(num_edge)  # batch中最大边数

    max_se_len = max([len(i) for i in alias_inputs])  # batch中最大序列长度

    edge_mask = [[1] * len(le) + [0] * (max_n_edge - len(le)) for le in alias_inputs]

    for idx in range(len(inputs)):
        # u_input = inputs[idx]
        effect_len = len(alias_inputs[idx])  # 有效长度
        node = n_node[idx]
        items.append(node + (max_n_node - len(node)) * [0])  # 补全长度
        locs.append(loc_seq[idx] + (max_se_len - len(loc_seq[idx])) * [0])

        effect_list = alias_inputs[idx]
        # ws = np.ones(max_n_edge)
        cols = []
        rows = []
        edg = []
        e_idx = 0

        for w in range(1 + min(window, effect_len - 1)):
            if w % 2 == 0:
                edge_idx = list(np.arange(e_idx, e_idx + effect_len - w))
                edg += edge_idx
                for ww in range(w + 1):
                    rows += effect_list[ww:ww + effect_len - w]
                    cols += edge_idx

                e_idx += len(edge_idx)

        u_H = sp.coo_matrix(([1.0] * len(rows), (rows, cols)), shape=(max_n_node, max_n_edge))
        HT.append(np.asarray(u_H.T.todense()))

        node_masks.append((max_se_len - len(alias_inputs[idx])) * [0] + [1] * len(alias_inputs[idx]))
        alias_inputs[idx] = (max_se_len - len(alias_inputs[idx])) * [0] + alias_inputs[idx]

        edge_inputs.append(edg + (max_n_edge - len(edg)) * [0])

    return alias_inputs, HT, items, node_masks, locs


def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of AppUsage2Vec model")

    parser.add_argument('--epoch', type=int, default=10, help="The number of epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="The size of batch")
    parser.add_argument('--dim', type=int, default=64, help="The embedding size of users and apps")
    parser.add_argument('--time_dim', type=int, default=8, help="The embedding size of time")
    parser.add_argument('--loc_dim', type=int, default=32, help="The embedding size of time")
    parser.add_argument('--seq_length', type=int, default=4, help="The length of previously used app sequence")
    parser.add_argument('--window', type=int, default=2, help="The max length of sliding window")
    parser.add_argument('--dropout', type=int, default=0.2, help="dropout coefficient")
    parser.add_argument('--alpha', type=float, default=0.1, help="Coefficient for LeakyReLu")
    parser.add_argument('--topk', type=float, default=5, help="Top k for loss function")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Coefficient for L2 regularization")
    parser.add_argument('--seed', type=int, default=2023, help="Random seed")
    parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop')
    parser.add_argument('--load_model', type=bool, default=False, help='load model')

    return parser.parse_args()


def main():
    args = parse_args()

    # random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)

    # data load
    location_emb = pd.read_csv('data/location_emb.txt', sep='\t', names=['idx', 'emb'])
    loc_emb_dic = {location_emb.iloc[i]['idx']: ast.literal_eval(location_emb.iloc[i]['emb'])
                   for i in range(len(location_emb))}

    # train_dataset = Data_Loading(mode='train')
    test_dataset = Data_Loading(mode='test')
    # train_dataset = Data(train_dataset)
    test_dataset = Data(test_dataset)

    with open('data/app2id.json', 'r') as f:
        app2id = json.load(f)

    with open('data/user2id.json', 'r') as f:
        user2id = json.load(f)

    num_users = len(user2id)
    num_apps = len(app2id)
    num_times = 48

    # model & optimizer
    model = torch.load('model_final.pth', map_location='cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # train & evaluation
    Ks = [1, 5, 10]
    accs, mrrs, ndcgs = {}, {}, {}



    model.eval()

    with torch.no_grad():
        slices = test_dataset.generate_batch(min(args.batch_size, test_dataset.length), False)

        for step in tqdm.tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
            i = slices[step]
            users, session_seqs, time_seqs, loc_seqs, targets = test_dataset.get_slice(i)
            scores = model('predict', session_seqs, time_seqs, loc_seqs, loc_emb_dic, targets)
            # pdb.set_trace()
            for u in users:
                user = int(u)
                if user not in accs:
                    accs[user] = [[], [], []]
                    mrrs[user] = [[], [], []]
                    ndcgs[user] = [[], [], []]
    
                sub_scores = scores.topk(Ks[-1])[1]
                sub_scores = sub_scores.cpu().detach().numpy()
    
                for score, target in zip(sub_scores, targets):
                    for i,k in enumerate(Ks):
                        accs[user][i].append(np.isin(target, score[:k]))
                        if len(np.where(score[:k] == target)[0]) == 0:
                            mrrs[user][i].append(0)
                            ndcgs[user][i].append(0)
                        else:
                            mrrs[user][i].append(1.0 / (np.where(score[:k] == target)[0][0] + 1))
                            ndcgs[user][i].append(1.0 / np.log2((np.where(score[:k] == target)[0][0] + 2)))

    for user in accs:
        for i in range(3):
            accs[user][i] = np.mean(accs[user][i])
            mrrs[user][i] = np.mean(mrrs[user][i])
            ndcgs[user][i] = np.mean(ndcgs[user][i])

    with open('data/active/tracenum/test1/accs.json', 'w') as f:
        json.dump(accs, f)
    with open('data/active/tracenum/test1/mrrs.json', 'w') as f:
        json.dump(mrrs, f)
    with open('data/active/tracenum/test1/ndcgs.json', 'w') as f:
        json.dump(ndcgs, f)




if __name__ == "__main__":
    main()