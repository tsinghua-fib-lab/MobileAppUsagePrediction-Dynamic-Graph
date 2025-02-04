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

from graph import Graph
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of AppUsage2Vec model")

    parser.add_argument('--epoch', type=int, default=10, help="The number of epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="The size of batch")
    parser.add_argument('--dim', type=int, default=64, help="The embedding size of users and apps")
    parser.add_argument('--time_dim', type=int, default=8, help="The embedding size of time")
    parser.add_argument('--loc_dim', type=int, default=32, help="The embedding size of time")
    parser.add_argument('--seq_length', type=int, default=4, help="The length of previously used app sequence")
    parser.add_argument('--window', type=int, default=8, help="The max length of sliding window")
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
    location_emb = pd.read_csv('./data/location_emb.txt', sep='\t', names=['idx', 'emb'])
    loc_emb_dic = {location_emb.iloc[i]['idx']: ast.literal_eval(location_emb.iloc[i]['emb'])
                   for i in range(len(location_emb))}

    train_dataset = Data_Loading(mode='train')
    test_dataset = Data_Loading(mode='test')
    train_dataset = Data(train_dataset)
    test_dataset = Data(test_dataset)
    
    with open('./data/app2id.json','r') as f:
      app2id = json.load(f)

    with open('./data/user2id.json','r') as f:
      user2id = json.load(f)


    num_users = len(user2id)
    num_apps = len(app2id)
    num_times = 48
    print(train_dataset.length)
    print(test_dataset.length)


    # model & optimizer
    if args.load_model == True:
      model = torch.load('model_dim'+str(args.dim)+'win2.pth', map_location='cpu')
    else:
      model = Graph(num_users, num_apps, num_times, args.dim, args.time_dim, args.loc_dim,
                  args.seq_length, args.window, args.dropout, args.batch_size, device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train & evaluation
    total_loss = 0
    itr = 1
    p_itr = 25000
    Ks = [1, 5, 10]
    acc_history = [[0, 0, 0]]
    best_acc = [0, 0, 0]
    mrr_history = [[0, 0, 0]]
    best_mrr = [0, 0, 0]
    ndcg_history = [[0, 0, 0]]
    best_ndcg = [0, 0, 0]
    bad_counter = 0

    for e in tqdm.tqdm(range(args.epoch)):

        model.train()

        slices = train_dataset.generate_batch(args.batch_size, True)

        for step in tqdm.tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
            i = slices[step]
            session_seqs, time_seqs, loc_seqs, targets = train_dataset.get_slice(i)

            optimizer.zero_grad()
            loss = model('train', session_seqs, time_seqs, loc_seqs, loc_emb_dic, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if itr % p_itr == 0:
                print("[TRAIN] Epoch: {} / Iter: {} Loss - {}".format(e + 1, itr, total_loss / p_itr))
                total_loss = 0
            itr += 1


        model.eval()

        with torch.no_grad():
            slices = test_dataset.generate_batch(min(args.batch_size, test_dataset.length), False)

            accs, mrrs, ndcgs = {}, {}, {}
            for k in Ks:
                accs[k] = []
                mrrs[k] = []
                ndcgs[k] = []

            for step in tqdm.tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
                i = slices[step]
                session_seqs, time_seqs, loc_seqs, targets = test_dataset.get_slice(i)
                scores = model('predict', session_seqs, time_seqs, loc_seqs, loc_emb_dic, targets)
                # targets = torch.LongTensor(targets).to(device)

                sub_scores = scores.topk(Ks[-1])[1]
                sub_scores = sub_scores.cpu().detach().numpy()

                for score, target in zip(sub_scores, targets):
                    for k in Ks:
                        accs[k].append(np.isin(target, score[:k]))
                        if len(np.where(score[:k] == target)[0]) == 0:
                            mrrs[k].append(0)
                            ndcgs[k].append(0)
                        else:
                            mrrs[k].append(1.0 / (np.where(score[:k] == target)[0][0] + 1))
                            ndcgs[k].append(1.0 / np.log2((np.where(score[:k] == target)[0][0] + 2)))

            for k in Ks:
                accs[k] = np.mean(accs[k])
                mrrs[k] = np.mean(mrrs[k])
                ndcgs[k] = np.mean(ndcgs[k])


        acc_history.append([accs[k] for k in Ks])
        mrr_history.append([mrrs[k] for k in Ks])
        ndcg_history.append([ndcgs[k] for k in Ks])

        print(
            "[EVALUATION] Epoch: {} - Acc: {:.5f}/{:.5f}/{:.5f}"
            .format(e + 1, accs[1], accs[5], accs[10]))
        print(
            "[EVALUATION] Epoch: {} - Mrr: {:.5f}/{:.5f}/{:.5f}"
            .format(e + 1, mrrs[1], mrrs[5], mrrs[10]))
        print(
            "[EVALUATION] Epoch: {} - NDCG: {:.5f}/{:.5f}/{:.5f}"
            .format(e + 1, ndcgs[1], ndcgs[5], ndcgs[10]))

        for i, k in enumerate(Ks):
            if accs[k] > best_acc[i]:
                best_acc[i] = accs[k]
                if i == 0:
                    flag = 1
            if mrrs[k] > best_mrr[i]:
                best_mrr[i] = mrrs[k]
            if ndcgs[k] > best_ndcg[i]:
                best_ndcg[i] = ndcgs[k]

            torch.save(model, 'model_dim'+str(args.dim)+'win2.pth')

            bad_counter += 1 - flag
            if bad_counter >= args.patience:
                break

    print("BEST ACC@1: {} / @5: {} / @10: {}"
          .format(best_acc[0], best_acc[1], best_acc[2]))
    print("BEST MRR@1: {} / @5: {} / @10: {}"
          .format(best_mrr[0], best_mrr[1], best_mrr[2]))
    print("BEST NDCG@1: {} / @5: {} / @10: {}"
          .format(best_ndcg[0], best_ndcg[1], best_ndcg[2]))





if __name__ == "__main__":
    main()
