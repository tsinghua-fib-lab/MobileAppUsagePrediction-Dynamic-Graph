import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import pdb
import numpy as np
import scipy.sparse as sp

from layers import *
from Modules import *
from utils import *


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)



class HGNN_ATT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, step, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.step = step
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, hidden_size, self.dropout, 0.2, transfer=False, concat=False)
        self.gat2 = HyperGraphAttentionLayerSparse(hidden_size, output_size, self.dropout, 0.2, transfer=True, concat=False)

    def forward(self, x, H):
        residual = x

        x, y = self.gat1(x, H)

        if self.step == 2:
            x = F.dropout(x, self.dropout, training=self.training)
            x += residual
            x, y = self.gat2(x, H)

        x = F.dropout(x, self.dropout, training=self.training)
        x += residual

        return x, x


class MLP(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat):
        super(MLP, self).__init__()
        self.hidden = torch.nn.Linear(in_feat, hidden_feat)
        self.predict = torch.nn.Linear(hidden_feat, out_feat)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x



class Graph(nn.Module):
    def __init__(self, n_users, n_apps, n_times, dim, time_dim, loc_dim, session_length,
                 window, dropout, batchsize, device):
        super(Graph, self).__init__()
        # app inherent embedding
        self.dim = dim
        self.window = window
        self.n_apps = n_apps
        self.device = device
        self.session_length = session_length


        self.app_inherent_emb_all = nn.Embedding(n_apps, dim)
        self.time_emb_all = nn.Embedding(n_times, time_dim)
        self.user_emb = nn.Embedding(n_users, dim)


        self.HGAT_Layers = HGNN_ATT(dim, dim, dim, 2, dropout)

        # self.pooling = nn.Linear(dim, dim)

        # for self-attention
        n_layers = 1
        n_head = 1

        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm_loc = nn.LayerNorm(loc_dim, eps=1e-6)

        # self-attention
        self.self_attention = nn.ModuleList([
            EncoderLayer(dim, dim, n_head, dim, dim, dropout=dropout)
            for _ in range(n_layers)])

        self.self_attention_loc = nn.ModuleList([
            EncoderLayer(loc_dim, loc_dim, n_head, loc_dim, loc_dim, dropout=dropout)
            for _ in range(n_layers)])

        # Transformer
        self.Transformer = TransAm(dim + time_dim + loc_dim, dropout=dropout)

        self.MLP = MLP(5, 8, batchsize)

        self.linear = nn.Linear(dim + time_dim + loc_dim, dim)


        self.LossFunction = nn.CrossEntropyLoss()


        
    def forward(self, mode, session_seq, time_seqs, loc_seqs, loc_emb_dic, target):
        g = {}
        targets = torch.LongTensor(target).to(self.device)

        # user_emb = self.user_emb(user).unsqueeze(0)
        for i in range(self.session_length):
            app_seq = [session_seq[j][i] for j in range(len(session_seq))]
            loc_seq = [loc_seqs[j][i] for j in range(len(loc_seqs))]
            times = [[time_seqs[j][i][0]] for j in range(len(time_seqs))] # [batch_size, 1]

            alias_inputs, HT, items, node_masks, locs = get_slice(app_seq, loc_seq, self.window)
            alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
            items = torch.LongTensor(items).to(self.device)
            HT = torch.LongTensor(HT).to(self.device)
            node_masks = torch.LongTensor(node_masks).to(self.device)


            times = torch.LongTensor(times).to(self.device)
            # t = times.expand(times.shape[0], items.shape[1])
            time_emb = self.time_emb_all(times)

            loc_emb = [[loc_emb_dic[locs[x][y]] for y in range(len(locs[x]))] for x in range(len(locs))]
            loc_emb = torch.Tensor(loc_emb).to(self.device)
            for layer in self.self_attention_loc:
                loc_emb, _ = layer(loc_emb, slf_attn_mask=get_pad_mask(node_masks, 0))
            loc_emb = loc_emb[torch.arange(node_masks.shape[0]).long(), node_masks.shape[1] - 1]  # [batch_size, loc_dim]

            loc_emb = self.layer_norm_loc(loc_emb).unsqueeze(1)  # [batch_size, 1, loc_dim]
            # loc_emb = lt.expand(time_emb.shape[0], time_emb.shape[1], 32)

            app_inherent_emb = self.app_inherent_emb_all(items)  # [batch_size, app_length, dim]

            rich_emb, _ = self.HGAT_Layers(app_inherent_emb, HT)  # [batch_size, app_length, dim+time_dim]

            # max-pooling
            # g[i] = torch.max(self.pooling(rich_emb), dim=1).values.unsqueeze(1)  # [batch_size, 1, dim]

            # self-attention
            get = lambda i: rich_emb[i][alias_inputs[i]]
            seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

            for layer in self.self_attention:
                output, slf_attn = layer(seq_hidden, slf_attn_mask=get_pad_mask(node_masks, 0))

            ht = output[torch.arange(node_masks.shape[0]).long(), node_masks.shape[1] - 1]  # [batch_size, dim + time_dim]

            g[i] = self.layer_norm(ht).unsqueeze(1)  # [batch_size, 1, dim]

            g[i] = torch.cat([g[i], time_emb, loc_emb], dim=2)  # [batch_size, 1, dim+time_dim+loc_dim]


        G = torch.cat([g[j]for j in g.keys()], dim=1)  # [batch_size, session_length, dim + time_dim]



        # Transformer
        ud_emb = self.Transformer(G, g[self.session_length - 1]) # [batch_size, dim + time_dim]


        #MLP
        last_items = alias_inputs[:,-4:]
        get = lambda i: rich_emb[i][last_items[i]]
        last_emb = torch.stack([get(i) for i in torch.arange(len(last_items)).long()])
        
        t = time_emb.expand(last_emb.shape[0], last_emb.shape[1], time_emb.shape[2])
        l = loc_emb.expand(last_emb.shape[0], last_emb.shape[1], loc_emb.shape[2])
        last_emb = torch.cat([last_emb, t, l], dim=2)
        
        M = torch.cat([ud_emb, last_emb], dim=1)
        

        final_emb = self.MLP(M.transpose(1,2)).transpose(1,2)
        
        # t = time_emb.expand(final_emb.shape[0], final_emb.shape[1], time_emb.shape[2])
        # l = loc_emb.expand(final_emb.shape[0], final_emb.shape[1], loc_emb.shape[2])
        
        # final_emb = torch.cat([final_emb, t, l], dim=2)
        
        final_emb = self.linear(final_emb) # [batch_size, dim]

        item_emb = self.app_inherent_emb_all.weight  # [n_apps, dim]

        scores = torch.matmul(final_emb.squeeze(1), item_emb.transpose(1,0))  # [batch_size, n_apps]


        if mode == 'predict':
            return scores
        else:
            loss = self.LossFunction(scores, targets)
            return loss

