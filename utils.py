import numpy as np
import scipy.sparse as sp
import pandas as pd
import ast


def Data_Loading(mode):

    if mode == 'train':
        data = pd.read_csv('./data/train.txt', sep='\t')
    else:
        data = pd.read_csv('./data/test.txt', sep='\t')

    data['session_seq'] = data['session_seq'].apply(ast.literal_eval)
    data['time_seq'] = data['time_seq'].apply(ast.literal_eval)
    data['loc_seq'] = data['loc_seq'].apply(ast.literal_eval)

    return data


class Data():
    def __init__(self, data):
        app_seqs = data['session_seq']
        time_seqs = data['time_seq']
        loc_seqs = data['loc_seq']
        self.app_seqs = np.asarray(app_seqs)
        self.time_seqs = np.asarray(time_seqs)
        self.loc_seqs = np.asarray(loc_seqs)
        self.targets = np.asarray(data['app'])
        self.length = len(app_seqs)


    def generate_batch(self, batch_size, shuffle=False):
        if shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
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
        session_seqs, time_seqs, = self.app_seqs[iList], self.time_seqs[iList]
        loc_seqs, targets = self.loc_seqs[iList], self.targets[iList]

        return session_seqs, time_seqs, loc_seqs, targets



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


# def get_one_graph(data, window):
#     '''
#     :param data:
#     :param window: max window size
#     :return:
#     alias_inputs: 替换序号后的app序列
#     HT: 邻接矩阵
#     items: 补全长度后的app sequence
#     node_masks
#     edge_mask
#     edge_inputs
#     '''
#     inputs = np.asarray(data)  # [batchsize, session sequence]
#     items, n_node, HT, alias_inputs, node_masks, node_dic = [], [], [], [], [], []
#     num_edge, edge_mask, edge_inputs = [], [], []
#
#     max_se_len = 0
#     # in every batch:
#     for u_input in inputs:
#         temp_s = u_input  # session sequence: app sequence * 3
#         temp_l = set()
#         L = 0
#         numedge = 0
#         for i in temp_s:
#             temp_l = temp_l | set(i)
#             L = L + len(i)
#             max_se_len = max(len(i), max_se_len)  # 最大app sequence 长度
#             min_s = min(window, len(i))
#             numedge += int((1 + min_s) * len(i) / 2 - (1 + min_s - 1) * min_s / 4)
#
#         temp_l = list(temp_l)
#         temp_dic = {temp_l[i]: i for i in range(len(temp_l))}
#         n_node.append(temp_l)
#
#         alias_inputs.append([[temp_dic[i] for i in j] for j in temp_s])
#         node_dic.append(temp_dic)
#
#         num_edge.append(numedge)
#
#     max_n_node = np.max([len(i) for i in n_node])  # batch中最大结点数
#
#     max_n_edge = max(num_edge) + 3 # batch中最大边数 + session个数
#
#
#
#     for i in range(len(inputs)):
#         node_mask = []
#         cols = []
#         rows = []
#         edg = []
#         e_idx = 0
#
#         node = n_node[i]
#         items.append(node + (max_n_node - len(node)) * [0])
#
#         for j in range(len(inputs[i])):
#             effect_len = len(alias_inputs[i][j])  # 有效长度
#             effect_list = alias_inputs[i][j]
#
#             node_mask.append((max_se_len - len(alias_inputs[i][j])) * [0] + [1] * len(alias_inputs[i][j]))
#             alias_inputs[i][j] = (max_se_len - len(alias_inputs[i][j])) * [0] + alias_inputs[i][j]
#
#
#             for w in range(1 + min(window, effect_len - 2)):
#                 if w % 2 == 1:
#                     edge_idx = list(np.arange(e_idx, e_idx + effect_len - w))
#                     edg += edge_idx
#                     for ww in range(w + 1):
#                         rows += effect_list[ww:ww + effect_len - w]
#                         cols += edge_idx
#
#                     e_idx += len(edge_idx)
#
#
#             rows += effect_list
#             cols += [e_idx] * effect_len
#             e_idx += 1
#
#         u_H = sp.coo_matrix(([1.0] * len(rows), (rows, cols)), shape=(max_n_node, max_n_edge))
#         HT.append(np.asarray(u_H.T.todense()))
#
#         node_masks.append(node_mask)
#
#         edge_inputs.append(edg + (max_n_edge - len(edg)) * [0])
#
#
#     return alias_inputs, HT, items, node_masks

def get_one_graph(data, window, session_length):
    '''
    :param data:
    :param window: max window size
    :return:
    alias_inputs: 替换序号后的app序列
    HT: 邻接矩阵
    items: 补全长度后的app sequence
    node_masks
    edge_mask
    edge_inputs
    '''
    inputs = np.asarray(data)  # [batchsize, session sequence]
    items, n_node, HT, alias_inputs, node_masks = [], [], [], [], []
    num_edge, edge_mask, edge_inputs = [], [], []
    same_node_dic = []
    numnode = []
    max_se_len = 0

    # in every batch:
    for u_input in inputs:
        temp_s = u_input  # session sequence: app sequence * 3
        L = 0
        node_num = 0
        numedge = 0
        node_dic = []
        same_dic = {}
        nodes = []
        for i in temp_s:
            temp_l = list(set(i))
            nodes = nodes + temp_l
            temp_dic = {temp_l[j]: (j+L) for j in range(len(temp_l))}
            L = L + len(temp_l)
            node_num = node_num + len(temp_l)
            max_se_len = max(len(i), max_se_len)  # 最大app sequence 长度
            min_s = min(window, len(i))
            numedge += int((1 + min_s) * len(i) / 2 - (1 + min_s - 1) * min_s / 4)

            node_dic.append(temp_dic)

            for j in temp_l:
                if j not in same_dic:
                    same_dic[j] = [temp_dic[j]]
                else:
                    same_dic[j].append(temp_dic[j])


        alias_inputs.append([[node_dic[j][i] for i in temp_s[j]] for j in range(len(temp_s))])
        numnode.append(node_num)
        num_edge.append(numedge)
        same_node_dic.append(same_dic)
        n_node.append(nodes)

    max_n_node = np.max(numnode)  # batch中最大结点数

    max_n_edge = max(num_edge) + np.max([len(dic) for dic in same_node_dic]) * session_length # batch中最大边数 + session个数



    for i in range(len(inputs)):
        node_mask = []
        cols = []
        rows = []
        edg = []
        e_idx = 0
        effect_same_node_dic = same_node_dic[i]

        node = n_node[i]
        items.append(node + (max_n_node - len(node)) * [0])

        for j in range(len(inputs[i])):
            effect_len = len(alias_inputs[i][j])  # 有效长度
            effect_list = alias_inputs[i][j]


            node_mask.append((max_se_len - len(alias_inputs[i][j])) * [0] + [1] * len(alias_inputs[i][j]))
            alias_inputs[i][j] = (max_se_len - len(alias_inputs[i][j])) * [0] + alias_inputs[i][j]


            for w in range(1 + min(window, effect_len - 1)):
                if w % 2 == 1:
                    edge_idx = list(np.arange(e_idx, e_idx + effect_len - w))
                    edg += edge_idx
                    for ww in range(w + 1):
                        rows += effect_list[ww:ww + effect_len - w]
                        cols += edge_idx

                    e_idx += len(edge_idx)

        for node in effect_same_node_dic.keys():
            if len(effect_same_node_dic[node]) > 1:
                rows += effect_same_node_dic[node]
                cols += [e_idx] * len(effect_same_node_dic[node])
                e_idx += 1



        u_H = sp.coo_matrix(([1.0] * len(rows), (rows, cols)), shape=(max_n_node, max_n_edge))
        HT.append(np.asarray(u_H.T.todense()))

        node_masks.append(node_mask)

        edge_inputs.append(edg + (max_n_edge - len(edg)) * [0])


    return alias_inputs, HT, items, node_masks