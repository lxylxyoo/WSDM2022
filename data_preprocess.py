import os
import ssl
from six.moves import urllib

import pandas as pd
import numpy as np

import torch
import dgl
import pickle

# === Below data preprocessing code are based on
# https://github.com/twitter-research/tgn

# Preprocess the raw data split each features

def preprocess(PATH, data_name): 
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(PATH) as f:
        if "DatasetB" not in data_name:
            s = next(f) # 跳过head
        for idx, line in enumerate(f):
            if idx % 10000 == 0:
                print("idx:", idx)
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])  # int(e[3])

            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)

# Re index nodes for DGL convience
def reindex(df, data_name, bipartite=True):
    if data_name == "DatasetB":
        if not os.path.exists("./data/graph/%s_d_u.pkl" % data_name) or not os.path.exists("./data/graph/%s_d_i.pkl" % data_name):
            print("dict not exists")
            d_u = {}
            for idx_u, u_id in enumerate(df.u.drop_duplicates()):
                d_u[u_id] = idx_u
            df.u = df.u.apply(d_u.get)

            d_i = {}
            for idx_i, i_id in enumerate(df.i.drop_duplicates()):
                d_i[i_id] = idx_i
            df.i = df.i.apply(d_i.get)

            pickle.dump(d_u, open("./data/graph/%s_d_u.pkl" % data_name, "wb"))
            pickle.dump(d_i, open("./data/graph/%s_d_i.pkl" % data_name, "wb"))
        else:
            print("dict already exists, load and map.")
            d_u = pickle.load(open("./data/graph/%s_d_u.pkl" % data_name, "rb"))
            d_i = pickle.load(open("./data/graph/%s_d_i.pkl" % data_name, "rb"))
            df.u = df.u.apply(d_u.get)
            df.i = df.i.apply(d_i.get)

    new_df = df.copy()
    if bipartite:
        # assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        # assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))
        upper_u = len(d_u)
        # upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df

# Save edge list, features in different file for data easy process data
def run(data_name, bipartite=True, mode="train"):
    PATH = ('./data/processed/{}_%s.csv' % mode).format(data_name)
    OUT_DF = ('./data/processed/ml_{}_%s.csv' % mode).format(data_name)
    OUT_FEAT = ('./data/processed/ml_{}_%s.npy' % mode).format(data_name)
    OUT_NODE_FEAT = ('./data/processed/ml_{}_node_%s.npy' % mode).format(data_name)

    df, feat = preprocess(PATH, data_name)
    new_df = reindex(df, data_name, bipartite)

    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    max_idx = max(new_df.u.max(), new_df.i.max())
    if data_name == "DatasetA":
        rand_feat = np.zeros((max_idx + 1, 8 + 3 + 8))
    elif data_name == "DatasetB":
        rand_feat = np.zeros((max_idx + 1, 768))

    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)

# === code from twitter-research-tgn end ===

# If you have new dataset follow by same format in Jodie,
# you can directly use name to retrieve dataset

def TemporalDataset(dataset, mode="train"):
    if not os.path.exists(('./data/graph/{}_%s.bin' % mode).format(dataset)):
        print("Start Process Data ...")
        run(dataset, bipartite=False if dataset=="DatasetA" else True, mode=mode)
        raw_connection = pd.read_csv(('./data/processed/ml_{}_%s.csv' % mode).format(dataset))
        raw_feature = np.load(('./data/processed/ml_{}_%s.npy' % mode).format(dataset))
        # -1 for re-index the node
        src = raw_connection['u'].to_numpy()-1
        dst = raw_connection['i'].to_numpy()-1
        # Create directed graph
        g = dgl.graph((src, dst), num_nodes=None if mode == "train" else 69984 if dataset == "DatasetA" else 1304045)  # 当成同构图来做
        g.edata['timestamp'] = torch.from_numpy(
            raw_connection['ts'].to_numpy())
        g.edata['label'] = torch.from_numpy(raw_connection['label'].to_numpy())
        g.edata['feats'] = torch.from_numpy(raw_feature[1:, :]).float()
        dgl.save_graphs(('./data/graph/{}_%s.bin' % mode).format(dataset), [g])
    else:
        print("Data exists, directly loads.")
        gs, _ = dgl.load_graphs(('./data/graph/{}_%s.bin' % mode).format(dataset))
        g = gs[0]
    return g

def TemporalDatasetA(mode):
    return TemporalDataset('DatasetA', mode=mode)

def TemporalDatasetB(mode):
    return TemporalDataset('DatasetB', mode=mode)

if __name__ == "__main__":
    g = TemporalDatasetB("train")
    print(g)