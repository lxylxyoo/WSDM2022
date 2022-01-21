import pandas as pd
import copy
import numpy as np

if __name__ == "__main__":
    edges_train_B = pd.read_csv("data/train_csvs/edges_train_B.csv", header=None)
    edges_train_B = pd.concat([edges_train_B[[0,1,3]], pd.Series([1.0] * edges_train_B.shape[0]), edges_train_B[[4]]], axis=1)
    edges_train_B.columns = ['src', 'tgt', 'ts', 'label', 'feat']
    edges_train_B.fillna(",".join(["0.0"] * 768), inplace=True)
    with open("data/processed/DatasetB_train.csv", "w") as fout:
        for i, row in edges_train_B.iterrows():
            fout.write(",".join((str(i)for i in row)) + "\n")