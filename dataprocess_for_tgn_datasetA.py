import pandas as pd
import copy

if __name__ == "__main__":
    edges_train_A = pd.read_csv("data/train_csvs/edges_train_A.csv", header=None)
    edges_train_A.columns = ['src', 'tgt', 'edge_type', 'ts']
    node_features = pd.read_csv("data/train_csvs/node_features.csv", header=None)
    node_features_src = copy.deepcopy(node_features)
    node_features_src.columns = ['src'] + ['src_' + str(i) for i in range(1,9)]
    edges_train_A2 = edges_train_A.join(node_features_src, on=['src'], how='left', lsuffix='left', rsuffix='right')
    edges_train_A2.fillna(-1.0, inplace=True)

    edges_type_features = pd.read_csv("data/train_csvs/edge_type_features.csv", header=None)
    edges_type_features.columns = ['edge_type', 'edge_1', 'edge_2', 'edge_3']
    edges_train_A3 = edges_train_A2.join(edges_type_features, on=['edge_type'], how='left', lsuffix='left', rsuffix='right')

    node_features_tgt = copy.deepcopy(node_features)
    node_features_tgt.columns = ['tgt'] + ['tgt_' + str(i) for i in range(1,9)]
    edges_train_A4 = edges_train_A3.join(node_features_tgt, on=['tgt'], how='left', lsuffix='left', rsuffix='right')
    edges_train_A4.fillna(-1.0, inplace=True)
    edges_train_A5 = edges_train_A4.drop(['srcright', 'edge_typeleft', 'edge_typeright', 'tgtright'], axis='columns')
    edges_train_A6 = pd.concat([edges_train_A5.iloc[:,:3], pd.Series([1.0] * edges_train_A5.shape[0]), edges_train_A5.iloc[:,3:]], axis=1)
    edges_train_A6.columns = ['src', 'tgt', 'ts', 'label'] + ['feat_' + str(i) for i in range(8 + 3 +8)]
    edges_train_A6.to_csv("data/processed/DatasetA_train_fill-1.csv", index=False)