import pandas as pd
import copy

if __name__ == "__main__":
    input_A_initial = pd.read_csv("data/test_csvs/input_A_initial.csv", header=None)
    input_A_initial.columns = ['src', 'tgt', 'edge_type', 'ts_start', 'ts_end', 'label']
    node_features = pd.read_csv("data/train_csvs/node_features.csv", header=None)
    node_features_src = copy.deepcopy(node_features)
    node_features_src.columns = ['src'] + ['src_' + str(i) for i in range(1,9)]
    input_A_initial2 = input_A_initial.join(node_features_src, on=['src'], how='left', lsuffix='left', rsuffix='right')
    input_A_initial2.fillna(-1.0, inplace=True)

    edges_type_features = pd.read_csv("data/train_csvs/edge_type_features.csv", header=None)
    edges_type_features.columns = ['edge_type', 'edge_1', 'edge_2', 'edge_3']
    input_A_initial3 = input_A_initial2.join(edges_type_features, on=['edge_type'], how='left', lsuffix='left', rsuffix='right')

    node_features_tgt = copy.deepcopy(node_features)
    node_features_tgt.columns = ['tgt'] + ['tgt_' + str(i) for i in range(1,9)]
    input_A_initial4 = input_A_initial3.join(node_features_tgt, on=['tgt'], how='left', lsuffix='left', rsuffix='right')
    input_A_initial4.fillna(-1.0, inplace=True)
    input_A_initial5 = input_A_initial4.drop(['srcright', 'edge_typeleft', 'edge_typeright', 'tgtright'], axis='columns')
    input_A_initial6_s = pd.concat([input_A_initial5.iloc[:,:5], input_A_initial5.iloc[:,5:]], axis=1).drop(['ts_end'], axis='columns')
    input_A_initial6_e = pd.concat([input_A_initial5.iloc[:,:5], input_A_initial5.iloc[:,5:]], axis=1).drop(['ts_start'], axis='columns')
    input_A_initial6_s.columns = ['src', 'tgt', 'ts', 'label'] + ['feat_' + str(i) for i in range(8 + 3 +8)]
    input_A_initial6_e.columns = ['src', 'tgt', 'ts', 'label'] + ['feat_' + str(i) for i in range(8 + 3 +8)]
    input_A_initial6_s.to_csv("data/processed/DatasetA_test_start_fill-1.csv", index=False)
    input_A_initial6_e.to_csv("data/processed/DatasetA_test_end_fill-1.csv", index=False)