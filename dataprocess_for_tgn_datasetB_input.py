import pandas as pd
import copy
import numpy as np
if __name__ == "__main__":
    input_B_initial = pd.read_csv("data/test_csvs/input_B.csv", header=None)
    input_B_initial.columns = ['src', 'tgt', 'edge_type', 'ts_start', 'ts_end']
    input_B_initial2 = pd.concat([input_B_initial, pd.Series([1.0] * input_B_initial.shape[0]), pd.DataFrame(np.zeros([input_B_initial.shape[0], 768]))], axis=1)
    input_B_initial_s = input_B_initial2.drop(['edge_type', 'ts_end'], axis='columns')
    input_B_initial_e = input_B_initial2.drop(['edge_type', 'ts_start'], axis='columns')
    input_B_initial_s.columns = ['src', 'tgt', 'ts', 'label'] + ['feat_' + str(i) for i in range(768)]
    input_B_initial_e.columns = ['src', 'tgt', 'ts', 'label'] + ['feat_' + str(i) for i in range(768)]
    input_B_initial_s.to_csv("data/processed/DatasetB_input_start.csv", index=False, header=False)
    input_B_initial_e.to_csv("data/processed/DatasetB_input_end.csv", index=False, header=False)