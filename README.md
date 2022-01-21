# WSDM2022

This is the code and instruction of TopoLab for ```WSDM CUP 2022```, task ```Temporal Link Prediction```.

## Requirements

* python3.7
* dgl == 0.6.1
* pytorch  == 1.7.0
* pandas
* numpy
* sklearn



## Run

1. Put data to correct path

&ensp;&ensp;```cp training files (edges_train_A.csv, edges_train_B.csv, edge_type_features.csv, node_features.csv) to dir data/train_csvs/```

&ensp;&ensp;```cp test files (input_A.csv, input_A_initial.py, input_B.csv, input_B_initial.csv to dir data/test_csvs/) ```

&ensp;&ensp;```cp final files (input_A.csv, input_B.csv) to dir data/final_test_csvs/```

2. Data preprocessing

&ensp;&ensp;```sh preprocess.sh```

3. Convert csv to dgl graph & training.

&ensp;&ensp;```sh train.sh DatasetA final final```

4. Fine-tune trained DatasetA model from step3,  replace `checkpoint_DatasetA_xx.pt` with the specific model you want to finetune.  You could find the trained models in PATH `model/DatasetA/pre-training/final/`

&ensp;&ensp;```sh finetune.sh DatasetA final final checkpoint_DatasetA_xx.pt```

5. Convert csv to dgl graph & training.

&ensp;&ensp;```sh train.sh DatasetB final final```

6.  The resulted final prediction files can be found in PATH `result/DatasetA/fine-tuning/final/` and `result/DatasetB/pre-training/final/`.





## File Structure
```
.
├── data
│   ├── final_test_csvs
│   │   ├── input_A.csv
│   │   ├── input_B.csv
│   │   ├── scale_prob_final_A.csv
│   │   └── scale_prob_final_B.csv
│   ├── graph
│   ├── processed
│   ├── test_csvs
│   │   ├── input_A.csv
│   │   ├── input_A_initial.csv
│   │   ├── input_B.csv
│   │   └── input_B_initial.csv
│   └── train_csvs
│       ├── edges_train_A.csv
│       ├── edges_train_B.csv
│       ├── edge_type_features.csv
│       └── node_features.csv
├── dataloading.py
├── data_preprocess.py
├── dataprocess_for_tgn_datasetA_final.py
├── dataprocess_for_tgn_datasetA_initial.py
├── dataprocess_for_tgn_datasetA_input.py
├── dataprocess_for_tgn_datasetA.py
├── dataprocess_for_tgn_datasetB_final.py
├── dataprocess_for_tgn_datasetB_initial.py
├── dataprocess_for_tgn_datasetB_input.py
├── dataprocess_for_tgn_datasetB.py
├── final.sh
├── finetune.sh
├── modules.py
├── predict.sh
├── preprocess.sh
├── tgn.py
├── train_predict.py
├── train.py
└── train.sh
```
