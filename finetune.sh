#!/bin/bash

dataset=$1
ckpt_dir=$2
result_dir=$3
pretrained_model=$4
python train_predict.py \
	--dataset ${dataset} \
	--batch_size 128 \
	--epochs 30 \
    --n_neighbors 10 \
    --lr 0.0005 \
    --predict \
	--fast_mode \
	--ckpt_dir ${ckpt_dir} \
	--result_dir ${result_dir} \
    --finetune_model ${pretrained_model} \
    --mode fine-tuning
