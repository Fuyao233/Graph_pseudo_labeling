#!/bin/bash

# 定义dataset参数列表
datasets=("yelp-chi"  "pokec" "arxiv-year" "chameleon" "cornell" "squirrel" "texas" "wisconsin")

# 定义model_name参数列表
model_names=("mlp" "GCN")



# 遍历所有dataset
for dataset in "${datasets[@]}"; do
    # 遍历所有model_name
    # for model_name in "${model_names[@]}"; do
        # 调用train.py脚本
    echo "Training with dataset ${dataset} and model ${model_name}"
    python train.py --dataset "$dataset" 
    # done
done
