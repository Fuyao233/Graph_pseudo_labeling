#!/bin/bash

# 定义dataset参数列表
# datasets=("yelp-chi" "squirrel" "chameleon" "texas" "cornell" "wisconsin")
datasets=("chameleon" "texas" "cornell" "wisconsin")

# 定义model_name参数列表
# model_names=("mlp" "GCN" "ourModel")
model_names=("ourModel")

seed_list=(0)


# 遍历所有dataset
for dataset in "${datasets[@]}"; do
    for model in "${model_names[@]}"; do
        for seed in "${seed_list[@]}"; do

                python train.py --dataset "$dataset" --model_name "$model" --seed  "$seed" 

        done
    done
done
