dataset_list=('cornell' 'texas' 'wisconsin' 'chameleon' 'squirrel' 'yelp-chi')
model_list=('mlp' 'GCN' 'GIN')
# model_list=('mlp' 'GCN' 'GIN' )
# edge_threshold=(0 0.2 0.4 0.6 0.8 0.9)

for dataset in "${dataset_list[@]}"; do
    for model in "${model_list[@]}"; do
        echo "$dataset" "$model"
        python train.py --dataset "$dataset" --model_name "$model" 
    done
done
