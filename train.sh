dataset_list=('cornell' 'texas' 'wiconsin' 'chameleon' 'squirrel' 'yelp-chi' 'deezer-europe' )
# dataset_list=('wisconsin')
# model_list=('GraphSAGE' 'ourModel' 'H2GCN')
model_list=('ourModel')
# model_list=('mlp' 'GCN' 'GIN' )

for dataset in "${dataset_list[@]}"; do
    for model in "${model_list[@]}"; do
        echo "$dataset"
        python train.py --dataset "$dataset" --model_name "$model"  
    done
done
