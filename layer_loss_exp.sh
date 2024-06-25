dataset_list=('cornell' 'texas' 'wisconsin' 'chameleon' 'squirrel')
model_list=('ourModel')

for dataset in "${dataset_list[@]}"; do
    for model in "${model_list[@]}"; do
        echo "$dataset" "$model"
        python layer_loss_exp.py --dataset "$dataset" --model_name "$model" 
    done
done
