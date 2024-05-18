# dataset_name=( yelp-chi ) # chameleon texas squirrel cornell wisconsin
dataset_name=(chameleon texas squirrel cornell wisconsin)


for j in "${dataset_name[@]}"; do
    str2="--dataset=$j"
    #echo $str1
    python find_hyperparameters.py $str2 
done

