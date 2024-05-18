dataset_name=(pokec yelp-chi) #) #) "REDDIT-BINARY" "COLLAB" "NCI1" "NCI109" "DD" ) # "COLLAB" "NCI1" "NCI109" "DD" 

for i in "${model_list[@]}"; do 
    for j in "${dataset_name[@]}"; do
        str1="--model=$i"
        str2="--dataset=$j"
        python run_dist.py $str1 $str2 --gpu=4
    done
done