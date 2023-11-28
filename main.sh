# datanames=("twitch-e" "fb100" "ogbn-proteins" "deezer-europe" "arxiv-year" "pokec" "snap-patents"
#            "yelp-chi" "ogbn-arxiv" "ogbn-products" "Cora" "CiteSeer" "PubMed" "chameleon" "cornell"
#            "film" "squirrel" "texas" "wisconsin" "genius" "twitch-gamer" "wiki")


datanames=("twitch-e") 
train_ratios=(0.2 0.4 0.6 0.8) 

for dataset in "${datanames[@]}"; do
    for ratio in "${train_ratios[@]}"; do
        python main.py --dataset "$dataset" --train_ratio "$ratio"
    done
done