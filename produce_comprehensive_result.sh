
seed_list=('0_A_B_random' '0_A_easy_B_hard' '0_A_hard_B_easy')

for seed in "${seed_list[@]}"; do
    python produce_comprehensive_result.py --tail  "$seed"
done