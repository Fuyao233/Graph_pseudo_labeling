
seed_list=('0_A_B_random_hard' '0_A_B_random_soft')

for seed in "${seed_list[@]}"; do
    python produce_comprehensive_result.py --tail  "$seed"
done