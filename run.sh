#!/bin/bash
set -e

DATASET_PATH="/home/dspl/CY/Bump/Dataset/ATI_coin_bump_5"

# ---------------------------------------------------------
run_train() {
    local model_name=$1
    local seed=$2
    local lr=$3
    local epochs=$4
    local batch_size=$5
    local lambda_val_1=$6
    local lambda_val_2=$7
    local coordconv=$8
    
    local full_model_name="${model_name}_${lambda_val_1}_${lambda_val_2}_${seed}"
    
    echo ">> [Train] Model: ${full_model_name} | Seed: ${seed} | LR: ${lr} | L1: ${lambda_val_1} | L2: ${lambda_val_2}"
    python3 main.py --is_train yes \
        --model_feature "$full_model_name" \
        --dataset_path "$DATASET_PATH" \
        --epochs $epochs --batch_size $batch_size \
        --initial_lr "$lr" --eval_epoch 5 --seed "$seed" \
        --lambda_val_1 "$lambda_val_1" \
        --lambda_val_2 "$lambda_val_2" \
        --rand_5 yes \
        --coordconv "$coordconv"
}

run_test() {
    local model_name=$1
    local seed=$2
    local mode=$3
    local lambda_val_1=$4
    local lambda_val_2=$5
    local coordconv=$6
    
    # 모델명에 람다 1, 2 모두 포함되도록 수정 (Train과 동일해야 체크포인트를 찾음)
    local full_model_name="${model_name}_${lambda_val_1}_${lambda_val_2}_${seed}"
    
    echo ">> [Test] Model: ${full_model_name} | Seed: ${seed} | Mode: ${mode} | L1: ${lambda_val_1} | L2: ${lambda_val_2}"
    python3 main.py --is_train no \
        --model_feature "$full_model_name" \
        --dataset_path "$DATASET_PATH" \
        --model_mode "$mode" \
        --batch_size_test 32 \
        --repeat_test yes --seed "$seed" \
        --lambda_val_1 "$lambda_val_1" \
        --lambda_val_2 "$lambda_val_2" \
        --coordconv "$coordconv"
}

# ---------------------------------------------------------

#sed -i 's/\r$//' run_bash.sh
# du -sh ~/CY/* 2>/dev/null | sort -hr | head -n 10

#인자 순서 -> model_name / seed / lr / epochs / batch_size / lambda_val_1 / lambda_val_2 / coordconv

# run_train "0c" 0 "1e-3" 150 6 0.1 0.1 yes
# run_test "0c" 0 "_final" 0.1 0.1 yes

# run_train "0d" 42 "5e-4" 150 6 0.2 0.1 no
# run_test "0d" 42 "_final" 0.2 0.1 no

# run_train "0b_2" 0 "5e-4" 150 6 0.2 0.0 yes
# run_test "0b_2" 0 "_final" 0.2 0.0 yes

# run_train "0d" 0 "5e-4" 150 6 0.2 0.1 no
# run_test "0d" 0 "_final" 0.2 0.1 no

# run_train "0d" 26 "5e-4" 150 6 0.2 0.1 no
# run_test "0d" 26 "_final" 0.2 0.1 no

# run_test "0a_r_only" 26 "_final" 0.0 0.0 yes

run_train "0b_2" 42 "5e-4" 150 6 0.2 0.0 yes
run_test "0b_2" 42 "_final" 0.2 0.0 yes

run_train "0c_3" 42 "5e-4" 150 6 0.2 0.2 yes
run_test "0c_3" 42 "_final" 0.2 0.2 yes

run_train "0b_2" 26 "5e-4" 150 6 0.2 0.0 yes
run_test "0b_2" 26 "_final" 0.2 0.0 yes