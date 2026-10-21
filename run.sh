#!/bin/bash
set -e

# 1. 기본 설정 (queue.sh 또는 실행 명령에서 덮어쓸 수 있음)
EXPERIMENT_NAME="${EXPERIMENT_NAME:-proposed}"
SEED="${SEED:-42}"
LAMBDA_C="${LAMBDA_C:-0.2}"
LAMBDA_T="${LAMBDA_T:-0.1}"
COORDCONV="${COORDCONV:-yes}"
USE_2D_CNN="${USE_2D_CNN:-yes}"
INITIAL_LR="${INITIAL_LR:-5e-4}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-6}"
DATASET_PATH="${DATASET_PATH:-/home/dspl/CY/Bump/Dataset/coin_bump_dataset_2}"
BATCH_SIZE_TEST="${BATCH_SIZE_TEST:-32}"
REPEAT_TEST="${REPEAT_TEST:-yes}"
CUDA_MODE="${CUDA_MODE:-single_gpu}"
MODEL_MODE="${MODEL_MODE:-_final}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
RESUME="${RESUME:-no}"
RESUME_PATH="${RESUME_PATH:-}"

# 서로 다른 실험이 같은 폴더를 쓰지 않도록 이름 생성
MODEL_FEATURE="${MODEL_FEATURE:-${EXPERIMENT_NAME}_s${SEED}}_lr${INITIAL_LR}_wd${WEIGHT_DECAY}_c${LAMBDA_C}_t${LAMBDA_T}"
COMMAND="${1:-train}"

echo "Mode: $COMMAND | Experiment: $MODEL_FEATURE | GPU: ${CUDA_VISIBLE_DEVICES:-default} | $CUDA_MODE"

case "$COMMAND" in
    train)
        python3 main.py \
            --is_train yes \
            --model_feature "$MODEL_FEATURE" \
            --dataset_path "$DATASET_PATH" \
            --epochs "$EPOCHS" \
            --initial_lr "$INITIAL_LR" \
            --batch_size "$BATCH_SIZE" \
            --lambda_c "$LAMBDA_C" \
            --lambda_t "$LAMBDA_T" \
            --coordconv "$COORDCONV" \
            --use_2d_cnn "$USE_2D_CNN" \
            --seed "$SEED" \
            --sample_5_scans yes \
            --weight_decay "$WEIGHT_DECAY" \
            --cuda_mode "$CUDA_MODE" \
            --resume "$RESUME" \
            --resume_path "$RESUME_PATH"
        ;;
    test)
        python3 main.py \
            --is_train no \
            --model_feature "$MODEL_FEATURE" \
            --dataset_path "$DATASET_PATH" \
            --lambda_c "$LAMBDA_C" \
            --lambda_t "$LAMBDA_T" \
            --coordconv "$COORDCONV" \
            --use_2d_cnn "$USE_2D_CNN" \
            --seed "$SEED" \
            --batch_size_test "$BATCH_SIZE_TEST" \
            --repeat_test "$REPEAT_TEST" \
            --model_mode "$MODEL_MODE" \
            --cuda_mode "$CUDA_MODE"
        ;;
    all)
        bash "$0" train
        bash "$0" test
        ;;
    *)
        echo "Usage: bash run.sh [train|test|all]"
        exit 1
        ;;
esac