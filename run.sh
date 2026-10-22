#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Paths and runtime
PYTHON="python3"
DATASET_PATH="./data"
MODEL_ROOT="./outputs"
GPU_IDS="0"
DEVICE="cuda"
CUDA_MODE="single_gpu"
NUM_WORKERS=4
EVAL_WORKERS=4

# Training
EXPERIMENT_NAME="full"
MODEL_FEATURE=""                 # Empty: automatic name; otherwise: exact model folder name
SEED=42
INITIAL_LR=5e-4
WEIGHT_DECAY=1e-5
LAMBDA_C=0.2
LAMBDA_T=0.1
COORDCONV="yes"
USE_2D_CNN="yes"
EPOCHS=150
BATCH_SIZE=6
SAMPLE_5_SCANS="yes"
EVAL_EPOCH=5
VAL_PATIENCE=5
RESUME="no"
RESUME_PATH=""

# Test
MODEL_MODE="final"               # final, mini, both
BATCH_SIZE_TEST=32
REPEAT_TEST="yes"

COMMAND="${1:-help}"
case "$COMMAND" in
    train|test|all) ;;
    help|-h|--help) echo "Edit run.sh settings, then run: bash run.sh train|test|all"; exit 0 ;;
    *) echo "Usage: bash run.sh train|test|all" >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Set options in run.sh." >&2; exit 2; fi

MODEL_MODE="${MODEL_MODE#_}"
case "$MODEL_MODE" in
    final|mini|both) ;;
    *) echo "MODEL_MODE must be final, mini or both." >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
MODEL_FEATURE="${MODEL_FEATURE:-${EXPERIMENT_NAME}_s${SEED}_lr${INITIAL_LR}_wd${WEIGHT_DECAY}_c${LAMBDA_C}_t${LAMBDA_T}}"

if [[ "$MODEL_FEATURE" == */* || "$MODEL_FEATURE" == . || "$MODEL_FEATURE" == .. ]]; then
    echo "MODEL_FEATURE must be a folder name, not a path." >&2; exit 2
fi

common=(--model_feature "$MODEL_FEATURE" --model_root "$MODEL_ROOT" --dataset_path "$DATASET_PATH"
        --seed "$SEED" --device "$DEVICE" --cuda_mode "$CUDA_MODE" --num_workers "$NUM_WORKERS")

printf '\nMode: %s | Model: %s\nData: %s\nOutput: %s/%s\n\n' \
       "$COMMAND" "$MODEL_FEATURE" "$DATASET_PATH" "$MODEL_ROOT" "$MODEL_FEATURE"

if [[ "$COMMAND" == train || "$COMMAND" == all ]]; then
    "$PYTHON" main.py "${common[@]}" --is_train yes \
        --epochs "$EPOCHS" --initial_lr "$INITIAL_LR" --weight_decay "$WEIGHT_DECAY" --batch_size "$BATCH_SIZE" \
        --lambda_c "$LAMBDA_C" --lambda_t "$LAMBDA_T" --coordconv "$COORDCONV" --use_2d_cnn "$USE_2D_CNN" \
        --sample_5_scans "$SAMPLE_5_SCANS" --eval_epoch "$EVAL_EPOCH" --val_patience "$VAL_PATIENCE" \
        --resume "$RESUME" --resume_path "$RESUME_PATH"
fi

if [[ "$COMMAND" == test || "$COMMAND" == all ]]; then
    modes=("$MODEL_MODE")
    if [[ "$MODEL_MODE" == both ]]; then modes=(final mini); fi

    for mode in "${modes[@]}"; do
        "$PYTHON" main.py "${common[@]}" --is_train no --model_mode "_$mode" \
            --batch_size_test "$BATCH_SIZE_TEST" --repeat_test "$REPEAT_TEST" --eval_workers "$EVAL_WORKERS"
    done
fi