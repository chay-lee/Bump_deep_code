#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

usage() {
    cat <<'HELP'
Usage: bash run.sh train|test|all
  EXPERIMENT_NAME=full|recon|consis|wo2d|wocoord
  MODEL_MODE=final|mini|both
  DATASET_PATH=./data MODEL_ROOT=./outputs
  MODEL_FEATURE=<existing folder name> for an existing model
All relative paths are resolved from the project directory.
HELP
}

COMMAND="${1:-help}"
case "$COMMAND" in
    -h|--help|help) usage; exit 0 ;;
    train|test|all) ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Use environment variables for options." >&2; exit 2; fi

# Paths and runtime
PYTHON="${PYTHON:-python3}"
DATASET_PATH="${DATASET_PATH:-./data}"
MODEL_ROOT="${MODEL_ROOT:-./outputs}"
DEVICE="${DEVICE:-cuda}"
CUDA_MODE="${CUDA_MODE:-single_gpu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

# Architecture and loss presets
EXPERIMENT_NAME="${EXPERIMENT_NAME:-full}"
default_c=0.2; default_t=0.1; default_cc=yes; default_2d=yes

case "$EXPERIMENT_NAME" in
    full) ;;
    recon) default_c=0; default_t=0 ;;
    consis) default_t=0 ;;
    wo2d) default_2d=no ;;
    wocoord) default_cc=no ;;
    *) echo "Unknown EXPERIMENT_NAME: $EXPERIMENT_NAME" >&2; exit 2 ;;
esac

LAMBDA_C="${LAMBDA_C:-$default_c}"
LAMBDA_T="${LAMBDA_T:-$default_t}"
COORDCONV="${COORDCONV:-$default_cc}"
USE_2D_CNN="${USE_2D_CNN:-$default_2d}"

# Training
SEED="${SEED:-42}"
INITIAL_LR="${INITIAL_LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-6}"
SAMPLE_5_SCANS="${SAMPLE_5_SCANS:-yes}"
EVAL_EPOCH="${EVAL_EPOCH:-5}"
VAL_PATIENCE="${VAL_PATIENCE:-5}"
RESUME="${RESUME:-no}"
RESUME_PATH="${RESUME_PATH:-}"

# Test
BATCH_SIZE_TEST="${BATCH_SIZE_TEST:-32}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
REPEAT_TEST="${REPEAT_TEST:-yes}"
MODEL_MODE="${MODEL_MODE:-final}"
MODEL_MODE="${MODEL_MODE#_}"

case "$MODEL_MODE" in
    final|mini|both) ;;
    *) echo "MODEL_MODE must be final, mini or both." >&2; exit 2 ;;
esac

MODEL_FEATURE="${MODEL_FEATURE:-${EXPERIMENT_NAME}_s${SEED}_lr${INITIAL_LR}_wd${WEIGHT_DECAY}_c${LAMBDA_C}_t${LAMBDA_T}}"
if [[ "$MODEL_FEATURE" == */* || "$MODEL_FEATURE" == . || "$MODEL_FEATURE" == .. ]]; then
    echo "MODEL_FEATURE must be a folder name, not a path." >&2; exit 2
fi

common=(--model_feature "$MODEL_FEATURE" --model_root "$MODEL_ROOT" --dataset_path "$DATASET_PATH"
        --seed "$SEED" --device "$DEVICE" --cuda_mode "$CUDA_MODE")

train_args=(--is_train yes --epochs "$EPOCHS" --initial_lr "$INITIAL_LR" --weight_decay "$WEIGHT_DECAY"
            --batch_size "$BATCH_SIZE" --lambda_c "$LAMBDA_C" --lambda_t "$LAMBDA_T"
            --coordconv "$COORDCONV" --use_2d_cnn "$USE_2D_CNN" --sample_5_scans "$SAMPLE_5_SCANS"
            --eval_epoch "$EVAL_EPOCH" --val_patience "$VAL_PATIENCE" --resume "$RESUME" --resume_path "$RESUME_PATH")

test_args=(--is_train no --batch_size_test "$BATCH_SIZE_TEST"
           --repeat_test "$REPEAT_TEST" --eval_workers "$EVAL_WORKERS")

printf '\nMode: %s | Model: %s\nData: %s\nOutput: %s/%s\n\n' \
       "$COMMAND" "$MODEL_FEATURE" "$DATASET_PATH" "$MODEL_ROOT" "$MODEL_FEATURE"

if [[ "$COMMAND" == train || "$COMMAND" == all ]]; then
    "$PYTHON" main.py "${common[@]}" "${train_args[@]}"
fi

if [[ "$COMMAND" == test || "$COMMAND" == all ]]; then
    modes=("$MODEL_MODE")
    if [[ "$MODEL_MODE" == both ]]; then modes=(final mini); fi

    for mode in "${modes[@]}"; do
        "$PYTHON" main.py "${common[@]}" "${test_args[@]}" --model_mode "_$mode"
    done
fi