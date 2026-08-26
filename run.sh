#!/bin/bash
set -e

# =============================================================================
# Experiment configuration
# =============================================================================

# Experiment name
EXPERIMENT_NAME="proposed"

# Reproducibility
SEED=42

# Loss weights
LAMBDA_C=0.2
LAMBDA_T=0.1

# Model configuration
COORDCONV="yes"
USE_2D_CNN="yes"

# Training hyperparameters
INITIAL_LR="5e-4"
EPOCHS=150
BATCH_SIZE=6

# Dataset
DATASET_PATH="./data"

# Evaluation
BATCH_SIZE_TEST=32
REPEAT_TEST="yes"

# =============================================================================
# Derived configuration
# =============================================================================

MODEL_FEATURE="${EXPERIMENT_NAME}_${LAMBDA_C}_${LAMBDA_T}_${SEED}"

COMMAND=${1:-train}

# =============================================================================
# Run
# =============================================================================

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
            --sample_5_scans yes
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
            --model_mode "_final"
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