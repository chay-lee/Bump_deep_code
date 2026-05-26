#!/bin/bash
set -e

DATASET_PATH="./data"
SEED=42

print_usage() {
    echo "Usage: bash run.sh [COMMAND] [VARIANT]"
    echo "Commands: train, test, all"
    echo "Variants: recon_only, recon_consis, proposed, wo_coordconv"
    exit 1
}

if [ "$#" -lt 2 ]; then print_usage; fi

COMMAND=$1
VARIANT=$2

# Map variants to hyperparameters (from Paper Table II)
case $VARIANT in
    recon_only)   LR="1e-3"; EPOCHS=130; BATCH=32; L1=0.0; L2=0.0; COORD="yes" ;;
    recon_consis) LR="1e-3"; EPOCHS=130; BATCH=6;  L1=0.1; L2=0.0; COORD="yes" ;;
    proposed)     LR="5e-4"; EPOCHS=150; BATCH=6;  L1=0.2; L2=0.1; COORD="yes" ;;
    wo_coordconv) LR="5e-4"; EPOCHS=150; BATCH=6;  L1=0.2; L2=0.1; COORD="no"  ;;
    *) echo "[Error] Unknown variant: $VARIANT"; print_usage ;;
esac

run_python() {
    local IS_TRAIN=$1
    local FULL_MODEL_NAME="${VARIANT}_${L1}_${L2}_${SEED}"
    
    if [ "$IS_TRAIN" = "yes" ]; then
        echo ">> [TRAIN] Model: $VARIANT | Seed: $SEED"
        python3 main.py --is_train yes \
            --model_feature "$FULL_MODEL_NAME" \
            --dataset_path "$DATASET_PATH" \
            --epochs "$EPOCHS" --batch_size "$BATCH" --initial_lr "$LR" \
            --lambda_val_1 "$L1" --lambda_val_2 "$L2" \
            --coordconv "$COORD" --seed "$SEED" \
            --eval_epoch 5 --rand_5 yes
    else
        echo ">> [TEST] Model: $VARIANT | Seed: $SEED"
        python3 main.py --is_train no \
            --model_feature "$FULL_MODEL_NAME" \
            --dataset_path "$DATASET_PATH" \
            --model_mode "_final" \
            --batch_size_test 32 \
            --lambda_val_1 "$L1" --lambda_val_2 "$L2" \
            --coordconv "$COORD" --seed "$SEED" \
            --repeat_test yes
    fi
}

case $COMMAND in
    train) run_python "yes" ;;
    test)  run_python "no"  ;;
    all)   run_python "yes"; run_python "no" ;;
    *)     echo "[Error] Unknown command: $COMMAND"; print_usage ;;
esac