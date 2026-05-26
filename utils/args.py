import argparse
import numpy as np

def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="3D Micro-Bump Height Estimation")

    # 1. Dynamic Arguments (Controlled by run.sh based on Model Variant)
    parser.add_argument("--is_train", type=str2bool, default=True)
    parser.add_argument("--model_feature", type=str, default="proposed_0.2_0.1_42", help="Model variant name")
    parser.add_argument("--dataset_path", type=str, default="./data", help="Root directory of the dataset")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--initial_lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lambda_val_1", type=float, default=0.2, help="Weight for consistency loss")
    parser.add_argument("--lambda_val_2", type=float, default=0.1, help="Weight for TV loss")
    parser.add_argument("--coordconv", type=str2bool, default=True, help="Enable Coordinate Convolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--rand_5", type=str2bool, default=False, help="Randomly select 5 scans for training")
    
    # Testing specific
    parser.add_argument("--repeat_test", type=str2bool, default=False, help="Evaluate across repeated scans")
    parser.add_argument("--batch_size_test", type=int, default=32)
    parser.add_argument("--model_mode", type=str, default="_final", help="Suffix of the checkpoint to load")


    # 2. Static / Fixed Arguments (Project specific constants)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_root", type=str, default="./weights", help="Directory to save/load checkpoints")
    parser.add_argument("--eval_epoch", type=int, default=5, help="Validation frequency")
    parser.add_argument("--val_patience", type=int, default=10, help="Early stopping patience")
    
    # 3D Domain specific parameters
    parser.add_argument("--depth", type=int, default=56, help="Depth of the 3D volume (D)")
    parser.add_argument("--y_range", type=int, nargs=2, default=[0, 32], help="Y-axis range")

    args = parser.parse_args()

    # Derived parameters
    args.scan_params = [args.depth, 2, 2 * np.sqrt(2)]

    return args