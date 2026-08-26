import argparse
import numpy as np


def str2bool(v):
    """Convert a string argument to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="3D Micro-Bump Height Estimation")

    # Training / model configuration
    parser.add_argument("--is_train", type=str2bool, default=True, help="Run in training mode")
    parser.add_argument("--model_feature", type=str, default="proposed_0.2_0.1_42", help="Model identifier used for checkpoints")
    parser.add_argument("--dataset_path", type=str, default="./data", help="Root directory of the dataset")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--initial_lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=6, help="Number of spatial regions per training batch")
    parser.add_argument("--lambda_c", type=float, default=0.2, help="Weight for the consistency loss")
    parser.add_argument("--lambda_t", type=float, default=0.1, help="Weight for the total variation loss")
    parser.add_argument("--coordconv", type=str2bool, default=True, help="Enable Coordinate Convolution")
    parser.add_argument("--use_2d_cnn", type=str2bool, default=True, help="Enable the 2D CNN refiner")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sample_5_scans", type=str2bool, default=True, help="Randomly select five repeated scans per spatial region")

    # Evaluation configuration
    parser.add_argument("--repeat_test", type=str2bool, default=False, help="Evaluate scan-to-scan repeatability")
    parser.add_argument("--batch_size_test", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--model_mode", type=str, default="_final", help="Suffix of the checkpoint to load")

    # Runtime / hardware configuration
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device used for training and evaluation")
    parser.add_argument("--cuda_mode", type=str, default="single_gpu", choices=["single_gpu", "multi_gpu"], help="GPU execution mode")
    parser.add_argument("--model_root", type=str, default="./weights", help="Directory for saving and loading checkpoints")

    # Training control
    parser.add_argument("--eval_epoch", type=int, default=5, help="Validation interval in epochs")
    parser.add_argument("--val_patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--resume", type=str2bool, default=False, help="Resume training from a checkpoint")
    parser.add_argument("--resume_path", type=str, default="", help="Path to the checkpoint used for resuming training")

    # 3D volume configuration
    parser.add_argument("--depth", type=int, default=56, help="Number of discrete height values in the 3D volume")
    parser.add_argument("--y_range", type=int, nargs=2, default=[0, 32], help="Y-axis range")

    args = parser.parse_args()

    # Derived acquisition parameters
    args.scan_params = [args.depth, 2, 2 * np.sqrt(2)]

    return args