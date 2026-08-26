import os
import random
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import numpy as np
import datetime

from utils.args import parse_args
from utils.dataset import load_datasets, make_dataloaders
from models.loss import get_loss_fn
from utils.torch_utils import load_checkpoint
from models.network import HeightEstimationNetwork

import train as train_module
from evaluation.metrics import calculate_and_save_mae_stats, calculate_and_save_bump_stats


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def update_args_for_test(checkpoint, current_args):
    """Preserve current run.sh arguments and add only missing checkpoint arguments."""
    checkpoint_args = checkpoint.get("args", {})

    if not isinstance(checkpoint_args, dict):
        checkpoint_args = vars(checkpoint_args)

    for key, value in checkpoint_args.items():
        if not hasattr(current_args, key):
            setattr(current_args, key, value)

    return current_args


def setup_model_on_device(model, args):
    if args.device == "cuda" and torch.cuda.is_available():
        cuda_mode = getattr(args, "cuda_mode", "multi_gpu")

        if cuda_mode == "single_gpu" or torch.cuda.device_count() == 1:
            model = torch.nn.DataParallel(model, device_ids=[0])
            print("Using single GPU...")
        else:
            model = torch.nn.DataParallel(model)
            print("Using multi GPU...")

        model = model.to(args.device)
    else:
        print("Using CPU...")

    return model


def run_test(model, args, scan_id, output_dir, checkpoint):
    scan_name = f"repeat_test/{scan_id}" if getattr(args, "repeat_test", False) else "test"
    print(f"\n[!] Running test for {scan_name}...")

    # Set dataset path dynamically for repeated tests
    if getattr(args, "repeat_test", False):
        dataset_path = os.path.join(getattr(args, "dataset_path_base", args.dataset_path), scan_name)
    else:
        dataset_path = args.dataset_path

    dataset = load_datasets(dataset_path, args, is_train=False)
    test_loader = make_dataloaders(dataset, args.batch_size_test, False)

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "test_result.txt"), "a") as txt_file:
        _, eval_str = train_module.predict_model(
            dataset_type=scan_name,
            check_time=True,
            model=model,
            loss_fn=checkpoint["loss"],
            epoch=checkpoint["epoch"],
            data_loader=test_loader,
            height_dir=output_dir
        )

        test_result_str = f"\n\nmodel{args.model_mode} evaluation on {scan_name}"
        test_result_str += eval_str
        txt_file.write(test_result_str)

        print(f"Test for {scan_name} finished and results saved.")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    args = parse_args()
    train_module.args = args

    print(f"Mode: {'Train' if args.is_train else 'Test'}")
    print(f"Feature: {args.model_feature}")
    print(f"Data Path: {args.dataset_path}")
    print(f"LR: {args.initial_lr}, Batch: {args.batch_size}")
    print(f"Lambda_c: {args.lambda_c}, Lambda_t: {args.lambda_t}\n")

    MODEL_DIR = os.path.join(args.model_root, args.model_feature)

    # Set fixed seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    random.seed(seed)

    # ==========================================
    # Train Mode
    # ==========================================
    if args.is_train:
        model = HeightEstimationNetwork(args)
        model = setup_model_on_device(model, args)

        os.makedirs(MODEL_DIR, exist_ok=True)

        # Backup current code state
        backup_dir = os.path.join(MODEL_DIR, "codes")

        if os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir)

        shutil.copytree(
            os.getcwd(),
            backup_dir,
            ignore=shutil.ignore_patterns(
                ".git",
                "data",
                "weights",
                "venv",
                ".venv",
                "__pycache__",
                "*.pyc"
            )
        )

        loss_fn = get_loss_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-5
        )

        start_epoch = 0
        resume = getattr(args, "resume", False)

        if resume:
            resume_path = getattr(args, "resume_path", "")

            if not os.path.exists(resume_path):
                print(f"\n[Warning] Resume path {resume_path} not found")
            else:
                print(f"\nResuming training from checkpoint: {resume_path}")

                checkpoint = torch.load(
                    resume_path,
                    map_location=args.device,
                    weights_only=False
                )

                model.module.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_dict"])

                start_epoch = checkpoint["epoch"] + 1

                print(f"Starting from Epoch {start_epoch}\n")

        dataset = load_datasets(args.dataset_path, args, is_train=True)
        train_loader, val_loader = make_dataloaders(
            dataset,
            args.batch_size,
            is_train=True
        )

        log_mode = "a" if start_epoch > 0 else "w"

        with open(os.path.join(MODEL_DIR, "training_log.txt"), log_mode) as txt_file:
            if log_mode == "a":
                current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                txt_file.write(
                    f"\n\n[INFO] Training resumed from Epoch {start_epoch} "
                    f"at {current_time_str}\n"
                )

            train_module.train(
                train_loader,
                val_loader,
                model,
                txt_file,
                loss_fn,
                optimizer,
                MODEL_DIR,
                lr_scheduler,
                start_epoch
            )

    # ==========================================
    # Test Mode
    # ==========================================
    else:
        checkpoint, _ = load_checkpoint(MODEL_DIR, args.model_mode)

        # Current run.sh arguments have priority.
        # Checkpoint arguments are used only for settings missing from current args.
        args = update_args_for_test(checkpoint, args)
        train_module.args = args

        model = HeightEstimationNetwork(args)
        model.load_state_dict(checkpoint["model"])
        model = setup_model_on_device(model, args)
        model.eval()

        results_root = os.path.join(MODEL_DIR, f"model{args.model_mode}")

        if getattr(args, "repeat_test", False):
            args.dataset_path_base = args.dataset_path

            for scan_id in range(1, 11):
                output_dir = os.path.join(
                    results_root,
                    "repeat_test",
                    str(scan_id)
                )

                run_test(
                    model,
                    args,
                    str(scan_id),
                    output_dir,
                    checkpoint
                )
        else:
            output_dir = os.path.join(results_root, "test")
            run_test(model, args, "test", output_dir, checkpoint)

        calculate_and_save_mae_stats(
            base_dir=results_root,
            mae_save_dir=results_root
        )

        calculate_and_save_bump_stats(
            base_dir=results_root,
            save_dir=results_root
        )

        print(f"\nTest finished for: {args.model_feature}")