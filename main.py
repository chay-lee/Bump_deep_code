import os
import random
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import numpy as np
import datetime
from pathlib import Path
from evaluation.pixel_metrics import PixelEvaluator
from evaluation.bump_metrics import BumpEvaluator

from utils.args import parse_args
from utils.dataset import load_datasets, make_dataloaders
from models.loss import get_loss_fn
from utils.torch_utils import load_checkpoint
from models.network import HeightEstimationNetwork

import train as train_module
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def update_args_for_test(checkpoint, current_args):
    from utils.args import str2bool
    saved = checkpoint.get("args", {})
    if not isinstance(saved, dict): saved = vars(saved)

    for key in ("coordconv", "use_2d_cnn", "depth", "y_range", "scan_params"):
        if key in saved:
            value = str2bool(saved[key]) if key in ("coordconv", "use_2d_cnn") else saved[key]
            setattr(current_args, key, value)
        elif key in ("coordconv", "use_2d_cnn"):
            raise ValueError(f"Checkpoint is missing architecture setting: {key}")
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


def run_test(model, args, scan_id, output_dir, checkpoint, pixel_evaluator, bump_evaluator):
    scan_name = f"repeat_test/{scan_id}" if args.repeat_test else "test"
    dataset_path = os.path.join(args.dataset_path, scan_name) if args.repeat_test else args.dataset_path
    print(f"\nRunning test: {scan_name}")

    dataset = load_datasets(dataset_path, args, is_train=False)
    if not len(dataset): raise ValueError(f"No test volumes: {dataset.volume_root}")
    loader = make_dataloaders(dataset, args.batch_size_test, False, args.num_workers)
    os.makedirs(output_dir, exist_ok=True)

    _, eval_str = train_module.predict_model(
        dataset_type=scan_name, check_time=True, model=model, loss_fn=None,
        epoch=checkpoint["epoch"], data_loader=loader, height_dir=str(output_dir),
        pixel_evaluator=pixel_evaluator, bump_evaluator=bump_evaluator, scan_id=scan_id
    )
    with open(os.path.join(output_dir, "test_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"model{args.model_mode} / {scan_name}\n{eval_str}\n")
    print(f"Saved: {output_dir}")


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

        backup_dir = os.path.join(MODEL_DIR, "codes")
        if os.path.exists(backup_dir): shutil.rmtree(backup_dir)
        shutil.copytree(".", backup_dir, ignore=shutil.ignore_patterns(".git", "data", "Dataset", "coin_bump_dataset", "weights", "venv", ".venv", "__pycache__", "*.pyc"))

        loss_fn = get_loss_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
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
        train_loader, val_loader = make_dataloaders(dataset, args.batch_size, True, args.num_workers)

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
        args = update_args_for_test(checkpoint, args)
        train_module.args = args

        model = HeightEstimationNetwork(args)
        state = checkpoint["model"]
        if state and all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state)
        model = setup_model_on_device(model, args)
        model.eval()

        scans = tuple(range(1, 11)) if args.repeat_test else (1,)
        split = "repeat_test" if args.repeat_test else "test"
        result_dir = Path(MODEL_DIR) / f"model{args.model_mode}" / split
        gt_root = Path(args.dataset_path) / split
        reference_dir = gt_root / "1" / "interfero" if args.repeat_test else gt_root / "interfero"

        pixel = PixelEvaluator(scans, reference_dir, result_dir)
        with BumpEvaluator(scans, gt_root, result_dir, args.eval_workers, args.repeat_test) as bump:
            for scan_id in scans:
                output_dir = result_dir / str(scan_id) if args.repeat_test else result_dir
                run_test(model, args, scan_id, output_dir, checkpoint, pixel, bump)
            metrics = {**pixel.finish(), **bump.finish()}

        lines = [
            "TEST METRICS",
            f"Model: {args.model_feature} | Checkpoint: {args.model_mode}",
            "Unit: 10^-2 um | STD: ddof=0",
            "Cohorts and exclusions: pixel_metrics.txt / height_summary.txt",
        ]
        lines += [f"{key}: {100*metrics[key]:.6f}" for key in ("E_p", "sigma_p", "E_b", "sigma_b", "sigma_s")]
        (result_dir / "test_summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
        print("\n".join(lines))
        print(f"\nSaved: {result_dir / 'test_summary.txt'}")