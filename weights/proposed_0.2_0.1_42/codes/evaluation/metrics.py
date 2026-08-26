import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def extract_bump_heights(base_name, pred_map, mask, heights_list):
    bump_records = []
    gt_heights_for_img = []
    pred_heights_for_img = []

    substrate_mask = (mask == 1)
    substrate_h = float(np.mean(pred_map[substrate_mask])) if np.any(substrate_mask) else 0.0

    for bump_idx in range(len(heights_list)):
        gt_peak_h = float(heights_list[bump_idx])
        bump_id = bump_idx + 2
        top_mask = (mask == bump_id)

        top_h = float(np.mean(pred_map[top_mask])) if np.any(top_mask) else 0.0
        pred_peak_h = top_h - substrate_h

        gt_heights_for_img.append(gt_peak_h)
        pred_heights_for_img.append(pred_peak_h)

        bump_records.append({
            "File Name": base_name,
            "Bump Index": bump_idx + 1,
            "GT_peak": gt_peak_h,
            "Pred_top": top_h,
            "Pred_substrate": substrate_h,
            "Pred_peak": pred_peak_h
        })

    return bump_records, gt_heights_for_img, pred_heights_for_img


def calculate_and_save_bump_stats(base_dir, save_dir):
    all_dfs = []

    for num in range(1, 11):
        csv_path = os.path.join(base_dir, f"repeat_test/{num}/bump_stats.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["Scan_ID"] = num
            all_dfs.append(df)

    if not all_dfs:
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    summary_df = combined_df.groupby(["File Name", "Bump Index"]).agg(
        GT_peak=("GT_peak", "first"),
        Pred_top_Mean=("Pred_top", "mean"),
        Pred_top_STD=("Pred_top", lambda x: np.std(x, ddof=0)),
        Pred_substrate_Mean=("Pred_substrate", "mean"),
        Pred_substrate_STD=("Pred_substrate", lambda x: np.std(x, ddof=0)),
        Pred_peak_Mean=("Pred_peak", "mean"),
        Pred_peak_STD=("Pred_peak", lambda x: np.std(x, ddof=0))
    ).reset_index()

    # E_b: mean absolute error of bump peak heights
    bump_peak_error = float(np.mean(np.abs(summary_df["Pred_peak_Mean"] - summary_df["GT_peak"])))

    # sigma_b: mean population STD across repeated scans
    bump_peak_repeatability = float(summary_df["Pred_peak_STD"].mean())

    mean_pred_peak = float(summary_df["Pred_peak_Mean"].mean())
    mean_gt_peak = float(summary_df["GT_peak"].mean())

    mean_pred_top = float(summary_df["Pred_top_Mean"].mean())
    mean_pred_top_std = float(summary_df["Pred_top_STD"].mean())

    mean_pred_substrate = float(summary_df["Pred_substrate_Mean"].mean())
    mean_pred_substrate_std = float(summary_df["Pred_substrate_STD"].mean())

    save_path = os.path.join(save_dir, "repeat_test", "bump_metrics.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        f.write(f"E_b                     | {bump_peak_error:.4f} um\n")
        f.write(f"sigma_b                 | {bump_peak_repeatability:.4f} um\n")
        f.write(f"Mean predicted peak     | {mean_pred_peak:.4f} um\n")
        f.write(f"Mean ground-truth peak  | {mean_gt_peak:.4f} um\n")
        f.write(f"Mean predicted top      | {mean_pred_top:.4f} um\n")
        f.write(f"Mean top STD            | {mean_pred_top_std:.4f} um\n")
        f.write(f"Mean substrate          | {mean_pred_substrate:.4f} um\n")
        f.write(f"Mean substrate STD      | {mean_pred_substrate_std:.4f} um\n")
        f.write("-" * 80 + "\n")
        f.write(summary_df.to_string(index=False, float_format="%.4f"))


def calculate_and_save_mae_stats(base_dir, mae_save_dir):
    save_path = os.path.join(mae_save_dir, "repeat_test", "pixel_metrics.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def load_heightmap_csv(path):
        return np.loadtxt(path, delimiter=",").astype(np.float32)

    mae_means_over_repeats = []
    rows = []

    pixelwise_preds_dict = {}

    for num in range(1, 11):
        repeat_dir = os.path.join(base_dir, f"repeat_test/{num}/map")
        gt_dir = os.path.join(repeat_dir, "gt")
        pred_dir = os.path.join(repeat_dir, "pred")

        if not os.path.isdir(gt_dir):
            continue

        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.csv")))
        pred_index = {Path(p).name: p for p in glob.glob(os.path.join(pred_dir, "*.csv"))}

        mae_list_per_repeat = []

        for gt_path in gt_files:
            name = Path(gt_path).name

            if name not in pred_index:
                continue

            pred_path = pred_index[name]

            gt = load_heightmap_csv(gt_path)
            pred = load_heightmap_csv(pred_path)

            if gt.shape != pred.shape:
                continue

            mae = np.mean(np.abs(gt - pred))
            mae_list_per_repeat.append(mae)

            if name not in pixelwise_preds_dict:
                pixelwise_preds_dict[name] = []

            pixelwise_preds_dict[name].append(pred)

        mean_mae_for_repeat = float(np.mean(mae_list_per_repeat)) if mae_list_per_repeat else np.nan

        mae_means_over_repeats.append(mean_mae_for_repeat)
        rows.append({"Scan": str(num), "Pixel_MAE": mean_mae_for_repeat})

    mean_pixel_stds_all_regions = []

    for name, pred_list in pixelwise_preds_dict.items():
        if len(pred_list) > 1:
            stacked_preds = np.stack(pred_list, axis=0)

            # Population STD (ddof=0), consistent with Eq. (14)
            pixel_std_map = np.std(stacked_preds, axis=0, ddof=0)
            mean_pixel_stds_all_regions.append(np.mean(pixel_std_map))

    pixel_repeatability = (
        float(np.mean(mean_pixel_stds_all_regions))
        if mean_pixel_stds_all_regions
        else np.nan
    )

    valid_maes = np.array(
        [m for m in mae_means_over_repeats if not np.isnan(m)],
        dtype=float
    )

    pixel_mae = float(np.mean(valid_maes)) if len(valid_maes) > 0 else np.nan
    pixel_mae_std_across_scans = float(np.std(valid_maes, ddof=0)) if len(valid_maes) > 0 else np.nan

    rows.append({"Scan": "E_p", "Pixel_MAE": pixel_mae})
    rows.append({"Scan": "STD_of_E_p", "Pixel_MAE": pixel_mae_std_across_scans})
    rows.append({"Scan": "sigma_p", "Pixel_MAE": pixel_repeatability})

    df = pd.DataFrame(rows, columns=["Scan", "Pixel_MAE"])

    with open(save_path, "w") as f:
        f.write(df.to_string(index=False, float_format="%.4f"))