import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd

def extract_bump_heights(base_name, pred_map, mask, heights_list):
    bump_records = []
    gt_heights_for_img = []
    pred_heights_for_img = []

    floor_mask = (mask == 1)
    floor_h = float(np.mean(pred_map[floor_mask])) if np.any(floor_mask) else 0.0

    for bump_idx in range(len(heights_list)):
        gt_h = float(heights_list[bump_idx])
        bump_id = bump_idx + 2
        bump_mask = (mask == bump_id)

        bump_mean = float(np.mean(pred_map[bump_mask])) if np.any(bump_mask) else 0.0
        pred_h = bump_mean - floor_h

        gt_heights_for_img.append(gt_h)
        pred_heights_for_img.append(pred_h)

        bump_records.append({
            "File Name": base_name,
            "Bump Index": bump_idx + 1,
            "GT_bump": gt_h,
            "Pred_top": bump_mean,
            "Pred_bottom": floor_h,
            "Pred_bump": pred_h
        })

    return bump_records, gt_heights_for_img, pred_heights_for_img

def calculate_and_save_bump_stats(base_dir, save_dir):
    all_dfs = []

    for num in range(1, 11):
        csv_path = os.path.join(base_dir, f"repeat_test/{num}/bump_stats.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['Scan_ID'] = num 
            all_dfs.append(df)

    if not all_dfs:
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    summary_df = combined_df.groupby(['File Name', 'Bump Index']).agg(
        GT_bump=('GT_bump', 'first'),
        Pred_top_Mean=('Pred_top', 'mean'),
        Pred_top_STD=('Pred_top', 'std'),
        Pred_bottom_Mean=('Pred_bottom', 'mean'),
        Pred_bottom_STD=('Pred_bottom', 'std'),
        Pred_bump_Mean=('Pred_bump', 'mean'),  
        Pred_bump_STD=('Pred_bump', 'std')
    ).reset_index()

    total_std_avg = summary_df['Pred_bump_STD'].mean()
    total_pred_avg = summary_df['Pred_bump_Mean'].mean()
    total_gt_avg = summary_df['GT_bump'].mean()
    total_bump_error = total_gt_avg - total_pred_avg

    total_top_avg = summary_df['Pred_top_Mean'].mean()
    total_top_std = summary_df['Pred_top_STD'].mean()
    total_bottom_avg = summary_df['Pred_bottom_Mean'].mean()
    total_bottom_std = summary_df['Pred_bottom_STD'].mean()

    save_path = os.path.join(save_dir, "repeat_test/Final_Bump_Stats.txt")
    with open(save_path, 'w') as f:
        f.write(f'total_pred_avg   | {total_pred_avg:.4f}\n')
        f.write(f'total_bump_error | {total_bump_error:.4f}\n')
        f.write(f'total_std_avg    | {total_std_avg:.4f}\n')
        f.write(f'total_top_avg    | {total_top_avg:.4f}\n')
        f.write(f'total_top_std    | {total_top_std:.4f}\n')
        f.write(f'total_bottom_avg | {total_bottom_avg:.4f}\n')
        f.write(f'total_bottom_std | {total_bottom_std:.4f}\n')
        f.write('-' * 80 + '\n')
        f.write(summary_df.to_string(index=False, float_format='%.4f'))

def calculate_and_save_mae_stats(base_dir, mae_save_dir):    
    save_dir = os.path.join(mae_save_dir, "repeat_test/MAE.txt")
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

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

        if not mae_list_per_repeat:
            mean_mae_for_repeat = np.nan
        else:
            mean_mae_for_repeat = float(np.mean(mae_list_per_repeat))

        mae_means_over_repeats.append(mean_mae_for_repeat)
        rows.append({"test_idx": str(num), "MAE": mean_mae_for_repeat})

    mean_pixel_stds_all_regions = []
    
    for name, pred_list in pixelwise_preds_dict.items():
        if len(pred_list) > 1:
            stacked_preds = np.stack(pred_list, axis=0) 
            pixel_std_map = np.std(stacked_preds, axis=0)
            mean_pixel_stds_all_regions.append(np.mean(pixel_std_map))
            
    final_pixelwise_std = float(np.mean(mean_pixel_stds_all_regions)) if mean_pixel_stds_all_regions else np.nan

    valid_maes = np.array([m for m in mae_means_over_repeats if not np.isnan(m)], dtype=float)
    
    mean_val = float(np.mean(valid_maes)) if len(valid_maes) > 0 else np.nan
    std_dev = float(np.std(valid_maes)) if len(valid_maes) > 0 else np.nan

    rows.append({"test_idx": "MEAN", "MAE": mean_val})
    rows.append({"test_idx": "STD_of_MAE", "MAE": std_dev})
    rows.append({"test_idx": "PIXEL_REPEATABILITY_STD", "MAE": final_pixelwise_std})

    df = pd.DataFrame(rows, columns=["test_idx", "MAE"])
    
    with open(save_dir, 'w') as f:
        f.write(df.to_string(index=False, float_format='%.4f'))