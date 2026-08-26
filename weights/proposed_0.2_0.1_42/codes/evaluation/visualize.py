import os
import cv2
import numpy as np

def save_bump_viz_img(save_dir, file_name, gt_map, pred_map, gt_heights, pred_heights, mask):
    if hasattr(gt_map, 'detach'):
        gt_map = gt_map.squeeze().detach().cpu().numpy()
        pred_map = pred_map.squeeze().detach().cpu().numpy()
        
    if hasattr(mask, 'detach'):
        mask = mask.squeeze().detach().cpu().numpy()

    h, w = gt_map.shape
        
    def to_gray_8u(array):
        min_val = array.min()
        max_val = array.max()
        norm = (array - min_val) / (max_val - min_val + 1e-8)
        gray = (norm * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    gt_bgr = to_gray_8u(gt_map)
    pred_bgr = to_gray_8u(pred_map)

    mosaic = np.full((h, 2 * w + 1, 3), 255, dtype=np.uint8)
    mosaic[:, 0:w, :] = gt_bgr
    mosaic[:, w+1:2*w+1, :] = pred_bgr
    cv2.line(mosaic, (w, 0), (w, h), (255, 255, 255), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3 
    color = (255, 255, 255)
    thickness = 1

    for i in range(len(gt_heights)):
        bump_id = i + 2
        y_coords, x_coords = np.where(mask == bump_id)
        
        if len(y_coords) == 0:
            continue
            
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        text_y = max(center_y - 26, 10)
        
        gt_text = f"{gt_heights[i]:.2f}"
        pred_text = f"{pred_heights[i]:.2f}"
        
        (w_gt, _), _ = cv2.getTextSize(gt_text, font, font_scale, thickness)
        gt_text_x = int(center_x - (w_gt / 2))
        cv2.putText(mosaic, gt_text, (gt_text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

        (w_pred, _), _ = cv2.getTextSize(pred_text, font, font_scale, thickness)
        pred_text_x = int(center_x + w + 1 - (w_pred / 2))
        cv2.putText(mosaic, pred_text, (pred_text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"{file_name}_viz.png"), mosaic)

def save_feature_map_npy(feature_map_tensor, file_name, save_dir, suffix):
    try:
        map_np = feature_map_tensor[0].detach().cpu().numpy() 
        
        os.makedirs(save_dir, exist_ok=True) 
        npy_path = os.path.join(save_dir, f"{file_name}_{suffix}.npy")
        np.save(npy_path, map_np)

    except Exception as e:
        print(f"Error saving feature map {suffix} for {file_name}: {e}")