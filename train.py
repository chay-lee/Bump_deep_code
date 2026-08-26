import datetime
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.torch_utils import save_model, time_sync
from evaluation.visualize import save_feature_map_npy, save_bump_viz_img
from evaluation.metrics import extract_bump_heights


# Set from main.py
args = None


def train(train_loader, val_loader, model, txt_file, loss_fn, optimizer, MODEL_DIR, lr_scheduler, start_epoch=0):

    scaler = torch.amp.GradScaler('cuda', enabled=(args.device != 'cpu'))

    verbose = getattr(args, 'verbose', False)
    val_loss_obs = Score_Observer('val_loss', verbose)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        txt_file.write(f'\n\n\nTrain epoch {epoch}  ({current_time})')
        txt_file.flush()

        sum_total, sum_recon, sum_consis, sum_tv = 0.0, 0.0, 0.0, 0.0
        n_samples = 0

        for i, data in enumerate(train_loader):
            inputs, interferos, mask, heights, file_name = data

            inputs = inputs.to(args.device, non_blocking=True).float()
            interferos = interferos.to(args.device, non_blocking=True).float()
            mask = mask.to(args.device, non_blocking=True)

            is_sequence = (inputs.dim() == 6)

            if is_sequence:
                B, S, C, D, H, W = inputs.shape
                H_out, W_out = interferos.shape[-2], interferos.shape[-1]
                inputs_flat = inputs.view(B * S, C, D, H, W)
                interferos_flat = interferos.view(B * S, 1, H_out, W_out)
                mask_flat = mask.view(B * S, H_out, W_out)
            else:
                B, C, D, H, W = inputs.shape
                S = 1
                H_out, W_out = interferos.shape[-2], interferos.shape[-1]
                inputs_flat = inputs
                interferos_flat = interferos
                mask_flat = mask

            recon_loss = torch.tensor(0.0, device=args.device)
            consis_loss = torch.tensor(0.0, device=args.device)
            tv_loss = torch.tensor(0.0, device=args.device)

            with torch.amp.autocast('cuda', enabled=(args.device != 'cpu')):
                preds_flat = model(inputs_flat)

                lambda_c = args.lambda_c
                lambda_t = args.lambda_t
                recon_weight = 1.0 - lambda_c - lambda_t

                recon_loss = loss_fn[0](preds_flat, interferos_flat)
                loss = recon_weight * recon_loss

                if lambda_c > 0.0 and is_sequence:
                    preds_reshaped = preds_flat.view(B, S, 1, H_out, W_out)
                    mask_reshaped = mask.view(B, S, H_out, W_out)
                    consis_loss = loss_fn[1](preds_reshaped, mask_reshaped)
                    loss = loss + (lambda_c * consis_loss)

                if lambda_t > 0.0:
                    tv_loss = loss_fn[2](preds_flat)
                    loss = loss + (lambda_t * tv_loss)

                sum_total += float(loss) * B
                sum_recon += float(recon_loss) * B
                sum_consis += float(consis_loss) * B
                sum_tv += float(tv_loss) * B
                n_samples += B

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        mean_train_loss = sum_total / n_samples
        mean_train_recon_loss = sum_recon / n_samples
        mean_train_consis_loss = sum_consis / n_samples
        mean_train_tv_loss = sum_tv / n_samples

        mem_GB = int(torch.cuda.max_memory_allocated() / (1024**3))
        lr = optimizer.param_groups[0]['lr']

        # Validation phase
        if (epoch + 1) % args.eval_epoch == 0:
            _, return_str = predict_model(
                'train', False, model, loss_fn, epoch, train_loader, MODEL_DIR
            )
            txt_file.write(f'{return_str} learning_rate:{lr:0.6f} max_Mem: {mem_GB:0.6f} GB')
            txt_file.flush()

            mean_val_loss, return_str = predict_model(
                'val', False, model, loss_fn, epoch, val_loader, MODEL_DIR
            )
            txt_file.write(f'{return_str}')
            txt_file.flush()

            patience, min_epoch = val_loss_obs.update(mean_val_loss, epoch, txt_file)

            # Save checkpoints
            dict_ = {
                'args': vars(args),
                'epoch': epoch,
                'loss': loss_fn,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_dict': lr_scheduler.state_dict()
            }

            model_to_save = model.module if isinstance(model, nn.DataParallel) else model

            if epoch - min_epoch == 0:
                save_model(model_to_save, dict_, MODEL_DIR, f"{args.model_feature}_mini.pt")
            elif patience == 0:
                save_model(model_to_save, dict_, MODEL_DIR, f"{args.model_feature}_zero-patience.pt")
            else:
                save_model(model_to_save, dict_, MODEL_DIR, f"{args.model_feature}_final.pt")

            if patience == args.val_patience:
                break

        else:
            txt_file.write('\n\ntrain')
            txt_file.write(
                f'Epoch:{epoch:d} '
                f'train_loss:{mean_train_loss:.4f}, '
                f'train_recon_loss:{mean_train_recon_loss:.4f}, '
                f'train_consis_loss:{mean_train_consis_loss:.4f}, '
                f'train_tv_loss:{mean_train_tv_loss:.4f}, '
                f'learning_rate:{lr:0.7f} '
                f'max_Mem:{mem_GB:0.6f} GB'
            )

        lr_scheduler.step()

    fin_timestamp = datetime.datetime.now()

    if verbose:
        print(f'\nTrain finish! {fin_timestamp}')

    txt_file.write(f'\nTrain finish! {fin_timestamp}\n')
    txt_file.flush()


# -----------------------------------------------------------------------------
def predict_model(dataset_type, check_time, model, loss_fn, epoch, data_loader, height_dir):

    return_str = f'\n\n{dataset_type}'
    log_loss = not check_time
    bump_records = []

    model.eval()

    if log_loss:
        sum_total_val, sum_recon_val, sum_consis_val, sum_tv_val = 0.0, 0.0, 0.0, 0.0
        n_samples_val = 0

    speed_result = torch.zeros(6, device=args.device) if check_time else None
    mean_loss_val = 0

    with torch.no_grad():
        if 'test' in dataset_type:
            height_map_gt = os.path.join(height_dir, "map", "gt")
            height_map_pred = os.path.join(height_dir, "map", "pred")
            feature_map_dir = os.path.join(height_dir, "feature_maps_npy")
            viz_dir = os.path.join(height_dir, "result_img")

            os.makedirs(height_map_gt, exist_ok=True)
            os.makedirs(height_map_pred, exist_ok=True)
            os.makedirs(feature_map_dir, exist_ok=True)
            os.makedirs(viz_dir, exist_ok=True)

        for i, data in enumerate(data_loader):

            if len(data) == 5:
                inputs, interferos, mask, heights_list, file_names = data
            else:
                inputs, interferos, mask, file_names = data

            if check_time:
                t1 = time_sync()

            inputs = inputs.to(args.device, non_blocking=True).float()
            interferos = interferos.to(args.device, non_blocking=True).float()
            mask = mask.to(args.device, non_blocking=True)

            if check_time:
                speed_result[1] += time_sync() - t1

            is_sequence = (inputs.dim() == 6)

            if is_sequence:
                B, S, C, D, H, W = inputs.shape
                H_out, W_out = interferos.shape[-2], interferos.shape[-1]
                inputs_flat = inputs.view(B * S, C, D, H, W)
                interferos_flat = interferos.view(B * S, 1, H_out, W_out)
                mask_flat = mask.view(B * S, H_out, W_out)
            else:
                B = inputs.shape[0]
                S = 1
                inputs_flat = inputs
                interferos_flat = interferos
                mask_flat = mask

            if check_time:
                t2 = time_sync()

            outputs = model(inputs_flat)
            preds_flat, x_cnn, x_argmax, x_softmax = outputs

            if check_time:
                speed_result[2] += time_sync() - t2
                speed_result[0] += B

            if is_sequence:
                preds = preds_flat.view(B, S, 1, H_out, W_out)
            else:
                preds = preds_flat

            if 'test' in dataset_type:
                save_feature_map_npy(x_cnn, file_names[0], feature_map_dir, suffix="x_cnn")
                save_feature_map_npy(x_argmax, file_names[0], feature_map_dir, suffix="x_argmax")
                save_feature_map_npy(x_softmax, file_names[0], feature_map_dir, suffix="x_softmax")

                for b in range(B):
                    base_name = file_names[b]

                    gt_map_b = interferos[b, 0].detach().cpu().numpy() if not is_sequence else interferos[b, 0, 0].detach().cpu().numpy()
                    pred_map_b = preds[b, 0].detach().cpu().numpy() if not is_sequence else preds[b, 0, 0].detach().cpu().numpy()
                    mask_b = mask[b].detach().cpu().numpy() if not is_sequence else mask[b, 0].detach().cpu().numpy()

                    records, gt_h_list, pred_h_list = extract_bump_heights(
                        base_name, pred_map_b, mask_b, heights_list[b]
                    )

                    bump_records.extend(records)
                    save_bump_viz_img(viz_dir, base_name, gt_map_b, pred_map_b, gt_h_list, pred_h_list, mask_b)

                    pred_csv_path = os.path.join(height_map_pred, f"{base_name}.csv")
                    np.savetxt(pred_csv_path, pred_map_b, delimiter=",", fmt="%.6f")

                    gt_csv_path = os.path.join(height_map_gt, f"{base_name}.csv")
                    np.savetxt(gt_csv_path, gt_map_b, delimiter=",", fmt="%.6f")

            if log_loss:
                lambda_c = args.lambda_c
                lambda_t = args.lambda_t
                recon_weight = 1.0 - lambda_c - lambda_t

                recon_loss = loss_fn[0](preds_flat, interferos_flat)
                loss = recon_weight * recon_loss

                consis_loss = torch.tensor(0.0, device=preds_flat.device)
                tv_loss = torch.tensor(0.0, device=preds_flat.device)

                if lambda_c > 0.0 and is_sequence:
                    mask_reshaped = mask.view(B, S, mask.shape[-2], mask.shape[-1])
                    consis_loss = loss_fn[1](preds, mask_reshaped)
                    loss = loss + (lambda_c * consis_loss)

                if lambda_t > 0.0:
                    tv_loss = loss_fn[2](preds_flat)
                    loss = loss + (lambda_t * tv_loss)

                sum_total_val += float(loss) * B
                sum_recon_val += float(recon_loss) * B
                sum_consis_val += float(consis_loss) * B
                sum_tv_val += float(tv_loss) * B
                n_samples_val += B

        mem_GB = int(torch.cuda.max_memory_allocated() / (1024**3))

        if check_time:
            speed_result[1:] *= 1000
            detection_time_per_image = (speed_result[1] + speed_result[2] + speed_result[3]) / speed_result[0]

            return_str += f'\n\nMax GPU memory for batch size {args.batch_size_test}: {mem_GB:.6f} GB'
            return_str += (
                f'\nTrained Epoch:{epoch:d}, '
                f'Data num:{int(speed_result[0])}, '
                f'pre-processing time:{speed_result[1]:0.8f} msec, '
                f'inference time:{speed_result[2]:0.8f} msec, '
                f'post-processing time:{speed_result[3]:0.8f} msec'
            )

        if log_loss and n_samples_val > 0:
            mean_loss_val = sum_total_val / n_samples_val
            mean_recon_loss_val = sum_recon_val / n_samples_val
            mean_consis_loss_val = sum_consis_val / n_samples_val
            mean_tv_loss_val = sum_tv_val / n_samples_val

            return_str += (
                f'\nEpoch:{epoch:d} '
                f'{dataset_type}_loss:{mean_loss_val:.4f}, '
                f'{dataset_type}_recon_loss:{mean_recon_loss_val:.4f}, '
                f'{dataset_type}_consis_loss:{mean_consis_loss_val:.4f}, '
                f'{dataset_type}_tv_loss:{mean_tv_loss_val:.4f}'
            )

        if len(bump_records) > 0:
            df = pd.DataFrame(bump_records)
            csv_save_path = os.path.join(height_dir, "bump_stats.csv")
            df.to_csv(csv_save_path, index=False, float_format='%.4f')

    return mean_loss_val, return_str


# -----------------------------------------------------------------------------
class Score_Observer:
    def __init__(self, name, verbose=True):
        self.name = name
        self.min_epoch = 0
        self.last = 0
        self.min = 10
        self.patience = 0
        self.verbose = verbose

    def update(self, score, epoch, txt_file):
        if score < self.min:
            self.min = score
            self.min_epoch = epoch

        if epoch == 0 or score < self.last:
            self.patience = 0
        else:
            self.patience += 1

            if self.verbose:
                print(f'patience: {self.patience}')

        txt_file.write(f'\n patience: {self.patience}')
        txt_file.flush()

        self.last = score

        return self.patience, self.min_epoch