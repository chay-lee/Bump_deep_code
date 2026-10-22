
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.torch_utils import save_model, time_sync

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

        sum_losses = torch.zeros(4, device=args.device, dtype=torch.float64)
        n_samples = 0

        for i, data in enumerate(train_loader):
            inputs, interferos, file_name = data

            inputs = inputs.to(args.device, non_blocking=True).float()
            interferos = interferos.to(args.device, non_blocking=True).float()

            is_sequence = (inputs.dim() == 6)

            if is_sequence:
                B, S, C, D, H, W = inputs.shape
                H_out, W_out = interferos.shape[-2], interferos.shape[-1]
                inputs_flat = inputs.view(B * S, C, D, H, W)
                interferos_flat = interferos.view(B * S, 1, H_out, W_out)
            else:
                B, C, D, H, W = inputs.shape
                S = 1
                H_out, W_out = interferos.shape[-2], interferos.shape[-1]
                inputs_flat = inputs
                interferos_flat = interferos

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
                    consis_loss = loss_fn[1](preds_reshaped)
                    loss = loss + (lambda_c * consis_loss)

                if lambda_t > 0.0:
                    tv_loss = loss_fn[2](preds_flat)
                    loss = loss + (lambda_t * tv_loss)

                sum_losses += torch.stack([
                    loss.detach(),
                    recon_loss.detach(),
                    consis_loss.detach(),
                    tv_loss.detach()
                ]).to(torch.float64) * B

                n_samples += B

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        mean_train_loss, mean_train_recon_loss, mean_train_consis_loss, mean_train_tv_loss = (
            (sum_losses / n_samples).tolist()
        )

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
def predict_model(dataset_type, check_time, model, loss_fn, epoch, data_loader, height_dir,
                  pixel_evaluator=None, bump_evaluator=None, scan_id=1):
    is_test, log_loss = 'test' in dataset_type, not check_time
    model.eval(); n_samples = 0; mean_loss = 0.0; times = np.zeros(3)
    sum_losses = torch.zeros(4, device=args.device, dtype=torch.float64) if log_loss else None

    if is_test:
        pred_dir, gt_dir = os.path.join(height_dir, 'map', 'pred'), os.path.join(height_dir, 'map', 'gt')
        os.makedirs(pred_dir, exist_ok=True); os.makedirs(gt_dir, exist_ok=True)

    with torch.no_grad():
        batches = tqdm(data_loader, desc=dataset_type, unit='batch', disable=not is_test)
        for inputs, interferos, file_names in batches:
            t0 = time_sync() if check_time else 0
            inputs = inputs.to(args.device, non_blocking=True).float()
            interferos = interferos.to(args.device, non_blocking=True).float()
            if check_time: times[0] += time_sync()-t0

            sequence = inputs.dim() == 6
            B, S = inputs.shape[0], inputs.shape[1] if sequence else 1
            H, W = interferos.shape[-2:]
            x = inputs.flatten(0, 1) if sequence else inputs
            gt_flat = interferos.flatten(0, 1) if sequence else interferos

            t0 = time_sync() if check_time else 0
            pred_flat = model(x)[0]
            if check_time: times[1] += time_sync()-t0
            preds = pred_flat.reshape(B, S, 1, H, W) if sequence else pred_flat
            n_samples += B

            if is_test:
                if sequence: raise ValueError('Test loader must return one scan per sample.')
                t0 = time_sync() if check_time else 0
                pred_maps, gt_maps = preds[:, 0].cpu().numpy(), interferos[:, 0].cpu().numpy()

                for stem, pred, gt in zip(file_names, pred_maps, gt_maps):
                    if pixel_evaluator is not None: pixel_evaluator.update(scan_id, stem, pred, gt)
                    np.savetxt(os.path.join(pred_dir, f'{stem}.csv'), pred, delimiter=',', fmt='%.6f')
                    if np.isfinite(gt).all():
                        np.savetxt(os.path.join(gt_dir, f'{stem}.csv'), gt, delimiter=',', fmt='%.6f')

                if bump_evaluator is not None: bump_evaluator.update_batch(scan_id, file_names, pred_maps)
                if check_time: times[2] += time_sync()-t0

            if log_loss:
                recon = loss_fn[0](pred_flat, gt_flat)
                consis, tv = pred_flat.new_zeros(()), pred_flat.new_zeros(())
                if args.lambda_c > 0 and sequence: consis = loss_fn[1](preds)
                if args.lambda_t > 0: tv = loss_fn[2](pred_flat)
                loss = (1-args.lambda_c-args.lambda_t)*recon + args.lambda_c*consis + args.lambda_t*tv
                sum_losses += torch.stack([loss, recon, consis, tv]).to(torch.float64)*B

    text = f'\n\n{dataset_type}'
    if log_loss and n_samples:
        mean_loss, recon, consis, tv = (sum_losses/n_samples).tolist()
        text += (f'\nEpoch:{epoch}, {dataset_type}_loss:{mean_loss:.4f}, '
                 f'{dataset_type}_recon_loss:{recon:.4f}, {dataset_type}_consis_loss:{consis:.4f}, '
                 f'{dataset_type}_tv_loss:{tv:.4f}')

    if check_time:
        mem = torch.cuda.max_memory_allocated()/(1024**3) if args.device == 'cuda' else 0.0
        text += (f'\nEpoch:{epoch}, patches:{n_samples}, peak_GPU_GiB:{mem:.3f}\n'
                 f'Total transfer:{times[0]:.3f}s, forward:{times[1]:.3f}s, saving/evaluation:{times[2]:.3f}s\n'
                 'Diagnostic timing only; excludes loader I/O and final aggregation; no warm-up exclusion.')
    return mean_loss, text


# -----------------------------------------------------------------------------

class Score_Observer:
    def __init__(self, name, verbose=True):
        self.name = name
        self.min_epoch = -1
        self.min = float('inf')
        self.patience = 0
        self.verbose = verbose

    def update(self, score, epoch, txt_file):

        if score < self.min:
            self.min = score
            self.min_epoch = epoch
            self.patience = 0

        else:
            self.patience += 1

        if self.verbose:
            print(f'patience: {self.patience}')

        txt_file.write(f'\n patience: {self.patience}')
        txt_file.flush()

        return self.patience, self.min_epoch
