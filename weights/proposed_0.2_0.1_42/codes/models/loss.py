import torch
import torch.nn as nn

def get_loss_fn():
    recon_loss_fn = masked_l1_loss
    consis_loss_fn = ConsistencyLoss()
    tv_loss_fn = TotalVariationLoss() 
    return recon_loss_fn, consis_loss_fn, tv_loss_fn

def masked_l1_loss(pred, gt):
    mask = (gt != 0).float()
    loss = torch.abs(pred - gt)
    masked_loss = loss * mask
    if mask.sum() > 0:
        return masked_loss.sum() / mask.sum()
    else:
        return torch.tensor(0.0, device=pred.device)

class ConsistencyLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds, masks=None):
        if preds.size(1) < 2:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
            
        preds_f32 = preds.to(torch.float32)
        mean_preds = preds_f32.mean(dim=1, keepdim=True)
        pixel_variance = ((preds_f32 - mean_preds) ** 2).mean(dim=1)
        pixel_std = torch.sqrt(pixel_variance + self.eps)
        
        loss = pixel_std.mean()
        return loss
    
class TotalVariationLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds): 
        if preds.dim() == 3:
            preds = preds.unsqueeze(1)

        # dx
        pred_dx = preds[:, :, :, 1:] - preds[:, :, :, :-1]
        loss_x = torch.sqrt(pred_dx**2 + self.eps)
        
        # dy
        pred_dy = preds[:, :, 1:, :] - preds[:, :, :-1, :]
        loss_y = torch.sqrt(pred_dy**2 + self.eps)

        total_smooth_loss = loss_x.mean() + loss_y.mean()
        
        return total_smooth_loss