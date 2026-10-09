#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import time
import torch
import torch.nn as nn

try:
    import thop  # type: ignore # for FLOPs computation
except ImportError:
    thop = None

# ==============================================================================
# 1. Model Checkpoint & I/O Functions
# ==============================================================================

def save_model(model, dict_, MODEL_DIR, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    save_dict = {'model': model.state_dict()}
    save_dict.update(dict_)
    torch.save(save_dict, os.path.join(MODEL_DIR, filename))


def load_checkpoint(model_dir, model_mode):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
    model_files_ = os.listdir(f'{model_dir}/')
    model_path = None
    
    for model_file_ in model_files_:
        if f'{model_mode}.pt' in model_file_:
            model_path = os.path.join(model_dir, model_file_)
            break  
            
    if model_path is None:
        raise FileNotFoundError(f"No checkpoint found for mode '{model_mode}' in {model_dir}")
        
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    return checkpoint, model_path


# ==============================================================================
# 2. PyTorch Hardware & Training Utilities
# ==============================================================================

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass 
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True