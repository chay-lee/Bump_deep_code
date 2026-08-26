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
    """학습된 모델의 가중치와 추가 메타데이터(epoch, loss 등)를 저장합니다."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    save_dict = {'model': model.state_dict()}
    save_dict.update(dict_)
    torch.save(save_dict, os.path.join(MODEL_DIR, filename))


def load_checkpoint(model_dir, model_mode):
    """지정된 디렉토리에서 특정 모델 모드(variant)의 .pt 파일을 찾아 로드합니다."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
    model_files_ = os.listdir(f'{model_dir}/')
    model_path = None
    
    for model_file_ in model_files_:
        if f'{model_mode}.pt' in model_file_:
            model_path = os.path.join(model_dir, model_file_)
            break  # 매칭되는 파일을 찾으면 루프를 종료합니다.
            
    if model_path is None:
        raise FileNotFoundError(f"No checkpoint found for mode '{model_mode}' in {model_dir}")
        
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    return checkpoint, model_path


# ==============================================================================
# 2. PyTorch Hardware & Training Utilities
# ==============================================================================

def time_sync():
    """CUDA 디바이스의 모든 스트림 커널이 완료될 때까지 대기하여 정확한 시간을 측정합니다."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def initialize_weights(model):
    """네트워크 레이어들의 초기 가중치 및 하이퍼파라미터를 설정합니다."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # 필요한 경우 커스텀 Conv2d 초기화 로직을 여기에 추가 가능
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True