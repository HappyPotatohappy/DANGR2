import torch
import numpy as np
import random
import os

def set_random_seed(seed=42):
    """랜덤 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, filename):
    """체크포인트 저장"""
    torch.save(state, filename)
    
def load_checkpoint(filename, model, optimizer=None):
    """체크포인트 로드"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_accuracy', 0)

def create_dir(path):
    """디렉토리 생성"""
    if not os.path.exists(path):
        os.makedirs(path)
