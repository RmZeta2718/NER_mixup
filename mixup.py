from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

Tsr = torch.Tensor

def mixup_data(x: Tsr, y: Tsr, alpha:float=1.0, device: str='cuda') -> Tuple[Tsr, Tsr, Tsr, float]:
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.CrossEntropyLoss, pred: Tsr, y_a: Tsr, y_b: Tsr, lam: float):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
