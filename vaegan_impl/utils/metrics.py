"""
Evaluation Metrics for VAEGAN
Reused from original VAEGAN implementation
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as linalg


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Gaussian RBF kernel.
    
    Args:
        source: Source samples (B, D)
        target: Target samples (B, D)
        kernel_mul: Multiplier for bandwidth
        kernel_num: Number of bandwidths
        fix_sigma: Fixed sigma or None (compute from data)
    
    Returns:
        MMD kernel matrix
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    
    L2_distance = ((total0 - total1) ** 2).sum(2)
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
    return sum(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Maximum Mean Discrepancy (MMD) using Gaussian RBF kernel.
    
    Args:
        source: Source samples (B, D)
        target: Target samples (B, D)
        kernel_mul: Multiplier for bandwidth
        kernel_num: Number of bandwidths
        fix_sigma: Fixed sigma or compute from data
    
    Returns:
        MMD score (scalar)
    """
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def SSD(y, y_pred):
    """
    Sum of Squared Differences (SSD).
    
    Args:
        y: True values (B, D)
        y_pred: Predicted values (B, D)
    
    Returns:
        SSD score per sample (B,)
    """
    return torch.sum((y - y_pred) ** 2, dim=1)


def SSD_mean(y, y_pred):
    """
    Mean SSD across samples.
    
    Args:
        y: True values (B, D)
        y_pred: Predicted values (B, D)
    
    Returns:
        Mean SSD across batch
    """
    return torch.mean(SSD(y, y_pred))


def PRD(y, y_pred):
    """
    Percent Relative Deviation (PRD).
    
    Args:
        y: True values (B, D)
        y_pred: Predicted values (B, D)
    
    Returns:
        PRD score (B,)
    """
    N = torch.sum((y_pred - y) ** 2, dim=1)
    D = torch.sum((y_pred - torch.mean(y, dim=0)) ** 2, dim=1)
    
    PRD = torch.sqrt(N / D) * 100
    return PRD


def PRD_mean(y, y_pred):
    """
    Mean PRD across samples.
    
    Args:
        y: True values (B, D)
        y_pred: Predicted values (B, D)
    
    Returns:
        Mean PRD across batch
    """
    return torch.mean(PRD(y, y_pred))


def COSS(y, y_pred):
    """
    Cosine Similarity (COSS).
    
    Args:
        y: True values (B, D)
        y_pred: Predicted values (B, D)
    
    Returns:
        Cosine similarity per sample (B,)
    """
    return torch.nn.functional.cosine_similarity(y, y_pred)


def COSS_mean(y, y_pred):
    """
    Mean Cosine Similarity across samples.
    
    Args:
        y: True values (B, D)
        y_pred: Predicted values (B, D)
    
    Returns:
        Mean Cosine similarity across batch
    """
    return torch.mean(COSS(y, y_pred))


def compute_all_metrics(real_data, generated_data):
    """
    Compute all evaluation metrics.
    
    Args:
        real_Real samples (B, D)
        generated_Generated samples (B, D)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # MMD
    metrics['MMD'] = mmd_rbf(real_data, generated_data).item()
    
    # SSD
    metrics['SSD'] = SSD_mean(real_data, generated_data).item()
    
    # PRD
    metrics['PRD'] = PRD_mean(real_data, generated_data).item()
    
    # COSS (higher is better, so invert)
    metrics['COSS_mean'] = COSS_mean(real_data, generated_data).item()
    metrics['COSS_neg'] = -metrics['COSS_mean']  # Lower is better
    
    return metrics


def save_metrics_to_csv(metrics_list, filename):
    """
    Save metrics to CSV file.
    
    Args:
        metrics_list: List of metric dictionaries
        filename: Output CSV file path
    """
    import pandas as pd
    df = pd.DataFrame(metrics_list)
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")