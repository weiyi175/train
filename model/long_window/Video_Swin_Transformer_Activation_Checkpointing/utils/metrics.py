#!/usr/bin/env python3
import torch, math
import numpy as np

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(-1)
    return (preds == targets).float().mean().item()

def precision_recall_f1(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Binary classification metrics (macro simplified for 2 classes)."""
    preds = logits.argmax(-1)
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1}

def binary_auc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute ROC AUC (binary) using pure torch/numpy (descending threshold sweep)."""
    if logits.ndim == 2 and logits.size(-1) == 2:
        probs = torch.softmax(logits, dim=-1)[:,1]
    else:
        probs = torch.sigmoid(logits.view(-1))
    y = targets.view(-1).float()
    # if single class present, AUC undefined
    if y.min() == y.max():
        return float('nan')
    probs_np = probs.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy().astype(np.float32)
    # sort by score desc
    order = np.argsort(-probs_np)
    y_np = y_np[order]
    probs_np = probs_np[order]
    # cumulative positives / negatives
    P = y_np.sum(); N = (1 - y_np).sum()
    tps = np.cumsum(y_np)
    fps = np.cumsum(1 - y_np)
    tpr = tps / (P + 1e-12)
    fpr = fps / (N + 1e-12)
    # prepend (0,0) and append (1,1)
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    auc = np.trapz(tpr, fpr)
    return float(auc)

def compute_all_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    acc = accuracy(logits, targets)
    prf = precision_recall_f1(logits, targets)
    auc = binary_auc(logits, targets)
    return {'acc': acc, 'precision': prf['precision'], 'recall': prf['recall'], 'f1': prf['f1'], 'auc': auc}

__all__ = ['accuracy','precision_recall_f1','binary_auc','compute_all_metrics']
