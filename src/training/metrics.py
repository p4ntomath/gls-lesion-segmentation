"""Pixel-wise segmentation metrics.

Proposal ref: §4.2.10 (eq. 3-6): Dice, IoU, precision, recall, computed from
TP/FP/FN/TN against binarised (threshold=0.5 by default) predictions.

All four metric functions share the same eps convention as
src/training/losses.py's dice_loss: eps is added to BOTH numerator and
denominator, so a sample with no true lesion pixels and no predicted lesion
pixels (tp=fp=fn=0) correctly scores 1.0 ("perfect", nothing to find and
nothing wrongly found) instead of 0.0 from 0/0-style degeneracy.
"""

from __future__ import annotations

import numpy as np
import torch

ArrayLike = np.ndarray  # after confusion_counts, everything downstream is numpy/python scalars


def _to_binary_numpy(x) -> np.ndarray:
    """Accept a torch.Tensor or numpy array (any int/float/bool dtype) and
    return a plain numpy bool array, detached and moved to CPU if needed."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).astype(bool)


def confusion_counts(pred_binary, target_binary) -> tuple[int, int, int, int]:
    """
    Args:
        pred_binary, target_binary: same-shape arrays/tensors of {0, 1} (or
            bool). Any shape is accepted -- typically (H, W) for one image
            or (B, 1, H, W) for a batch; counts are summed over all elements.

    Returns:
        (tp, fp, fn, tn) as plain Python ints.
    """
    pred = _to_binary_numpy(pred_binary)
    target = _to_binary_numpy(target_binary)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")

    tp = int(np.sum(pred & target))
    fp = int(np.sum(pred & ~target))
    fn = int(np.sum(~pred & target))
    tn = int(np.sum(~pred & ~target))
    return tp, fp, fn, tn


def dice_coefficient(tp: int, fp: int, fn: int, eps: float = 1e-6) -> float:
    """Eq. 3: Dice = 2*TP / (2*TP + FP + FN)."""
    return (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)


def iou_score(tp: int, fp: int, fn: int, eps: float = 1e-6) -> float:
    """Eq. 4: IoU = TP / (TP + FP + FN)."""
    return (tp + eps) / (tp + fp + fn + eps)


def precision_score(tp: int, fp: int, eps: float = 1e-6) -> float:
    """Eq. 5: Precision = TP / (TP + FP)."""
    return (tp + eps) / (tp + fp + eps)


def recall_score(tp: int, fn: int, eps: float = 1e-6) -> float:
    """Eq. 6: Recall = TP / (TP + FN)."""
    return (tp + eps) / (tp + fn + eps)


def binarize(probs, threshold: float = 0.5) -> np.ndarray:
    """Convenience: threshold a probability map (post-sigmoid, NOT raw
    logits) into a binary {0, 1} numpy array. threshold default matches
    proposal §4.2.7/§4.2.10 (0.5, or a value selected on the validation set
    and then held fixed)."""
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    return (np.asarray(probs) >= threshold).astype(np.uint8)


def segmentation_metrics(pred_binary, target_binary, eps: float = 1e-6) -> dict:
    """Convenience wrapper: run confusion_counts once, return all four
    metrics plus the raw counts in a single dict -- avoids recomputing
    confusion_counts four times when you want every metric for one sample
    or one batch."""
    tp, fp, fn, tn = confusion_counts(pred_binary, target_binary)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice_coefficient(tp, fp, fn, eps),
        "iou": iou_score(tp, fp, fn, eps),
        "precision": precision_score(tp, fp, eps),
        "recall": recall_score(tp, fn, eps),
    }