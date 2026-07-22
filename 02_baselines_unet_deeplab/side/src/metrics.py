"""Losses and IoU metrics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target.clamp(min=0), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        denom = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes: int, class_weights=None, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        if self.ce.weight is not None and self.ce.weight.device != logits.device:
            self.ce.weight = self.ce.weight.to(logits.device)
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


@torch.no_grad()
def confusion_matrix(pred, target, num_classes: int) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)
    k = (target >= 0) & (target < num_classes)
    pred, target = pred[k], target[k]
    idx = num_classes * target + pred
    cm = torch.bincount(idx, minlength=num_classes**2)
    return cm.reshape(num_classes, num_classes).float()


def iou_from_cm(
    cm: torch.Tensor,
    exclude_background: bool = True,
    bg_index: int = 0,
):
    """mIoU defaults to foreground-only (excludes background)."""
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom.clamp(min=1e-6), torch.zeros_like(tp))
    present = denom > 0
    if exclude_background and cm.shape[0] > 1:
        present = present.clone()
        present[bg_index] = False
    miou = iou[present].mean().item() if present.any() else 0.0
    return iou, miou
