"""Loss functions for training.

Proposal ref: §4.2.8 — "The training objective will combine binary
cross-entropy loss and Dice loss."

BCEDiceLoss operates on raw logits (matches unet.py / attention_unet.py,
both of which return logits, no sigmoid applied). Internally it applies
sigmoid once to get probabilities for the Dice term, and uses the
numerically-stable BCEWithLogitsLoss (rather than sigmoid + BCELoss) for
the BCE term.
"""

from __future__ import annotations

import torch
from torch import nn


def dice_loss(pred_probs: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """
    1 - Dice coefficient, computed per-sample then averaged over the batch
    (rather than flattening the whole batch into one giant Dice), so a
    single large or unusually easy/hard image in a batch doesn't dominate.

    Args:
        pred_probs: (B, 1, H, W) probabilities in [0, 1] (post-sigmoid).
        target:     (B, 1, H, W) binary ground truth, values in {0, 1}.
        eps: smoothing term added to BOTH numerator and denominator. This
            matters for degenerate cases: if a sample has no lesion pixels
            at all (target all zero) and the model correctly predicts all
            zero, intersection=0 and union=0. Without eps in the numerator
            too, that would score as dice=0 (worst possible) instead of the
            correct dice=1 (perfect match).

            eps is scaled in PIXEL-COUNT terms, not as a tiny fractional
            constant -- intersection/union here are sums over raw pixel
            probabilities, not normalised to [0, 1]. sigmoid() never
            outputs exactly 0, so for an all-background image the residual
            "leakage" summed over every pixel can easily exceed a naive
            1e-6: at 256x256 (65,536 px) even a confidently negative
            prediction (logit=-20) sums to ~1.3e-4, well above 1e-6, which
            would wrongly penalise an essentially-perfect prediction.
            eps=1.0 (one pixel's worth of smoothing) is the standard
            convention in Dice loss implementations for exactly this
            reason and is safe across the image sizes used in this project
            (256 or 512).

    Returns:
        Scalar tensor, the batch-averaged Dice loss.
    """
    pred_flat = pred_probs.flatten(start_dim=1)   # (B, H*W)
    target_flat = target.flatten(start_dim=1)      # (B, H*W)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice_per_sample = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice_per_sample.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined loss: bce_weight * BCE + (1 - bce_weight) * Dice.

    Args:
        bce_weight: weighting between the two terms, in [0, 1]. Default 0.5
            (equal weighting) per proposal §4.2.8, which does not specify a
            different split.
        dice_eps: smoothing term passed to dice_loss (see its docstring --
            default 1.0, scaled in pixel-count terms, not a tiny fraction).
    """

    def __init__(self, bce_weight: float = 0.5, dice_eps: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= bce_weight <= 1.0:
            raise ValueError(f"bce_weight must be in [0, 1], got {bce_weight}")
        self.bce_weight = bce_weight
        self.dice_eps = dice_eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw model output, NOT passed through sigmoid.
            target: (B, 1, H, W) binary ground truth, values in {0, 1},
                same dtype as logits (float).
        """
        bce_loss = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        d_loss = dice_loss(probs, target, eps=self.dice_eps)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * d_loss