"""Supervised depth loss for fine-tuning (step 3, option B)."""
import torch


def silog_loss(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, variance_focus: float = 0.85) -> torch.Tensor:
    """Scale-invariant log loss (Eigen et al. 2014, as in BTS / AdaBins): the common step-3 loss
    with ground truth (option B). ``pred`` and ``gt`` are positive depths, ``valid`` a boolean mask."""
    if not valid.any():
        return pred.sum() * 0      # no valid pixel in the batch: zero loss, zero gradient (not NaN)
    d = torch.log(pred[valid]) - torch.log(gt[valid])
    return 10.0 * torch.sqrt((d ** 2).mean() - variance_focus * d.mean() ** 2 + 1e-8)
