"""Depth metrics for the comparison protocol, with the same crops, caps and alignment for every model.

* Crops: Eigen crop on NYU (rows 45:471, cols 41:601 of 480x640, scaled to the GT size) and
  Garg crop on KITTI; depth caps 10 m (NYU) and 80 m (KITTI), minimum 1e-3 m.
* Alignment, applied per image before the metrics:
  - ``none``: metric prediction as is;
  - ``median``: depth scaled by median(gt) / median(pred) (Monodepth2 convention);
  - ``scale_shift``: least-squares scale and shift fitted on *inverse* depth (MiDaS convention),
    for affine-invariant models, then converted back to depth.
* Metrics: AbsRel, SqRel, RMSE, RMSE-log, delta < 1.25 / 1.25^2 / 1.25^3.

Inputs are depth maps in metres with the same spatial size as the ground truth.
"""
from __future__ import annotations

import numpy as np
import torch

CAPS = {"nyu": 10.0, "kitti": 80.0}
MIN_DEPTH = 1e-3
METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")


def eval_mask(gt: torch.Tensor, dataset: str) -> torch.Tensor:
    """Valid-pixel mask (H, W): depth range + Eigen (NYU) or Garg (KITTI) crop."""
    h, w = gt.shape[-2:]
    valid = (gt > MIN_DEPTH) & (gt < CAPS[dataset])
    crop = torch.zeros_like(valid)
    if dataset == "nyu":
        crop[int(45 / 480 * h):int(471 / 480 * h), int(41 / 640 * w):int(601 / 640 * w)] = True
    elif dataset == "kitti":
        crop[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)] = True
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return valid & crop


def align(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Align a predicted depth map to the ground truth on the valid pixels."""
    if mode == "none":
        return pred
    if mode == "median":
        return pred * (gt[mask].median() / pred[mask].median())
    if mode == "scale_shift":
        inv_p, inv_g = 1.0 / pred[mask].clamp(min=MIN_DEPTH), 1.0 / gt[mask]
        a = torch.stack([inv_p, torch.ones_like(inv_p)], dim=1).double()
        sol = torch.linalg.lstsq(a, inv_g.double().unsqueeze(1)).solution.squeeze(1)
        s, t = sol[0], sol[1]
        inv = (s * (1.0 / pred.clamp(min=MIN_DEPTH)).double() + t).float()
        return 1.0 / inv.clamp(min=1.0 / 1e4)
    raise ValueError(f"unknown alignment {mode!r}")


def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, dataset: str, alignment: str = "median") -> dict[str, float]:
    """Metrics for one image. ``pred`` and ``gt``: (H, W) depth in metres."""
    pred, gt = pred.float().squeeze(), gt.float().squeeze()
    mask = eval_mask(gt, dataset)
    if mask.sum() == 0:
        raise ValueError("no valid ground-truth pixel")
    p = align(pred, gt, mask, alignment)[mask].clamp(MIN_DEPTH, CAPS[dataset])
    g = gt[mask]
    ratio = torch.maximum(p / g, g / p)
    return {
        "abs_rel": torch.mean(torch.abs(p - g) / g).item(),
        "sq_rel": torch.mean((p - g) ** 2 / g).item(),
        "rmse": torch.sqrt(torch.mean((p - g) ** 2)).item(),
        "rmse_log": torch.sqrt(torch.mean((torch.log(p) - torch.log(g)) ** 2)).item(),
        "a1": (ratio < 1.25).float().mean().item(),
        "a2": (ratio < 1.25 ** 2).float().mean().item(),
        "a3": (ratio < 1.25 ** 3).float().mean().item(),
    }


def mean_metrics(per_image: list[dict[str, float]]) -> dict[str, float]:
    """Average over the test set (each image weighs the same, as in the standard benchmarks)."""
    return {k: float(np.mean([m[k] for m in per_image])) for k in METRICS}
