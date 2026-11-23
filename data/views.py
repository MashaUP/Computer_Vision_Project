"""LeJEPA views from a batch of RGB images in [0, 1] (B, 3, H, W).

* **classic**: ``n_global`` global + ``n_local`` local views, each with its own random resized
  crop, horizontal flip and colour augmentation.
* **dense** (METER recipe): ``n_global // 2`` global and ``n_local // 2`` local crops, each seen
  twice with different colour augmentation only, so the two views of a crop are aligned token by
  token. Views ``2k`` and ``2k + 1`` share the crop.

``shared_batch_aug: true`` applies each transform to the whole batch tensor at once, so every
image of the batch gets the same crop and colour parameters. That is what the phase-1 dense code
did (torchvision v2 samples one set of parameters per call); it is kept only to reproduce phase 1.

Normalisation is not applied here: every model normalises its own input in ``encode``. The order
of the random draws matches the phase-1 code, so with the same seed the views are identical.
"""
from __future__ import annotations

import torch
from torchvision.transforms import v2


def spatial_transform(size: int, scale) -> v2.Compose:
    return v2.Compose([v2.RandomResizedCrop(size, scale=tuple(scale), antialias=True), v2.RandomHorizontalFlip(p=0.5)])


def colour_transform(global_view: bool, solarize_p: float = 0.0) -> v2.Compose:
    ops = [
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=23 if global_view else 9)], p=0.5),
    ]
    if solarize_p > 0:
        ops.append(v2.RandomSolarize(threshold=128.0 / 255.0, p=solarize_p))
    return v2.Compose(ops)


def _apply(t, images: torch.Tensor, shared: bool) -> torch.Tensor:
    return t(images) if shared else torch.stack([t(im) for im in images])


def _independent(images, size, scale, n, global_view, recipe):
    spatial, colour = spatial_transform(size, scale), colour_transform(global_view, recipe.solarize_p)
    both = v2.Compose([spatial, colour])
    return [_apply(both, images, recipe.shared_batch_aug) for _ in range(n)]


def _paired(images, size, scale, n, global_view, recipe):
    if n % 2:
        raise ValueError("dense views come in pairs: n must be even")
    spatial, colour = spatial_transform(size, scale), colour_transform(global_view, recipe.solarize_p)
    views = []
    for _ in range(n // 2):
        crop = _apply(spatial, images, recipe.shared_batch_aug)
        views += [_apply(colour, crop, recipe.shared_batch_aug) for _ in range(2)]
    return views


def make_views(images: torch.Tensor, recipe) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return ``(global_views, local_views)`` for a ``training.lejepa.Recipe``."""
    make = _independent if recipe.variant == "classic" else _paired
    g = make(images, recipe.global_size, recipe.global_scale, recipe.n_global, True, recipe)
    l = make(images, recipe.local_size, recipe.local_scale, recipe.n_local, False, recipe)
    return g, l
