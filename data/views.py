"""LeJEPA views from a batch of RGB images in [0, 1].

* classic (official LeJEPA): ``n_global`` global + ``n_local`` local views, each with its own
  random resized crop, horizontal flip and colour augmentation.
* dense (METER recipe): ``n_global // 2`` global crops and ``n_local // 2`` local crops, each
  seen twice with different colour augmentation only, so that the two views of a crop are
  spatially aligned token by token.

Views are returned as lists of tensors (B, 3, S, S), globals first, in the order the loss
expects: for dense, views ``2k`` and ``2k + 1`` share the same crop.
"""
from __future__ import annotations

import torch
from torchvision.transforms import v2


def colour_transform(global_view: bool) -> v2.Compose:
    return v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=23 if global_view else 9)], p=0.5),
    ])


def _independent_views(images: torch.Tensor, size: int, scale: tuple[float, float], n: int,
                       global_view: bool) -> list[torch.Tensor]:
    spatial = v2.Compose([v2.RandomResizedCrop(size, scale=scale, antialias=True), v2.RandomHorizontalFlip()])
    colour = colour_transform(global_view)
    return [torch.stack([colour(spatial(im)) for im in images]) for _ in range(n)]


def _paired_views(images: torch.Tensor, size: int, scale: tuple[float, float], n: int,
                  global_view: bool) -> list[torch.Tensor]:
    if n % 2:
        raise ValueError("dense views come in pairs: n must be even")
    crop = v2.RandomResizedCrop(size, scale=scale, antialias=True)
    colour = colour_transform(global_view)
    views = []
    for _ in range(n // 2):
        base = torch.stack([crop(im) for im in images])
        views += [torch.stack([colour(im) for im in base]) for _ in range(2)]
    return views


def make_views(images: torch.Tensor, recipe) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return ``(global_views, local_views)`` for a ``training.lejepa.Recipe``."""
    make = _independent_views if recipe.variant == "classic" else _paired_views
    g = make(images, recipe.global_size, tuple(recipe.global_scale), recipe.n_global, True)
    l = make(images, recipe.local_size, tuple(recipe.local_scale), recipe.n_local, False)
    return g, l
