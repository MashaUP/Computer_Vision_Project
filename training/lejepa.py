"""LeJEPA pre-training objective for any encoder, in two variants selected by the recipe.

Both variants act on the deepest encoder map (stride 32) and use SIGReg from the official
``lejepa`` library (galilai-group/lejepa @ c293d29):

* **classic** (official recipe): pooled embedding, MLP projector with BatchNorm, invariance to
  the centre of the global views, SIGReg on every view; ``loss = lambda * SIGReg + (1 - lambda) * inv``.
* **dense** (METER recipe): 4x4 token grid, per-token projector with LayerNorm; invariance is
  ``alpha * dense_inv + (1 - alpha) * global_inv`` where ``dense_inv`` aligns the two colour views
  of the same crop token by token and ``global_inv`` aligns the spatially averaged tokens of all
  views to their centre; SIGReg runs over all tokens.

Recipes live in ``configs/lejepa/{classic,dense}.yaml``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import yaml
from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley

from data.views import make_views
from models.lejepa_heads import ClassicProjector, DenseProjector

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "lejepa"


@dataclass
class Recipe:
    variant: str                      # "classic" | "dense"
    lambd: float                      # SIGReg weight
    n_global: int
    global_size: int
    global_scale: list[float]
    n_local: int
    local_size: int
    local_scale: list[float]
    proj_dims: list[int]
    num_slices: int
    epps_points: int
    epps_t_max: float | None = None   # None = library default
    alpha: float = 0.5                # dense only: dense vs. global invariance
    grid: int = 4                     # dense only: token grid side
    lr: float = 5e-4
    weight_decay: float = 5e-4
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.variant not in ("classic", "dense"):
            raise ValueError(f"variant must be classic or dense, got {self.variant!r}")


def load_recipe(name_or_path: str, **overrides) -> Recipe:
    """Load ``configs/lejepa/<name>.yaml`` (or a path) and apply keyword overrides."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg.update(overrides)
    known = {f.name for f in fields(Recipe)}
    extra = {k: cfg.pop(k) for k in list(cfg) if k not in known}
    return Recipe(**cfg, extra=extra)


class LeJEPALoss(nn.Module):
    """Projector + SIGReg. ``forward(deep_global, deep_local)`` takes the deepest maps of the
    global and local views (lists of (B, C, h, w)) and returns ``{"loss", "inv", "sigreg"}``."""

    def __init__(self, in_channels: int, recipe: Recipe):
        super().__init__()
        self.recipe = recipe
        epps = {"n_points": recipe.epps_points}
        if recipe.epps_t_max is not None:
            epps["t_max"] = recipe.epps_t_max
        if recipe.variant == "classic":
            self.projector = ClassicProjector(in_channels, tuple(recipe.proj_dims))
            self.sigreg = SlicingUnivariateTest(univariate_test=EppsPulley(**epps), num_slices=recipe.num_slices)
        else:
            self.projector = DenseProjector(in_channels, tuple(recipe.proj_dims), grid=recipe.grid)
            self.sigreg = SlicingUnivariateTest(univariate_test=EppsPulley(**epps), num_slices=recipe.num_slices,
                                                reduction="mean")

    def _project(self, maps: list[torch.Tensor]) -> torch.Tensor:
        # views of the same size go through the projector together: (V * B, ...) -> (V, B, ...)
        z = self.projector(torch.cat(maps))
        return z.reshape(len(maps), -1, *z.shape[1:])

    def forward(self, deep_global: list[torch.Tensor], deep_local: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        r = self.recipe
        zg, zl = self._project(deep_global), self._project(deep_local)
        z = torch.cat([zg, zl])                                     # classic (V, B, D) | dense (V, B, N, D)
        if r.variant == "classic":
            centre = zg.mean(0, keepdim=True)                       # centre of the global views
            inv = (centre - z).square().mean()
            sig = torch.stack([self.sigreg(z[v]) for v in range(z.shape[0])]).mean()
        else:
            pairs = z.reshape(z.shape[0] // 2, 2, *z.shape[1:])
            dense_inv = (pairs[:, 0] - pairs[:, 1]).square().mean()
            pooled = z.mean(2)                                      # (V, B, D)
            global_inv = (pooled.mean(0, keepdim=True) - pooled).square().mean()
            inv = r.alpha * dense_inv + (1 - r.alpha) * global_inv
            v, b, n, d = z.shape
            sig = self.sigreg(z.reshape(v, b * n, d))
        loss = r.lambd * sig + (1 - r.lambd) * inv
        return {"loss": loss, "inv": inv.detach(), "sigreg": sig.detach()}


def lejepa_step(encode: Callable[[torch.Tensor], list[torch.Tensor]], criterion: LeJEPALoss,
                images: torch.Tensor) -> dict[str, torch.Tensor]:
    """Make the views of ``images`` (B, 3, H, W in [0, 1]), encode them and compute the loss."""
    g, l = make_views(images, criterion.recipe)
    deep_g = [f for f in torch.split(encode(torch.cat(g))[-1], images.shape[0])]
    deep_l = [f for f in torch.split(encode(torch.cat(l))[-1], images.shape[0])]
    return criterion(deep_g, deep_l)


def recipe_dict(recipe: Recipe) -> dict:
    return asdict(recipe)
