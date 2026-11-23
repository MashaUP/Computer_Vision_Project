"""LeJEPA pre-training objective for any encoder, in two variants selected by the recipe.

Both act on the deepest encoder map (stride 32) and use SIGReg from the official ``lejepa``
library (galilai-group/lejepa @ c293d29). ``loss = lambd * SIGReg + (1 - lambd) * invariance``.

* **classic**: global-average-pooled embedding and an MLP projector. Invariance pulls every view to
  the centre of the global views (``centre: global``, LeJEPA paper) or of all views
  (``centre: all``, phase-1 code and the official minimal example). SIGReg runs on each view and
  is averaged (``sigreg_input: per_view``) or on all views flattened together (``flat``, phase 1).
* **dense** (METER recipe): 4x4 token grid and a per-token projector. Invariance is
  ``(1 - alpha) * dense_inv + alpha * global_inv``, where ``dense_inv`` aligns the two colour views
  of the same crop token by token and ``global_inv`` aligns the token-averaged views to their
  centre; SIGReg runs on every view over all its tokens.

Recipes are in ``configs/lejepa/``. ``classic`` and ``dense`` are the protocol recipes;
``classic_phase1`` and ``dense_phase1`` reproduce the course-project code exactly
(``scripts/verify_phase1_equivalence.py``).
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
    variant: str                       # "classic" | "dense"
    lambd: float                       # SIGReg weight
    n_global: int                      # number of global views (dense: 2 per crop)
    global_size: int
    global_scale: list[float]
    n_local: int
    local_size: int
    local_scale: list[float]
    proj_dims: list[int]
    proj_norm: str                     # "batchnorm" | "layernorm"
    num_slices: int
    epps_points: int
    epps_t_max: float | None = None    # None = library default
    centre: str = "global"             # classic: "global" | "all"
    sigreg_input: str = "per_view"     # classic: "per_view" | "flat"
    clamp: float | None = None         # clamp projections to [-clamp, clamp] before the loss (phase 1: 100)
    alpha: float = 0.5                 # dense: weight of the global invariance term
    grid: int = 4                      # dense: token grid side
    solarize_p: float = 0.0
    shared_batch_aug: bool = False     # phase-1 dense: one set of augmentation parameters per batch
    encode_views: str = "by_size"      # "by_size": views of one size in one encoder batch; "per_view" (phase-1 classic)
    # optimisation (used by training/trainer.py)
    lr: float = 5e-4
    weight_decay: float = 5e-4
    epochs: int = 50
    warmup_epochs: int = 10
    eta_min_ratio: float = 1e-3        # cosine floor = lr * eta_min_ratio
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        checks = {"variant": ("classic", "dense"), "proj_norm": ("batchnorm", "layernorm"),
                  "centre": ("global", "all"), "sigreg_input": ("per_view", "flat"),
                  "encode_views": ("by_size", "per_view")}
        for k, allowed in checks.items():
            if getattr(self, k) not in allowed:
                raise ValueError(f"{k} must be one of {allowed}, got {getattr(self, k)!r}")


def load_recipe(name_or_path: str, dataset: str | None = None, **overrides) -> Recipe:
    """Load ``configs/lejepa/<name>.yaml`` (or a path); apply its ``by_dataset[dataset]`` block,
    then keyword overrides."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(path.read_text())
    by_dataset = cfg.pop("by_dataset", {}) or {}
    if dataset is not None:
        cfg.update(by_dataset.get(dataset, {}))
    cfg.update(overrides)
    known = {f.name for f in fields(Recipe)}
    extra = {k: cfg.pop(k) for k in list(cfg) if k not in known}
    return Recipe(**cfg, extra=extra)


class LeJEPALoss(nn.Module):
    """Projector + SIGReg. ``forward(deep_global, deep_local)`` takes the deepest maps of the global
    and local views (lists of (B, C, h, w)) and returns ``{"loss", "inv", "sigreg"}`` (plus the two
    invariance terms for dense)."""

    def __init__(self, in_channels: int, recipe: Recipe):
        super().__init__()
        self.recipe = recipe
        norm = nn.BatchNorm1d if recipe.proj_norm == "batchnorm" else nn.LayerNorm
        if recipe.variant == "classic":
            self.projector = ClassicProjector(in_channels, tuple(recipe.proj_dims), norm_layer=norm)
        else:
            self.projector = DenseProjector(in_channels, tuple(recipe.proj_dims), grid=recipe.grid, norm_layer=norm)
        epps = {"n_points": recipe.epps_points}
        if recipe.epps_t_max is not None:
            epps["t_max"] = recipe.epps_t_max
        kw = {"reduction": "mean"} if (recipe.variant == "dense" or recipe.sigreg_input == "flat") else {}
        self.sigreg = SlicingUnivariateTest(univariate_test=EppsPulley(**epps), num_slices=recipe.num_slices, **kw)

    def _project(self, maps: list[torch.Tensor]) -> torch.Tensor:
        z = self.projector(torch.cat(maps))                 # views of one size together
        return z.reshape(len(maps), -1, *z.shape[1:])        # (V, B, ...)

    def forward(self, deep_global: list[torch.Tensor], deep_local: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        r = self.recipe
        n_g = len(deep_global)
        z = torch.cat([self._project(deep_global), self._project(deep_local)])   # (V, B, D) | (V, B, N, D)
        z = z.float()
        if r.clamp is not None:
            z = torch.clamp(z, -r.clamp, r.clamp)
        out = {}
        if r.variant == "classic":
            centre = (z[:n_g] if r.centre == "global" else z).mean(0, keepdim=True)
            inv = (centre - z).square().mean()
            if r.sigreg_input == "flat":
                sig = self.sigreg(z.flatten(0, 1).contiguous())
            else:
                sig = torch.stack([self.sigreg(z[v]) for v in range(z.shape[0])]).mean()
        else:
            pairs = z.reshape(z.shape[0] // 2, 2, *z.shape[1:])
            dense_inv = (pairs[:, 0] - pairs[:, 1]).square().mean()
            pooled = z.mean(2)                                # (V, B, D)
            global_inv = (pooled.mean(0, keepdim=True) - pooled).square().mean()
            inv = (1 - r.alpha) * dense_inv + r.alpha * global_inv
            v, b, n, d = z.shape
            sig = self.sigreg(z.reshape(v, b * n, d).contiguous())
            out.update(dense_inv=dense_inv.detach(), global_inv=global_inv.detach())
        out.update(loss=(1 - r.lambd) * inv + r.lambd * sig, inv=inv.detach(), sigreg=sig.detach())
        return out


def lejepa_step(encode: Callable[[torch.Tensor], list[torch.Tensor]], criterion: LeJEPALoss,
                images: torch.Tensor) -> dict[str, torch.Tensor]:
    """Make the views of ``images`` (B, 3, H, W in [0, 1]), encode them and compute the loss.

    With ``encode_views: by_size`` the views of one size go through the encoder as one batch, so
    BatchNorm sees all of them together; with ``per_view`` each view is a separate forward pass
    (phase-1 classic)."""
    g, l = make_views(images, criterion.recipe)
    if criterion.recipe.encode_views == "per_view":
        return criterion([encode(v)[-1] for v in g], [encode(v)[-1] for v in l])
    b = images.shape[0]
    deep_g = list(torch.split(encode(torch.cat(g))[-1], b))
    deep_l = list(torch.split(encode(torch.cat(l))[-1], b))
    return criterion(deep_g, deep_l)


def recipe_dict(recipe: Recipe) -> dict:
    return asdict(recipe)
