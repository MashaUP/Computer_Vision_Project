"""Import the third-party baselines from their pinned submodules without touching their code.

Each baseline is imported with its repository first on ``sys.path``. Afterwards its top-level
modules are moved out of the global module table, so that:

* their generic names (``networks``, ``layers``, ``datasets``, ``model``, ``options``, ``scripts``...)
  never shadow ours or each other (FlexDepth and BoRe-Depth both ship a ``datasets`` package);
* our own packages (``utils``, ``data``, ``models``...) are restored if a baseline shadowed them.

Only the architectures are used: method-specific training heuristics are not imported.

Exception: ZipDepth's own loss (``ZipDepthLoss``) is imported for step 3, together with its
teacher, Depth Anything V2, used offline to generate the pseudo-depth targets
(``scripts/make_pseudo_depth.py``).

Pinned commits (see ``.gitmodules`` and ``docs/references.md``):
FlexDepth b04ae3a, ZipDepth 91f3fd2, BoRe-Depth db801e2, Depth Anything V2 a561b84.
"""
from __future__ import annotations

import contextlib
import os
import sys
import types
from pathlib import Path

import torch.nn as nn

SOTA_DIR = Path(__file__).resolve().parent
REPOS = {
    "flexdepth": SOTA_DIR / "flexdepth",
    "zipdepth": SOTA_DIR / "zipdepth",
    "bore": SOTA_DIR / "bore_depth",
    "depth_anything_v2": SOTA_DIR / "depth_anything_v2",
}

# Depth Anything V2 configurations, as in its run.py
DAV2_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class BaselineNotAvailable(RuntimeError):
    """The submodule is not checked out (run ``git submodule update --init``)."""


def _top_level_names(repo: Path) -> set[str]:
    names = set()
    for p in repo.iterdir():
        if p.suffix == ".py":
            names.add(p.stem)
        elif p.is_dir() and not p.name.startswith(".") and any(p.glob("*.py")):
            names.add(p.name)
    return names


@contextlib.contextmanager
def isolated_import(key: str, chdir: bool = False, stub_modules: dict[str, types.ModuleType] | None = None):
    """Context manager: import from ``REPOS[key]`` without leaking its module names."""
    repo = REPOS[key]
    if not repo.is_dir() or not any(repo.iterdir()):
        raise BaselineNotAvailable(f"{repo} is empty: run `git submodule update --init --recursive`")
    names = _top_level_names(repo)
    stub_modules = stub_modules or {}
    shadowed = {m: sys.modules.pop(m) for m in list(sys.modules)
                if m.split(".")[0] in names or m in stub_modules}
    before = set(sys.modules)
    sys.modules.update(stub_modules)
    sys.path.insert(0, str(repo))
    cwd = os.getcwd()
    if chdir:
        os.chdir(repo)
    try:
        yield repo
    finally:
        if chdir:
            os.chdir(cwd)
        sys.path.remove(str(repo))
        for m in set(sys.modules) - before:
            if m.split(".")[0] in names:
                sys.modules[f"_sota_{key}.{m}"] = sys.modules.pop(m)
        for m in stub_modules:
            sys.modules.pop(m, None)
        sys.modules.update(shadowed)


def _mmcv_stub() -> dict[str, types.ModuleType]:
    """BoRe-Depth needs mmcv only for norm-layer construction and checkpoint loading.

    mmcv is hard to install for every CUDA build, so these two calls are replaced by their
    plain PyTorch equivalents; weights are loaded by us with strict key checking instead.
    """
    mmcv = types.ModuleType("mmcv")
    cnn = types.ModuleType("mmcv.cnn")
    runner = types.ModuleType("mmcv.runner")

    def build_norm_layer(cfg, num_features):
        kind = cfg.get("type", "BN")
        layer = nn.BatchNorm2d(num_features) if kind in ("BN", "SyncBN", "BN2d") else nn.LayerNorm(num_features)
        return "bn", layer

    cnn.build_norm_layer = build_norm_layer
    runner.load_checkpoint = lambda *a, **k: None
    runner.load_state_dict = lambda *a, **k: None
    mmcv.cnn, mmcv.runner = cnn, runner
    return {"mmcv": mmcv, "mmcv.cnn": cnn, "mmcv.runner": runner}


def load_flexdepth(scale: str = "n"):
    """FlexDepth with a randomly initialised YOLO11-seg encoder (no COCO weights)."""
    # the YOLO model config is read relative to the repository root, hence chdir
    with isolated_import("flexdepth", chdir=True):
        from options import FlexdepthOptions
        import networks

        opt = FlexdepthOptions().parser.parse_args(
            ["--encoder_model_type", f"yolo11{scale}-seg", "--decoder_model_type", f"flex{scale}", "--dy_mu"])
        encoder = networks.YOLOEncoder(False, scale=f"{scale}-seg")
        decoder = networks.FlexDepthDecoder(encoder.num_ch_enc, opt.scales, scale=scale, opt=opt)
    return encoder, decoder


def load_zipdepth(variant: str = "base"):
    """Full ZipDepth model (its ``encoder`` and ``decoder`` are used separately)."""
    with isolated_import("zipdepth"):
        from zipdepth.model.architecture import create_model

        model = create_model(variant)
    return model


def load_bore():
    """BoRe-Depth DepthNet (MPViT-Tiny encoder + decoder); no pretrained weights are loaded."""
    with isolated_import("bore", stub_modules=_mmcv_stub()):
        from model.DepthNet import DepthNet

        model = DepthNet()
    return model


def load_zipdepth_loss(**kwargs) -> nn.Module:
    """ZipDepth's own step-3 loss: SSI (median/MAD) + multi-scale gradient matching, defaults as in its trainer."""
    with isolated_import("zipdepth"):
        from zipdepth.loss.depth_loss import ZipDepthLoss

        loss = ZipDepthLoss(**kwargs)
    return loss


def load_depth_anything_v2(encoder: str = "vitl") -> nn.Module:
    """Depth Anything V2 (randomly initialised; load the released weights with ``load_state_dict``)."""
    if encoder not in DAV2_CONFIGS:
        raise ValueError(f"unknown Depth Anything V2 encoder {encoder!r}: {sorted(DAV2_CONFIGS)}")
    with isolated_import("depth_anything_v2"):
        from depth_anything_v2.dpt import DepthAnythingV2

        model = DepthAnythingV2(**DAV2_CONFIGS[encoder])
    return model
