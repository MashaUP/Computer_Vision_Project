"""Strict loaders for published / external checkpoints (step 1 and ``init: public`` in step 3).

Supported formats:
* ``flexdepth``: directory containing ``encoder.pth`` and ``depth.pth`` (or ``encoder.pth`` alone when
  ``encoder_only=True``). Metadata keys written by FlexDepth's ``save_model`` are extracted and returned.
* ``zipdepth``: checkpoint file (``model_state_dict`` or raw state dict), stripping ``module.`` /
  ``_orig_mod.`` prefixes and ignoring only the ``mean`` / ``std`` normalisation buffers.
* ``bore``: PyTorch Lightning ``.ckpt`` (``state_dict``) or raw state dict, stripping ``module.`` and
  ``depth_net.`` prefixes and ignoring only the training-time ``pose_net.*`` weights.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import torch

from models.baselines import DepthBaseline

FLEXDEPTH_META_KEYS = frozenset({
    "height",
    "width",
    "use_stereo",
    "decoder_model_type",
    "c3k",
    "all_dysample",
    "dysample_group8",
    "upsample",
})


def _strip_prefixes(key: str, prefixes: tuple[str, ...]) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _torch_load(path: Path, *, allow_unsafe_fallback: bool = False):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        if not allow_unsafe_fallback:
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def _load_flexdepth(model: DepthBaseline, source: Path, *, encoder_only: bool) -> dict:
    if source.is_dir():
        enc_path = source / "encoder.pth"
        dec_path = source / "depth.pth"
    elif source.is_file() and encoder_only:
        enc_path = source
        dec_path = None
    else:
        raise ValueError(f"flexdepth weights must be a checkpoint directory (or encoder.pth with encoder_only=True): {source}")

    if not enc_path.is_file():
        raise FileNotFoundError(f"missing FlexDepth encoder weights: {enc_path}")

    raw_enc = dict(_torch_load(enc_path))
    meta = {k: raw_enc.pop(k) for k in list(raw_enc) if k in FLEXDEPTH_META_KEYS}
    expected_dec_type = f"flex{getattr(model, 'scale', 'n')}"
    if "decoder_model_type" in meta and meta["decoder_model_type"] != expected_dec_type:
        raise ValueError(
            f"checkpoint decoder_model_type={meta['decoder_model_type']!r} does not match model {expected_dec_type!r}"
        )
    model.encoder.load_state_dict(raw_enc, strict=True)

    if not encoder_only:
        if dec_path is None or not dec_path.is_file():
            raise FileNotFoundError(f"missing FlexDepth decoder weights: {dec_path}")
        raw_dec = _torch_load(dec_path)
        model.decoder.load_state_dict(raw_dec, strict=True)

    return meta


def _load_zipdepth(model: DepthBaseline, source: Path, *, encoder_only: bool) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"missing ZipDepth checkpoint: {source}")
    ckpt = _torch_load(source)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cleaned = {_strip_prefixes(k, ("module.", "_orig_mod.")): v for k, v in sd.items()}

    enc_sd, dec_sd, unexpected = {}, {}, []
    for k, v in cleaned.items():
        if k.startswith("encoder."):
            enc_sd[k[len("encoder."):]] = v
        elif k.startswith("decoder."):
            dec_sd[k[len("decoder."):]] = v
        elif k in ("mean", "std"):
            continue
        else:
            unexpected.append(k)

    if unexpected:
        raise ValueError(f"unexpected ZipDepth checkpoint keys: {unexpected[:5]}")

    model.encoder.load_state_dict(enc_sd, strict=True)
    if not encoder_only:
        model.decoder.load_state_dict(dec_sd, strict=True)
    return {}


def _load_bore(model: DepthBaseline, source: Path, *, encoder_only: bool) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"missing BoRe-Depth checkpoint: {source}")
    ckpt = _torch_load(source, allow_unsafe_fallback=True)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    enc_sd, dec_sd, unexpected = {}, {}, []
    for k, v in sd.items():
        key = _strip_prefixes(k, ("module.",))
        if key.startswith("pose_net."):
            continue
        if key.startswith("depth_net."):
            key = key[len("depth_net."):]
        if key.startswith("encoder."):
            enc_sd[key[len("encoder."):]] = v
        elif key.startswith("decoder."):
            dec_sd[key[len("decoder."):]] = v
        else:
            unexpected.append(k)

    if unexpected:
        raise ValueError(f"unexpected BoRe-Depth checkpoint keys: {unexpected[:5]}")

    model.encoder.load_state_dict(enc_sd, strict=True)
    if not encoder_only:
        model.decoder.load_state_dict(dec_sd, strict=True)
    return {}


def load_public_weights(model: DepthBaseline, source: str | Path, *, encoder_only: bool = False) -> dict:
    """Strictly load public/external weights into ``model`` and return any extracted metadata."""
    path = Path(source)
    if model.name == "flexdepth":
        return _load_flexdepth(model, path, encoder_only=encoder_only)
    if model.name == "zipdepth":
        return _load_zipdepth(model, path, encoder_only=encoder_only)
    if model.name == "bore":
        return _load_bore(model, path, encoder_only=encoder_only)
    raise ValueError(f"public weights loader is not supported for model {model.name!r}")
