"""Strict loaders for published / external checkpoints (step 1 and ``init: public`` in step 3).

Supported formats:
* ``flexdepth``: directory containing ``encoder.pth`` and ``depth.pth`` (or ``encoder.pth`` alone when
  ``encoder_only=True``). Metadata keys written by FlexDepth's ``save_model`` are extracted and returned;
  scalar ``*.total_ops`` / ``*.total_params`` buffers left by the ``thop`` FLOPs profiler (present in the
  released KITTI weights) are dropped.
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


def _torch_load(path: Path):
    """Load a checkpoint with ``weights_only=True`` only: never unpickle arbitrary objects.

    External checkpoints come from third-party downloads, and ``weights_only=False`` would execute
    whatever the pickle contains. If a checkpoint holds non-tensor objects that the safe loader
    rejects, extract its state dict once in a trusted environment instead of loosening this call.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as e:
        raise RuntimeError(
            f"{path} cannot be loaded with weights_only=True (it contains non-tensor objects). "
            "Unsafe unpickling is disabled on purpose: re-save its state dict from a trusted copy."
        ) from e


PROFILER_BUFFER_SUFFIXES = (".total_ops", ".total_params")


def _drop_profiler_buffers(sd: dict) -> dict:
    """Remove the scalar ``total_ops`` / ``total_params`` buffers that ``thop.profile`` registers on every module.

    They are bookkeeping, not weights (the released FlexDepth KITTI checkpoints carry them, all zero), and the
    official evaluation drops them with a non-strict load. Only scalar tensors with these exact suffixes are
    removed, so any other mismatch still fails the strict load.
    """
    return {
        k: v for k, v in sd.items()
        if not (k.endswith(PROFILER_BUFFER_SUFFIXES) and torch.is_tensor(v) and v.numel() == 1)
    }


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

    raw_enc = _drop_profiler_buffers(dict(_torch_load(enc_path)))
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
        raw_dec = _drop_profiler_buffers(dict(_torch_load(dec_path)))
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
    ckpt = _torch_load(source)
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
