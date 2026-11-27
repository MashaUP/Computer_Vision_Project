"""One interface for every encoder-decoder in the comparison.

``encode(x)`` takes RGB in [0, 1] (B, 3, H, W), applies the input normalisation of the original
method and returns the feature pyramid, finest first: ``feats[-1]`` is the deepest map
(stride 32), the one LeJEPA is applied to. ``decode(feats, hw)`` returns the decoder's native
output and ``to_depth(out)`` converts it to a positive depth.

The ``output_kind`` of each model says what ``decode`` returns:

* ``sigmoid_disparity``: FlexDepth, Monodepth2 convention, in (0, 1).
* ``relu_inverse_depth``: ZipDepth, affine-invariant inverse depth in [0, inf).
* ``depth``: BoRe-Depth (1 / (10 * sigmoid + 0.01), SC-Depth style) and METER (metric depth).

How decoder outputs are trained in step 3 (photometric vs. GT loss) is still open.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sota import loader


class DepthBaseline(nn.Module):
    name = "base"
    output_kind = "depth"

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_frozen = False

    def freeze_encoder(self) -> "DepthBaseline":
        """Step 3: encoder frozen *and* in eval mode, so BatchNorm statistics never change.

        ``train()`` below keeps it in eval mode afterwards (FlexDepth's ``--freeze_encoder`` only
        disabled gradients and left BatchNorm updating).
        """
        self.encoder_frozen = True
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        return self

    def train(self, mode: bool = True) -> "DepthBaseline":
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()  # note: BoRe's MPViT.train() returns None, so never chain .eval()
        return self

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError

    def decode(self, feats: list[torch.Tensor], hw: tuple[int, int]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), x.shape[-2:])

    def to_depth(self, out: torch.Tensor, min_depth: float = 0.1, max_depth: float = 80.0) -> torch.Tensor:
        """Native output -> positive depth (scale is learned by the decoder; metrics align it)."""
        return out.clamp(min=min_depth)

    def deepest_channels(self) -> int:
        with torch.no_grad():
            was_training = self.encoder.training
            self.encoder.eval()
            c = self.encode(torch.rand(1, 3, 128, 128, device=next(self.parameters()).device))[-1].shape[1]
            self.encoder.train(was_training)
        return c


class FlexDepth(DepthBaseline):
    name = "flexdepth"
    output_kind = "sigmoid_disparity"

    def __init__(self, scale: str = "n"):
        with loader.isolated_import("flexdepth", chdir=True):
            from options import FlexdepthOptions
            import networks

            opts_obj = FlexdepthOptions()
            opts_obj.options = opts_obj.parser.parse_args(
                ["--encoder_model_type", f"yolo11{scale}-seg", "--decoder_model_type", f"flex{scale}", "--dy_mu"]
            )
            opts_obj._apply_decoder_defaults()
            opts_obj._validate_decoder_options()
            opt = opts_obj.options
            enc = networks.YOLOEncoder(False, scale=f"{scale}-seg")
            dec = networks.FlexDepthDecoder(enc.num_ch_enc, opt.scales, scale=scale, opt=opt)
        super().__init__(enc, dec)
        self.scale = scale
        # FlexDepth's FLOPs profiler forwards torch.empty() through the live encoder: BN running stats can come out NaN
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.reset_running_stats()

    def encode(self, x):
        return list(self.encoder(x))  # YOLOEncoder normalises internally: (x - 0.45) / 0.225

    def decode(self, feats, hw):
        return self.decoder(feats)[("disp", 0)]

    def to_depth(self, out, min_depth=0.1, max_depth=80.0):
        # Monodepth2 disp_to_depth: sigmoid disparity spans [1/max_depth, 1/min_depth]
        return 1.0 / (1.0 / max_depth + (1.0 / min_depth - 1.0 / max_depth) * out)


class ZipDepth(DepthBaseline):
    name = "zipdepth"
    output_kind = "relu_inverse_depth"

    def __init__(self, variant: str = "base"):
        model = loader.load_zipdepth(variant)
        super().__init__(model.encoder, model.decoder)
        self.register_buffer("mean", model.mean.clone(), persistent=False)
        self.register_buffer("std", model.std.clone(), persistent=False)
        self.variant = variant

    def encode(self, x):
        s_half, feats = self.encoder((x - self.mean) / self.std)  # ZipDepth normalises in its forward
        return [s_half, *feats]

    def decode(self, feats, hw):
        return self.decoder(feats[0], feats[1:], tuple(hw))

    def to_depth(self, out, min_depth=0.1, max_depth=80.0):
        return 1.0 / (out + 1.0 / max_depth)  # inverse depth, bounded by max_depth


class BoReDepth(DepthBaseline):
    name = "bore"
    output_kind = "depth"

    def __init__(self):
        model = loader.load_bore()
        super().__init__(model.encoder, model.decoder)
        self.register_buffer("mean", torch.full((1, 3, 1, 1), 0.45), persistent=False)
        self.register_buffer("std", torch.full((1, 3, 1, 1), 0.225), persistent=False)

    def encode(self, x):
        return list(self.encoder((x - self.mean) / self.std))  # SC-Depth style normalisation

    def decode(self, feats, hw):
        return self.decoder(feats)[0]  # finest-scale depth


class METER(DepthBaseline):
    """Phase-1 METER (``models/meter.py``) behind the same interface."""
    name = "meter"
    output_kind = "depth"

    def __init__(self, arch: str = "xxs", skip_from_transformer: bool = False):
        from models.meter import METERModel

        m = METERModel(arch, skip_from_transformer=skip_from_transformer, verbose=False)
        super().__init__(m.encoder, m.decoder)
        self.arch = arch
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def encode(self, x):
        out, skips = self.encoder((x - self.mean) / self.std)  # ImageNet normalisation (phase 1)
        return [*skips, out]

    def decode(self, feats, hw):
        return self.decoder(feats[-1], feats[:-1])


def build_model(name: str, variant: str | None = None) -> DepthBaseline:
    """``flexdepth`` (n|s|m...), ``zipdepth`` (base|small), ``bore``, ``meter`` (xxs|xs|s)."""
    if name == "flexdepth":
        return FlexDepth(variant or "n")
    if name == "zipdepth":
        return ZipDepth(variant or "base")
    if name == "bore":
        return BoReDepth()
    if name == "meter":
        return METER(variant or "xxs")
    raise ValueError(f"unknown model {name!r}")
