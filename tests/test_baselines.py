"""The three baselines, built from their pinned submodules (skipped if the submodules are missing)."""
import pytest
import torch

from models.baselines import build_model
from sota.loader import BaselineNotAvailable
from training.lejepa import LeJEPALoss, lejepa_step, load_recipe

EXPECTED = {  # encoder params (M), pyramid channels finest -> deepest
    "flexdepth": (1.37, [16, 64, 128, 128, 256]),
    "zipdepth": (6.63, [24, 48, 96, 192, 384]),
    "bore": (5.63, [64, 96, 176, 216, 216]),
}


def _build(name):
    try:
        return build_model(name)
    except BaselineNotAvailable as e:
        pytest.skip(str(e))


@pytest.mark.parametrize("name", list(EXPECTED))
def test_pyramid(name):
    m = _build(name).eval()
    params, channels = EXPECTED[name]
    assert abs(sum(p.numel() for p in m.encoder.parameters()) / 1e6 - params) < 0.01
    with torch.no_grad():
        feats = m.encode(torch.rand(1, 3, 192, 640))
    assert [f.shape[1] for f in feats] == channels
    assert feats[-1].shape[-2:] == (192 // 32, 640 // 32)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_frozen_encoder_decoder_step(name):
    """Encoder frozen and in eval mode: a decoder step changes no encoder weight or BN statistic."""
    m = _build(name).freeze_encoder()
    m.train()  # must keep the frozen encoder in eval mode
    assert not m.encoder.training and m.decoder.training
    before = {k: v.clone() for k, v in m.encoder.state_dict().items()}
    opt = torch.optim.AdamW(m.decoder.parameters(), lr=1e-3)
    x = torch.rand(2, 3, 192, 640)
    with torch.no_grad():
        feats = m.encode(x)
    disp = m.to_disparity(m.decode(feats, x.shape[-2:]))
    disp.mean().backward()
    opt.step()
    assert all(torch.equal(v, before[k]) for k, v in m.encoder.state_dict().items())
    assert torch.isfinite(disp).all() and disp.min() >= 0 and disp.max() <= 1


def test_baselines_do_not_leak_module_names():
    _build("flexdepth"), _build("bore"), _build("zipdepth")
    import sys

    for leaked in ("networks", "layers", "options", "model", "datasets"):
        assert leaked not in sys.modules or not str(getattr(sys.modules[leaked], "__file__", "")).startswith(
            str(__import__("sota").__path__[0]))


@pytest.mark.parametrize("name", list(EXPECTED))
@pytest.mark.parametrize("variant", ["classic", "dense"])
def test_lejepa_step(name, variant):
    m = _build(name).train()
    crit = LeJEPALoss(m.deepest_channels(), load_recipe(variant, num_slices=64))
    out = lejepa_step(m.encode, crit, torch.rand(2, 3, 256, 320))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in m.encoder.parameters())
