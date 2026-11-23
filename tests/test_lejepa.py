import pytest
import torch

from data.views import make_views
from models.baselines import METER
from training.lejepa import LeJEPALoss, lejepa_step, load_recipe


@pytest.mark.parametrize("variant", ["classic", "dense"])
def test_recipe_and_views(variant, images):
    r = load_recipe(variant)
    g, l = make_views(images, r)
    assert len(g) == r.n_global and len(l) == r.n_local
    assert g[0].shape == (2, 3, r.global_size, r.global_size)
    assert l[0].shape == (2, 3, r.local_size, r.local_size)
    assert all(v.min() >= -1e-6 and v.max() <= 1.0 + 1e-6 for v in g + l)


def test_dense_pairs_share_the_crop(images):
    """With colour augmentation switched off, the two views of each dense crop are identical."""
    r = load_recipe("dense")
    torch.manual_seed(0)
    from data import views as V

    identity = lambda global_view, solarize_p=0.0: (lambda im: im)  # noqa: E731
    orig, V.colour_transform = V.colour_transform, identity
    try:
        g, l = make_views(images, r)
    finally:
        V.colour_transform = orig
    for a, b in zip(g[0::2] + l[0::2], g[1::2] + l[1::2]):
        assert torch.equal(a, b)


@pytest.mark.parametrize("variant", ["classic", "dense"])
def test_lejepa_step_on_meter(variant, images):
    model = METER("xxs").train()
    # METER's MobileViT blocks need view sizes divisible by 32 (phase 1 used 256 / 128 px);
    # fewer slices keep the CPU test fast
    r = load_recipe(variant, num_slices=64, global_size=256, local_size=128)
    crit = LeJEPALoss(model.deepest_channels(), r)
    out = lejepa_step(model.encode, crit, images)
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
    out["loss"].backward()
    grads = [p.grad for p in model.encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
