"""CPU tests for step-3 self-supervised photometric loss (option A), TripletList, and ZipDepth fusion layer."""
import numpy as np
import pytest
import torch
from PIL import Image

from sota.loader import BaselineNotAvailable
from training.photometric import (
    PhotometricLoss,
    photometric_error,
    smoothness_loss,
    transformation_from_parameters,
    warp_image,
)
from training.trainer import RunConfig, Trainer


def test_warp_identity_and_transform_invert():
    aa = torch.tensor([[[0.05, -0.03, 0.02]]])
    t = torch.tensor([[[0.1, -0.05, 0.2]]])
    m_fwd = transformation_from_parameters(aa, t, invert=False)
    m_inv = transformation_from_parameters(aa, t, invert=True)
    assert torch.allclose(m_fwd @ m_inv, torch.eye(4).unsqueeze(0), atol=1e-5)

    h, w = 32, 48
    ys, xs = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    img = torch.stack([xs, ys, 0.5 * (xs + ys)], dim=0).unsqueeze(0)
    depth = torch.full((1, 1, h, w), 2.5)
    K = torch.tensor([[[w * 0.8, 0.0, w * 0.5, 0.0],
                       [0.0, h * 0.8, h * 0.5, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]]])
    inv_K = torch.linalg.inv(K)
    warped = warp_image(img, depth, torch.eye(4).unsqueeze(0), K, inv_K)
    assert photometric_error(warped, img).max().item() < 1e-4


def test_smoothness_loss_constant_and_gradient():
    img = torch.rand(2, 3, 24, 32)
    const_disp = torch.full((2, 1, 24, 32), 0.5, requires_grad=True)
    assert smoothness_loss(const_disp, img).item() == pytest.approx(0.0, abs=1e-6)

    var_disp = (torch.rand(2, 1, 24, 32) + 0.1).requires_grad_(True)
    loss = smoothness_loss(var_disp, img)
    assert loss.item() > 0.0
    loss.backward()
    assert var_disp.grad is not None and torch.isfinite(var_disp.grad).all()


def test_photometric_loss_static_and_gradients():
    crit = PhotometricLoss(smooth_weight=1e-3, pose_pretrained=False).train()
    h, w = 32, 48
    K = torch.tensor([[[w * 0.8, 0.0, w * 0.5, 0.0],
                       [0.0, h * 0.8, h * 0.5, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]]]).repeat(2, 1, 1)
    inv_K = torch.linalg.inv(K)
    frame = torch.rand(2, 3, h, w)
    depth = (torch.rand(2, 1, h, w) * 5.0 + 0.5).requires_grad_(True)

    out = crit(depth, frame, frame, frame, frame, frame, frame, K, inv_K)
    assert set(out) == {"loss", "reproj", "smooth"}
    assert all(torch.isfinite(v) for v in out.values())
    out["loss"].backward()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()
    assert any(p.grad is not None for p in crit.pose_net.parameters())


@pytest.fixture
def synthetic_triplets(tmp_path, monkeypatch):
    data, runs = tmp_path / "data", tmp_path / "runs"
    (data / "frames").mkdir(parents=True)
    rng = np.random.default_rng(1)
    lines = []
    for i in range(6):
        for tag in ("a", "b", "c"):
            Image.fromarray(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)).save(data / f"frames/{i}_{tag}.png")
        lines.append(f"frames/{i}_a.png frames/{i}_b.png frames/{i}_c.png")
    (data / "triplets.txt").write_text("\n".join(lines))
    monkeypatch.setenv("LG_DATA_ROOT", str(data))
    monkeypatch.setenv("LG_RUNS_ROOT", str(runs))
    return data, runs


def test_trainer_photometric_flexdepth_and_resume(synthetic_triplets):
    _, runs = synthetic_triplets
    cfg = RunConfig(
        task="finetune", name="photo_flex", model="flexdepth", variant="n", dataset="nyu",
        train_list="triplets.txt", init="random", bn_recal_batches=1, loss="photometric",
        pose_pretrained=False, image_size=[96, 128], batch_size=2, num_workers=0,
        epochs=2, warmup_epochs=1, max_steps_per_epoch=2, device="cpu", log_every=1,
    )
    try:
        tr = Trainer(cfg)
    except BaselineNotAvailable as e:
        pytest.skip(str(e))

    enc_before = {k: v.clone() for k, v in tr.model.encoder.state_dict().items()}
    tr.epochs = 1
    tr.fit()

    tr2 = Trainer(cfg)
    assert tr2.epoch == 1 and tr2.step == 2
    tr2.fit()

    after = torch.load(runs / "photo_flex/model.pt", weights_only=True)
    for k, v in enc_before.items():
        assert torch.equal(after[f"encoder.{k}"], v), k


def test_trainer_photometric_zipdepth_trains_cross_scale_only(synthetic_triplets):
    _, runs = synthetic_triplets
    cfg = RunConfig(
        task="finetune", name="photo_zip", model="zipdepth", variant="base", dataset="kitti",
        train_list="triplets.txt", init="random", bn_recal_batches=1, loss="photometric",
        pose_pretrained=False, image_size=[96, 128], batch_size=2, num_workers=0,
        epochs=1, warmup_epochs=1, max_steps_per_epoch=2, device="cpu", log_every=1,
    )
    try:
        tr = Trainer(cfg)
    except BaselineNotAvailable as e:
        pytest.skip(str(e))

    enc_before = {k: v.clone() for k, v in tr.model.encoder.state_dict().items()}
    tr.fit()

    after = torch.load(runs / "photo_zip/model.pt", weights_only=True)
    for k, v in enc_before.items():
        if k.startswith("cross_scale.low_to_high."):
            assert not torch.equal(after[f"encoder.{k}"], v), f"{k} should have updated"
        else:
            assert torch.equal(after[f"encoder.{k}"], v), f"{k} should stay frozen"
