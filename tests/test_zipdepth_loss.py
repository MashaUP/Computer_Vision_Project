"""Step 3 for ZipDepth: its own loss on Depth Anything V2 pseudo-depth (CPU, random teacher weights)."""
import numpy as np
import pytest
import torch
from PIL import Image

from data.datasets import PseudoDepthList, frame_path, load_pseudo_depth, pseudo_depth_path, target_frame
from sota.loader import BaselineNotAvailable, load_zipdepth_loss
from training.trainer import RunConfig, Trainer


def _loss_or_skip():
    try:
        return load_zipdepth_loss()
    except BaselineNotAvailable as e:
        pytest.skip(str(e))


def test_zipdepth_loss_ignores_the_stored_scale_and_shift():
    """The per-image normalisation of the stored pseudo-depth must not change the loss."""
    loss = _loss_or_skip()
    pred, target = torch.rand(2, 1, 32, 48), torch.rand(2, 1, 32, 48)
    a, _ = loss(pred, target)
    b, _ = loss(pred, 3.7 * target + 0.2)
    assert torch.allclose(a, b, atol=1e-5)
    c, _ = loss(target, target)
    assert c.item() < 1e-6


def test_target_frame_and_frame_path(tmp_path):
    assert target_frame(["a.png", "b.png", "c.png"]) == "b.png"
    assert target_frame(["a.png"]) == "a.png"
    raw = "nyu/raw/kitchen_0001/r-1.ppm"
    assert frame_path(tmp_path, raw, "nyu") == tmp_path / raw          # no rectified PNG: the raw frame
    png = tmp_path / "nyu/rect/kitchen_0001/r-1.png"
    png.parent.mkdir(parents=True)
    png.touch()
    assert frame_path(tmp_path, raw, "nyu") == png
    assert frame_path(tmp_path, raw, "kitti") == tmp_path / raw
    assert pseudo_depth_path(tmp_path, "dav2_vitl", raw) == tmp_path / "pseudo_depth/dav2_vitl/nyu/raw/kitchen_0001/r-1.png"


@pytest.fixture
def frames(tmp_path, monkeypatch):
    data, runs = tmp_path / "data", tmp_path / "runs"
    (data / "frames").mkdir(parents=True)
    rng = np.random.default_rng(2)
    lines = []
    for i in range(4):
        for tag in ("a", "b", "c"):
            Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(data / f"frames/{i}_{tag}.png")
        lines.append(f"frames/{i}_a.png frames/{i}_b.png frames/{i}_c.png")
    (data / "triplets.txt").write_text("\n".join(lines))
    monkeypatch.setenv("LG_DATA_ROOT", str(data))
    monkeypatch.setenv("LG_RUNS_ROOT", str(runs))
    monkeypatch.setenv("LG_CACHE_ROOT", str(tmp_path / "cache"))
    return data, runs


def test_make_pseudo_depth_writes_resumes_and_feeds_the_dataset(frames):
    data, _ = frames
    from scripts import make_pseudo_depth as mpd

    try:
        res = mpd.run("nyu", [data / "triplets.txt"], teacher="dav2_test", encoder="vits", weights=None,
                      input_size=70, device="cpu", limit=3)
    except BaselineNotAvailable as e:
        pytest.skip(str(e))
    assert res["written"] == 3
    out = pseudo_depth_path(data, "dav2_test", "frames/0_b.png")      # the middle frame of the triplet
    png = np.asarray(Image.open(out))
    assert png.dtype == np.uint16 and png.shape == (60, 80) and png.max() == 65535
    assert not pseudo_depth_path(data, "dav2_test", "frames/0_a.png").exists()

    with pytest.raises(FileNotFoundError, match="1/4 frames"):          # one frame still missing
        PseudoDepthList(data / "triplets.txt", data, (32, 48), "nyu", teacher="dav2_test")
    assert mpd.run("nyu", [data / "triplets.txt"], teacher="dav2_test", encoder="vits", weights=None,
                   input_size=70, device="cpu")["written"] == 1         # resume: only the missing one
    ds = PseudoDepthList(data / "triplets.txt", data, (32, 48), "nyu", teacher="dav2_test", train=False)
    rgb, target = ds[0]
    assert rgb.shape == (3, 32, 48) and target.shape == (1, 32, 48)
    assert 0.0 <= target.min() and target.max() <= 1.0
    assert torch.allclose(load_pseudo_depth(out).max(), torch.tensor(1.0))


def test_to_png16_handles_zero_and_negative():
    from scripts.make_pseudo_depth import to_png16

    assert to_png16(np.zeros((3, 4), np.float32)).max() == 0
    q = to_png16(np.array([[-1.0, 0.0, 2.0, 4.0]], np.float32))
    assert q.tolist() == [[0, 0, 32768, 65535]]


def _write_pseudo(data, teacher="dav2_vitl"):
    rng = np.random.default_rng(3)
    for i in range(4):
        p = pseudo_depth_path(data, teacher, f"frames/{i}_b.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 65535, (60, 80)).astype(np.uint16)).save(p)


def test_trainer_zipdepth_loss_trains_the_decoder_only(frames):
    data, runs = frames
    _write_pseudo(data)
    cfg = RunConfig(
        task="finetune", name="zip_own", model="zipdepth", variant="base", dataset="nyu",
        train_list="triplets.txt", init="random", bn_recal_batches=1, loss="zipdepth",
        image_size=[64, 96], batch_size=2, num_workers=0, epochs=1, warmup_epochs=1,
        max_steps_per_epoch=2, device="cpu", log_every=1,
    )
    try:
        tr = Trainer(cfg)
    except BaselineNotAvailable as e:
        pytest.skip(str(e))
    assert tr.val_alignment == "scale_shift"
    enc_before = {k: v.clone() for k, v in tr.model.encoder.state_dict().items()}
    dec_before = {k: v.clone() for k, v in tr.model.decoder.state_dict().items()}
    tr.fit()

    after = torch.load(runs / "zip_own/model.pt", weights_only=True)
    for k, v in enc_before.items():
        if k.startswith("cross_scale.low_to_high."):
            assert not torch.equal(after[f"encoder.{k}"], v), f"{k} should have updated"
        else:
            assert torch.equal(after[f"encoder.{k}"], v), f"{k} should stay frozen"
    assert any(not torch.equal(after[f"decoder.{k}"], v) for k, v in dec_before.items())
    row = (runs / "zip_own/metrics.csv").read_text().splitlines()
    assert "train_ssi" in row[0] and "train_grad" in row[0]


def test_zipdepth_loss_is_refused_for_other_models(frames):
    data, _ = frames
    _write_pseudo(data)
    cfg = RunConfig(task="finetune", name="zip_flex", model="flexdepth", variant="n", dataset="nyu",
                    train_list="triplets.txt", init="random", loss="zipdepth", image_size=[64, 96],
                    num_workers=0, device="cpu")
    try:
        with pytest.raises(ValueError, match="ZipDepth's own loss"):
            Trainer(cfg)
    except BaselineNotAvailable as e:
        pytest.skip(str(e))
