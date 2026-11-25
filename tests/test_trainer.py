"""End-to-end trainer runs on a tiny synthetic dataset (CPU): pre-training, resume, fine-tuning
with a frozen encoder (random + BatchNorm re-estimation, and LeJEPA init), evaluation."""
import csv

import numpy as np
import pytest
import torch
from PIL import Image

from models.baselines import build_model
from sota.loader import BaselineNotAvailable
from training.losses import silog_loss
from training.trainer import EpochSampler, RunConfig, Trainer, evaluate_run


@pytest.fixture
def roots(tmp_path, monkeypatch):
    data, runs = tmp_path / "data", tmp_path / "runs"
    (data / "rgb").mkdir(parents=True)
    (data / "depth").mkdir()
    rng = np.random.default_rng(0)
    lines = []
    for i in range(8):
        Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).save(data / f"rgb/{i}.png")
        depth_mm = rng.integers(500, 9000, (96, 128)).astype(np.uint16)
        Image.fromarray(depth_mm).save(data / f"depth/{i}.png")
        lines.append(f"rgb/{i}.png depth/{i}.png")
    (data / "pairs.txt").write_text("\n".join(lines))
    (data / "frames.txt").write_text("\n".join(ln.split()[0] for ln in lines))
    monkeypatch.setenv("LG_DATA_ROOT", str(data))
    monkeypatch.setenv("LG_RUNS_ROOT", str(runs))
    return data, runs


def _pretrain_cfg(**kw):
    base = dict(task="pretrain", name="pre", model="meter", variant="xxs", dataset="nyu", recipe="dense",
                train_list="frames.txt", image_size=[128, 192], batch_size=2, num_workers=0, epochs=2,
                warmup_epochs=1, max_steps_per_epoch=2, log_every=1, device="cpu")
    base.update(kw)
    return RunConfig(**base)


def test_pretrain_resume_and_finetune(roots):
    _, runs = roots
    first = Trainer(_pretrain_cfg(epochs=2))
    first.epochs = 1                                              # interrupted after epoch 0
    first.fit()
    tr = Trainer(_pretrain_cfg(epochs=2))                         # same run name and config: resumes
    assert tr.epoch == 1 and tr.step == 2
    tr.fit()
    rows = list(csv.DictReader(open(runs / "pre/metrics.csv")))
    assert [r["epoch"] for r in rows] == ["0", "1"] and rows[-1]["step"] == "4"
    pretrained = torch.load(runs / "pre/encoder.pt", weights_only=True)

    ft = dict(task="finetune", model="meter", variant="xxs", dataset="nyu", train_list="pairs.txt",
              val_list="pairs.txt", image_size=[128, 192], batch_size=2, num_workers=0, epochs=1,
              warmup_epochs=1, max_steps_per_epoch=2, device="cpu", log_every=1)
    Trainer(RunConfig(name="ft_lejepa", init="lejepa", encoder_ckpt=str(runs / "pre/encoder.pt"), **ft)).fit()
    after = torch.load(runs / "ft_lejepa/model.pt", weights_only=True)
    for k, v in pretrained.items():                              # frozen encoder: untouched, BN stats included
        assert torch.equal(after[f"encoder.{k}"], v), k
    row = list(csv.DictReader(open(runs / "ft_lejepa/metrics.csv")))[-1]
    assert float(row["val_abs_rel"]) > 0

    Trainer(RunConfig(name="ft_random", init="random", bn_recal_batches=2, **ft)).fit()
    res = evaluate_run(RunConfig(name="ft_random", init="random", **ft), "pairs.txt")
    assert set(res) == {"none", "median", "scale_shift"} and all(np.isfinite(list(res["median"].values())))
    assert (runs / "ft_random/test_metrics_pairs.json").exists()


@pytest.mark.parametrize("recipe", ["classic", "classic_phase1", "dense_phase1"])
def test_pretrain_recipes_run(roots, recipe):
    model = "flexdepth" if recipe == "classic" else "meter"      # METER needs view sizes divisible by 32
    cfg = _pretrain_cfg(name=f"pre_{recipe}", recipe=recipe, model=model, variant=None, epochs=1)
    try:
        Trainer(cfg).fit()
    except Exception as e:  # noqa: BLE001
        if "submodule" in str(e):
            pytest.skip(str(e))
        raise


def test_resume_refuses_a_different_config(roots):
    first = Trainer(_pretrain_cfg(epochs=2))
    first.epochs = 1
    first.fit()
    with pytest.raises(ValueError, match="lr"):
        Trainer(_pretrain_cfg(epochs=2, lr=1e-3))
    Trainer(_pretrain_cfg(epochs=2, num_workers=1, log_every=7))  # harmless keys may change


def test_resume_in_the_middle_of_an_epoch(roots):
    _, runs = roots
    cfg = _pretrain_cfg(epochs=1, max_steps_per_epoch=4, ckpt_minutes=0)   # checkpoint after every step
    tr, orig, calls = Trainer(cfg), None, []
    orig = tr._loss

    def interrupted(batch):
        if len(calls) == 2:
            raise KeyboardInterrupt
        calls.append(1)
        return orig(batch)

    tr._loss = interrupted
    with pytest.raises(KeyboardInterrupt):
        tr.fit()
    tr2 = Trainer(cfg)
    assert (tr2.epoch, tr2.batch_in_epoch, tr2.step, tr2.n) == (0, 2, 2, 2) and tr2.sums["loss"] > 0
    tr2.fit()
    rows = list(csv.DictReader(open(runs / "pre/metrics.csv")))
    assert len(rows) == 1 and rows[0]["step"] == "4"


def test_finetune_val_every_and_bn_list(roots):
    _, runs = roots
    cfg = RunConfig(task="finetune", name="ft", model="meter", variant="xxs", dataset="nyu", train_list="pairs.txt",
                    val_list="pairs.txt", bn_list="frames.txt", init="random", bn_recal_batches=1, epochs=2,
                    val_every=2, warmup_epochs=1, image_size=[128, 192], batch_size=2, num_workers=0,
                    max_steps_per_epoch=1, device="cpu")
    Trainer(cfg).fit()
    rows = list(csv.DictReader(open(runs / "ft/metrics.csv")))
    assert rows[0]["val_abs_rel"] == "" and float(rows[1]["val_abs_rel"]) > 0     # same columns every epoch


def test_epoch_sampler_resumes_with_the_tail_of_the_permutation():
    s = EpochSampler(10, seed=0)
    s.set(3)
    full = list(s)
    s.set(3, 4)
    assert sorted(full) == list(range(10)) and list(s) == full[4:] and len(s) == 6
    s.set(4)
    assert list(s) != full                                        # another epoch, another order


def test_silog_without_valid_pixel_is_zero_not_nan():
    p = torch.rand(2, 1, 4, 4, requires_grad=True) + 0.1
    loss = silog_loss(p, p.detach(), torch.zeros_like(p, dtype=torch.bool))
    assert loss == 0
    loss.backward()


def test_depth_conversion_uses_the_default_min_depth():
    """FlexDepth at sigmoid 0.5 must give ~0.2 m (min_depth 0.1), not 0.002 m (min_depth 1e-3)."""
    try:
        model = build_model("flexdepth")
    except BaselineNotAvailable as e:
        pytest.skip(str(e))
    tr = Trainer.__new__(Trainer)
    tr.model, tr.cap = model, 10.0
    depth = tr._depth(torch.full((1, 1, 4, 4), 0.5), (4, 4))
    assert depth.min() >= 0.1 and depth.max() <= 10.0


def test_deepest_channels_on_the_model_device():
    """The probe image is created where the parameters are (it crashed on GPU/MPS)."""
    assert build_model("meter").to("meta").deepest_channels() == 160


def test_check_list_rejects_test_and_val_scenes(tmp_path, monkeypatch, capsys):
    from data.datasets import check_list
    monkeypatch.setenv("LG_DATA_ROOT", str(tmp_path))
    (tmp_path / "splits").mkdir()
    lst = tmp_path / "frames.txt"
    lst.write_text("nyu/raw/kitchen_0001a/r-1000.1-1.ppm\nnyu/raw/office_0003/r-1000.2-2.ppm\n")
    check_list(lst, "nyu", "pretrain")                                     # no split files: warns, does not fail
    assert "missing" in capsys.readouterr().out
    (tmp_path / "splits/nyu_test_scenes.txt").write_text("kitchen_0001\n")
    (tmp_path / "splits/nyu_raw_val_scenes.txt").write_text("office_0003\n")
    with pytest.raises(ValueError, match="kitchen_0001"):
        check_list(lst, "nyu", "val")                                      # val role: only the test scenes are banned
    lst.write_text("nyu/raw/office_0003/r-1000.2-2.ppm\n")
    check_list(lst, "nyu", "val")
    with pytest.raises(ValueError, match="office_0003"):
        check_list(lst, "nyu", "train")
    lab = tmp_path / "labeled.txt"                                         # labelled pairs: the scene is the 3rd column
    lab.write_text("nyu/labeled/train/0001_rgb.png nyu/labeled/train/0001_depth.png office_0003a\n")
    with pytest.raises(ValueError, match="office_0003"):
        check_list(lab, "nyu", "train")
    other = tmp_path / "other.txt"
    other.write_text("some/path.png other/depth.png\n")
    check_list(other, "nyu", "train")
    assert "do not name their scene" in capsys.readouterr().out
    check_list(lst, "kitti", "train")                                      # KITTI: not checked yet
