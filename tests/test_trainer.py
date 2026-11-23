"""End-to-end trainer runs on a tiny synthetic dataset (CPU): pre-training, resume, fine-tuning
with a frozen encoder (random + BatchNorm re-estimation, and LeJEPA init), evaluation."""
import csv

import numpy as np
import pytest
import torch
from PIL import Image

from training.trainer import RunConfig, Trainer, evaluate_run


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
    Trainer(_pretrain_cfg(epochs=1)).fit()                       # stops after epoch 0
    tr = Trainer(_pretrain_cfg(epochs=2))                         # same run name: resumes
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
