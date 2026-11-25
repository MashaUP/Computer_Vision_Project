"""FlexDepth NYU training script: the original trainer on a tiny synthetic NYU raw set (CPU, a few seconds)."""
import importlib.util

import numpy as np
import pytest
import torch
from PIL import Image

from sota.loader import SOTA_DIR

for mod in ("tensorboardX", "wandb", "prefetch_generator", "skimage", "six", "thop"):
    pytest.importorskip(mod)
pytestmark = pytest.mark.skipif(not (SOTA_DIR / "flexdepth" / "trainer.py").exists(), reason="flexdepth submodule missing")


def _load_script():
    spec = importlib.util.spec_from_file_location("train_flexdepth_nyu", SOTA_DIR.parent / "scripts" / "train_flexdepth_nyu.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_one_step_on_synthetic_nyu(tmp_path, monkeypatch):
    monkeypatch.setenv("LG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LG_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("LG_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    rng = np.random.default_rng(0)
    rows = []
    for k in range(4):
        frames = []
        for j in range(3):
            f = data / "nyu/raw/scene_0001" / f"r-{1000 + 10 * k + j:.6f}-{k * 3 + j:010d}.ppm"
            f.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)).save(f)
            frames.append(str(f.relative_to(data)))
        rows.append(" ".join(frames))
    (data / "splits").mkdir()
    (data / "splits/train.txt").write_text("\n".join(rows) + "\n")
    (data / "splits/val.txt").write_text("\n".join(rows[:2]) + "\n")

    import torchvision
    # the pose network always asks torchvision for ImageNet ResNet-18 weights: no network in the test
    monkeypatch.setattr("torchvision.models._api.load_state_dict_from_url",
                        lambda *a, **k: torchvision.models.resnet18().state_dict())
    script = _load_script()
    work = script.prepare_workdir("n", pretrained=False)
    monkeypatch.chdir(work)                                    # cfg/ is symlinked here, as in the real run
    extra = ["--weights_init", "scratch", "--device", "cpu", "--num_workers", "0", "--batch_size", "2",
             "--height", "96", "--width", "128", "--num_epochs", "1", "--no_eval"]
    # later options win in argparse, so the tiny test values override the recipe's
    flex_argv = [*script.RECIPE, "--encoder_model_type", "yolo11n-seg", "--decoder_model_type", "flexn",
                 "--data_path", str(data), "--log_dir", str(tmp_path / "runs/t"), "--model_name", "m", *extra]
    with script.build(flex_argv) as (_, NYUTrainer, opts):
        trainer = NYUTrainer(opts, data / "splits/train.txt", data / "splits/val.txt")
        inputs = next(iter(trainer.train_loader))
        assert inputs[("color_aug", 0, 0)].shape == (2, 3, 96, 128) and set(k for k in inputs if k[0] == "color") >= {("color", -1, 0)}
        trainer.set_train()
        for _ in range(2):
            outputs, losses = trainer.process_batch(inputs)
            assert torch.isfinite(losses["loss"])
            losses["loss"].backward()
            trainer.model_optimizer.step()
            trainer.model_optimizer.zero_grad()
