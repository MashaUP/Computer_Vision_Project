"""Tests for strict public-weights loading, ``init: public``, and ``eval-public``."""
import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from main import main as cli_main
from models.baselines import build_model
from models.public_weights import FLEXDEPTH_META_KEYS, load_public_weights
from scripts.train_flexdepth_nyu import RECIPE
from sota.loader import BaselineNotAvailable, isolated_import
from training.trainer import RunConfig, Trainer, evaluate_public


def _build(name, variant=None):
    try:
        return build_model(name, variant)
    except BaselineNotAvailable as e:
        pytest.skip(str(e))


def _state_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_flexdepth_decoder_matches_nyu_training_recipe():
    m = _build("flexdepth", "n")
    with isolated_import("flexdepth", chdir=True):
        import networks
        from options import FlexdepthOptions

        old_argv, sys.argv = sys.argv, [
            "train.py", *RECIPE, "--encoder_model_type", "yolo11n-seg", "--decoder_model_type", "flexn",
        ]
        try:
            opt = FlexdepthOptions().parse()
        finally:
            sys.argv = old_argv
        enc = networks.YOLOEncoder(False, scale="n-seg")
        dec = networks.FlexDepthDecoder(enc.num_ch_enc, opt.scales, scale="n", opt=opt)

    assert set(m.decoder.state_dict()) == set(dec.state_dict())


def test_flexdepth_public_weights_roundtrip_and_strictness(tmp_path):
    src = _build("flexdepth", "n")
    ckpt_dir = tmp_path / "weights_49"
    ckpt_dir.mkdir()
    meta = {
        "height": 288,
        "width": 384,
        "use_stereo": False,
        "decoder_model_type": "flexn",
        "c3k": False,
        "all_dysample": False,
        "dysample_group8": False,
        "upsample": "dysample_pl",
    }
    assert set(meta) == FLEXDEPTH_META_KEYS
    enc_sd = {**src.encoder.state_dict(), **meta}
    torch.save(enc_sd, ckpt_dir / "encoder.pth")
    torch.save(src.decoder.state_dict(), ckpt_dir / "depth.pth")

    dst = _build("flexdepth", "n")
    loaded_meta = load_public_weights(dst, ckpt_dir)
    assert loaded_meta == meta
    assert _state_equal(dst.encoder.state_dict(), src.encoder.state_dict())
    assert _state_equal(dst.decoder.state_dict(), src.decoder.state_dict())

    # Unexpected key in encoder.pth fails strict loading
    bad_enc = tmp_path / "bad_enc"
    bad_enc.mkdir()
    torch.save({**enc_sd, "unexpected.weight": torch.zeros(1)}, bad_enc / "encoder.pth")
    torch.save(src.decoder.state_dict(), bad_enc / "depth.pth")
    with pytest.raises(RuntimeError):
        load_public_weights(dst, bad_enc)

    # Missing key in depth.pth fails strict loading
    bad_dec = tmp_path / "bad_dec"
    bad_dec.mkdir()
    torch.save(enc_sd, bad_dec / "encoder.pth")
    dec_missing = dict(src.decoder.state_dict())
    dec_missing.pop(next(iter(dec_missing)))
    torch.save(dec_missing, bad_dec / "depth.pth")
    with pytest.raises(RuntimeError):
        load_public_weights(dst, bad_dec)

    # Mismatched decoder_model_type raises ValueError
    bad_type = tmp_path / "bad_type"
    bad_type.mkdir()
    torch.save({**enc_sd, "decoder_model_type": "flexs"}, bad_type / "encoder.pth")
    torch.save(src.decoder.state_dict(), bad_type / "depth.pth")
    with pytest.raises(ValueError, match="decoder_model_type"):
        load_public_weights(dst, bad_type)


def test_zipdepth_public_weights_roundtrip_and_strictness(tmp_path):
    src = _build("zipdepth", "base")
    raw_sd = {f"module._orig_mod.{k}": v for k, v in src.state_dict().items()}
    raw_sd["mean"] = torch.zeros(1, 3, 1, 1)
    raw_sd["std"] = torch.ones(1, 3, 1, 1)
    ckpt = tmp_path / "zipdepth.pth"
    torch.save({"model_state_dict": raw_sd}, ckpt)

    dst = _build("zipdepth", "base")
    assert load_public_weights(dst, ckpt) == {}
    assert _state_equal(dst.encoder.state_dict(), src.encoder.state_dict())
    assert _state_equal(dst.decoder.state_dict(), src.decoder.state_dict())

    bad_ckpt = tmp_path / "zipdepth_bad.pth"
    torch.save({"model_state_dict": {**raw_sd, "extra_head.weight": torch.zeros(1)}}, bad_ckpt)
    with pytest.raises(ValueError, match="unexpected ZipDepth"):
        load_public_weights(dst, bad_ckpt)


def test_bore_public_weights_roundtrip_and_strictness(tmp_path):
    src = _build("bore")
    sd = {f"depth_net.{k}": v for k, v in src.state_dict().items()}
    sd["pose_net.dummy.weight"] = torch.zeros(1)
    ckpt = tmp_path / "bore.ckpt"
    torch.save({"state_dict": sd}, ckpt)

    dst = _build("bore")
    assert load_public_weights(dst, ckpt) == {}
    assert _state_equal(dst.encoder.state_dict(), src.encoder.state_dict())
    assert _state_equal(dst.decoder.state_dict(), src.decoder.state_dict())

    bad_ckpt = tmp_path / "bore_bad.ckpt"
    torch.save({"state_dict": {**sd, "other.weight": torch.zeros(1)}}, bad_ckpt)
    with pytest.raises(ValueError, match="unexpected BoRe-Depth"):
        load_public_weights(dst, bad_ckpt)


@pytest.fixture
def synthetic_nyu(tmp_path, monkeypatch):
    data, runs = tmp_path / "data", tmp_path / "runs"
    (data / "rgb").mkdir(parents=True)
    (data / "depth").mkdir()
    (data / "splits").mkdir()
    rng = np.random.default_rng(0)
    lines = []
    for i in range(4):
        Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)).save(data / f"rgb/{i}.png")
        Image.fromarray(rng.integers(500, 9000, (480, 640)).astype(np.uint16)).save(data / f"depth/{i}.png")
        lines.append(f"rgb/{i}.png depth/{i}.png")
    (data / "splits/nyu_labeled_test.txt").write_text("\n".join(lines))
    (data / "pairs.txt").write_text("\n".join(lines))
    monkeypatch.setenv("LG_DATA_ROOT", str(data))
    monkeypatch.setenv("LG_RUNS_ROOT", str(runs))
    return data, runs


def test_evaluate_public_and_cli_flexdepth(synthetic_nyu, tmp_path):
    _, _ = synthetic_nyu
    src = _build("flexdepth", "n")
    ckpt_dir = tmp_path / "weights_49"
    ckpt_dir.mkdir()
    torch.save({**src.encoder.state_dict(), "height": 288, "width": 384, "decoder_model_type": "flexn"},
               ckpt_dir / "encoder.pth")
    torch.save(src.decoder.state_dict(), ckpt_dir / "depth.pth")

    out_json = tmp_path / "nyu_eval.json"
    res = evaluate_public(
        "flexdepth", "nyu", ckpt_dir, "splits/nyu_labeled_test.txt",
        variant="n", device="cpu", out_json=out_json,
    )
    assert set(res) == {"none", "median", "scale_shift"}
    assert all(np.isfinite(list(res["median"].values())))
    saved = json.loads(out_json.read_text())
    assert saved["image_size"] == [288, 384] and saved["n"] == 4

    cli_json = tmp_path / "cli_eval.json"
    rc = cli_main([
        "eval-public", "--model", "flexdepth", "--variant", "n", "--dataset", "nyu",
        "--weights", str(ckpt_dir), "--test-list", "splits/nyu_labeled_test.txt",
        "--device", "cpu", "--out", str(cli_json),
    ])
    assert rc == 0 and cli_json.exists()


def test_trainer_finetune_init_public_keeps_encoder_frozen(synthetic_nyu, tmp_path):
    _, runs = synthetic_nyu
    src = _build("flexdepth", "n")
    ckpt_dir = tmp_path / "weights_49"
    ckpt_dir.mkdir()
    torch.save({**src.encoder.state_dict(), "decoder_model_type": "flexn"}, ckpt_dir / "encoder.pth")
    torch.save(src.decoder.state_dict(), ckpt_dir / "depth.pth")

    cfg = RunConfig(
        task="finetune", name="ft_public", model="flexdepth", variant="n", dataset="nyu",
        train_list="pairs.txt", init="public", encoder_ckpt=str(ckpt_dir),
        image_size=[128, 160], batch_size=2, num_workers=0, epochs=1, warmup_epochs=1,
        max_steps_per_epoch=2, device="cpu", log_every=1,
    )
    Trainer(cfg).fit()
    after = torch.load(runs / "ft_public/model.pt", weights_only=True)
    for k, v in src.encoder.state_dict().items():
        assert torch.equal(after[f"encoder.{k}"], v), k
