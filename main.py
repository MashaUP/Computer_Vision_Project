"""Single entry point of the project: ``python main.py <command> [options]``.

Available now
  env     Python / PyTorch / CUDA, visible GPUs with the precision each one will use, data and runs roots.
  check   build every model and run one LeJEPA step (classic and dense) on random images.
          Runs on CPU by default, so it needs no GPU booking on the lab machines.

Next (see README, "Status"): download, pretrain, finetune, evaluate.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch


def cmd_env(_: argparse.Namespace) -> int:
    from utils import device, paths

    print(f"python {sys.version.split()[0]}")
    print(device.describe())
    print(f"data root : {paths.data_root()}\nruns root : {paths.runs_root()}\ncache root: {paths.cache_root()}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from models.baselines import build_model
    from sota.loader import BaselineNotAvailable
    from training.lejepa import LeJEPALoss, lejepa_step, load_recipe
    from utils.device import autocast, pick_device, precision_for
    from utils.seed import set_seed

    set_seed(0)
    dev = pick_device(args.device)
    precision = precision_for(dev)
    print(f"device {dev} | precision {precision}")
    failures = 0
    for name in args.models:
        try:
            model = build_model(name).to(dev).train()
        except BaselineNotAvailable as e:
            print(f"{name:10s} SKIPPED: {e}")
            continue
        enc_m = sum(p.numel() for p in model.encoder.parameters()) / 1e6
        dec_m = sum(p.numel() for p in model.decoder.parameters()) / 1e6
        row = [f"{name:10s} encoder {enc_m:.2f}M decoder {dec_m:.2f}M"]
        for variant in ("classic", "dense"):
            sizes = {"global_size": 256, "local_size": 128} if name == "meter" else {}
            crit = LeJEPALoss(model.deepest_channels(), load_recipe(variant, num_slices=args.slices, **sizes)).to(dev)
            x = torch.rand(args.batch, 3, 256, 320, device=dev)
            t0 = time.time()
            with autocast(dev, precision):
                out = lejepa_step(model.encode, crit, x)
            out["loss"].backward()
            ok = bool(torch.isfinite(out["loss"]))
            failures += not ok
            row.append(f"{variant} loss {out['loss'].item():.4f} ({'ok' if ok else 'NOT FINITE'}, {time.time() - t0:.1f}s)")
            model.zero_grad(set_to_none=True)
        print(" | ".join(row))
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env", help="show the environment").set_defaults(func=cmd_env)
    p = sub.add_parser("check", help="build the models and run one LeJEPA step")
    p.add_argument("--models", nargs="+", default=["flexdepth", "zipdepth", "bore", "meter"])
    p.add_argument("--device", default="cpu", help="cpu (default, no booking needed), cuda, cuda:0 ...")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--slices", type=int, default=64, help="SIGReg slices for the check (recipes use more)")
    p.set_defaults(func=cmd_check)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
