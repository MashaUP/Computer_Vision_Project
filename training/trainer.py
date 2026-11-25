"""One trainer for the whole protocol.

* ``task: pretrain``: LeJEPA pre-training of the encoder, classic or dense (any recipe in
  ``configs/lejepa/``). Trains encoder + projector; writes ``encoder.pt``.
* ``task: finetune``: step 3. The encoder is initialised (``random``, ``lejepa`` from a pre-training
  run, or ``public``), **frozen in eval mode**, and only the decoder is trained with the same
  procedure for every arm. For ``random`` the BatchNorm statistics are first re-estimated on
  the unlabelled frames of ``bn_list`` (without it, on the RGB column of ``train_list``).

Common to both: seeding, automatic precision (fp32 on Pascal, fp16 autocast otherwise, never
bf16), AdamW, per-epoch linear warm-up + cosine schedule (as in phase 1), gradient clipping,
CSV logging, and **checkpoint + automatic resume**: ``last.ckpt`` is written atomically at the end
of every epoch and every ``ckpt_minutes``; re-running the same command continues from it (lab
machines restart every three months). The sample order is the same as without the interruption; the
random augmentation after a mid-epoch resume is not. A resume with a different configuration (other
than ``device``, ``num_workers``, ``log_every``, ``ckpt_minutes``) is refused.

Run directory: ``<runs root>/<name>/`` with ``config.yaml``, ``metrics.csv``, ``last.ckpt``.
"""
from __future__ import annotations

import dataclasses
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

from data.datasets import DepthPairs, ImageList, check_list
from evaluation.metrics import CAPS, METRICS, MIN_DEPTH, depth_metrics, mean_metrics
from models.baselines import build_model
from training.lejepa import LeJEPALoss, lejepa_step, load_recipe
from training.losses import silog_loss
from utils import paths
from utils.device import autocast, pick_device, precision_for
from utils.io import append_csv_row, save_json
from utils.seed import seed_worker, set_seed

DEPTH_SCALE = {"nyu": 1000.0, "kitti": 256.0}
RESUME_FREE = {"device", "num_workers", "log_every", "ckpt_minutes"}   # may change between a run and its resume


@dataclass
class RunConfig:
    task: str                              # "pretrain" | "finetune"
    name: str                              # run directory name
    model: str                             # flexdepth | zipdepth | bore | meter
    variant: str | None = None
    dataset: str = "nyu"                   # nyu | kitti
    train_list: str = ""                   # list file (absolute, or relative to the data root)
    val_list: str | None = None
    image_size: list[int] = field(default_factory=lambda: [256, 320])  # (H, W) fed to the network
    crop: list[int] | None = None          # finetune: random crop at full resolution (KITTI: [352, 1216])
    # pretrain
    recipe: str = "classic"
    # finetune
    init: str = "random"                   # random | lejepa | public
    encoder_ckpt: str | None = None        # lejepa: path to encoder.pt of a pre-training run
    loss: str = "silog"                    # silog (ground truth); photometric: pending the step-3 decision
    bn_recal_batches: int = 200
    bn_list: str | None = None             # random init: list of unlabelled frames for BatchNorm (default: train_list)
    # optimisation (pretrain: recipe values unless set here)
    epochs: int | None = None
    batch_size: int = 32
    lr: float | None = None
    weight_decay: float | None = None
    warmup_epochs: int | None = None
    eta_min_ratio: float | None = None
    grad_clip: float = 1.0
    num_workers: int = 4
    seed: int = 0
    deterministic: bool = False
    device: str = "auto"
    ckpt_minutes: float = 30.0
    log_every: int = 50
    val_every: int = 1                     # validate every N epochs (and always after the last one)
    max_steps_per_epoch: int | None = None  # pilot runs: cut every epoch short

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict | None = None) -> "RunConfig":
        cfg = yaml.safe_load(Path(path).read_text()) or {}
        cfg.update(overrides or {})
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(cfg) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**cfg)


def parse_overrides(pairs: list[str]) -> dict:
    """``key=value`` strings, values parsed as YAML (``epochs=2``, ``crop=[352,1216]``)."""
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k] = yaml.safe_load(v)
    return out


class EpochSampler(Sampler):
    """Deterministic permutation per (seed, epoch), started at ``start`` (to resume mid-epoch)."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set(self, epoch: int, start: int = 0):
        self.epoch, self.start = epoch, start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())

    def __len__(self):
        return self.n - self.start


def _resolve(p: str | None) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else paths.data_root() / p


def _schedule(opt, epochs, warmup, eta_min_ratio, lr):
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=max(warmup, 1))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warmup, 1), eta_min=lr * eta_min_ratio)
    return torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[max(warmup, 1)])


def _save_atomic(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s):
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


@torch.no_grad()
def recalibrate_batchnorm(encode, encoder: torch.nn.Module, batches, device) -> int:
    """Re-estimate BatchNorm running statistics (cumulative average) without touching any weight."""
    bns = [m for m in encoder.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    encoder.train()
    n = 0
    for x in batches:
        encode(x.to(device))
        n += 1
    encoder.eval()
    return n


class Trainer:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.dir = paths.runs_root() / cfg.name
        self.dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.seed, cfg.deterministic)
        self.device = pick_device(cfg.device)
        self.precision = precision_for(self.device)
        self.model = build_model(cfg.model, cfg.variant).to(self.device)
        self.cap = CAPS[cfg.dataset]
        if cfg.task == "pretrain":
            self._setup_pretrain()
        elif cfg.task == "finetune":
            self._setup_finetune()
        else:
            raise ValueError(f"unknown task {cfg.task!r}")
        self.opt = torch.optim.AdamW(self.params, lr=self.lr, weight_decay=self.wd)
        self.sched = _schedule(self.opt, self.epochs, self.warmup, self.eta_min_ratio, self.lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "fp16-mixed")
        self.sampler = EpochSampler(len(self.train_set), cfg.seed)
        self.epoch, self.batch_in_epoch, self.step = 0, 0, 0
        self.sums, self.n = {}, 0      # running train means of the current epoch
        self.last_ckpt = self.dir / "last.ckpt"
        if self.last_ckpt.exists():
            self._resume()
        else:
            (self.dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))

    # ------------------------------------------------------------------ setup
    def _hp(self, name, default):
        v = getattr(self.cfg, name)
        return default if v is None else v

    def _setup_pretrain(self):
        cfg = self.cfg
        self.recipe = load_recipe(cfg.recipe, dataset=cfg.dataset)
        self.model.train()
        self.criterion = LeJEPALoss(self.model.deepest_channels(), self.recipe).to(self.device)
        self.params = [p for p in list(self.model.encoder.parameters()) + list(self.criterion.parameters())
                       if p.requires_grad]
        self.lr, self.wd = self._hp("lr", self.recipe.lr), self._hp("weight_decay", self.recipe.weight_decay)
        self.epochs, self.warmup = self._hp("epochs", self.recipe.epochs), self._hp("warmup_epochs", self.recipe.warmup_epochs)
        self.eta_min_ratio = self._hp("eta_min_ratio", self.recipe.eta_min_ratio)
        check_list(_resolve(cfg.train_list), cfg.dataset, "pretrain")
        self.train_set: Dataset = ImageList(_resolve(cfg.train_list), paths.data_root(), cfg.image_size)
        self.val_set = None

    def _setup_finetune(self):
        cfg = self.cfg
        if cfg.loss != "silog":
            raise NotImplementedError("photometric step-3 loss: pending the decision on option (A)")
        scale = DEPTH_SCALE[cfg.dataset]
        check_list(_resolve(cfg.train_list), cfg.dataset, "train")
        if cfg.val_list:
            check_list(_resolve(cfg.val_list), cfg.dataset, "val")
        self.train_set = DepthPairs(_resolve(cfg.train_list), paths.data_root(), cfg.image_size, scale,
                                    crop=cfg.crop, train=True)
        self.val_set = (DepthPairs(_resolve(cfg.val_list), paths.data_root(), cfg.image_size, scale)
                        if cfg.val_list else None)
        if not (self.dir / "last.ckpt").exists():   # initialisation only for a fresh run
            self._init_encoder()
        self.model.freeze_encoder()
        self.model.train()                           # decoder trains, encoder stays in eval
        self.params = [p for p in self.model.decoder.parameters() if p.requires_grad]
        self.lr, self.wd = self._hp("lr", 1e-4), self._hp("weight_decay", 1e-2)
        self.epochs, self.warmup = self._hp("epochs", 25), self._hp("warmup_epochs", 1)
        self.eta_min_ratio = self._hp("eta_min_ratio", 1e-2)

    def _init_encoder(self):
        cfg = self.cfg
        if cfg.init == "lejepa":
            state = torch.load(cfg.encoder_ckpt, map_location="cpu", weights_only=True)
            self.model.encoder.load_state_dict(state, strict=True)
        elif cfg.init == "random":
            if cfg.bn_list is None:
                print("[bn] warning: no bn_list, using the RGB frames of train_list (not unlabelled frames)")
            check_list(_resolve(cfg.bn_list or cfg.train_list), cfg.dataset, "bn")
            frames = ImageList(_resolve(cfg.bn_list or cfg.train_list), paths.data_root(), cfg.image_size)
            frames.items = [it[:1] for it in frames.items]
            loader = DataLoader(frames, batch_size=self.cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(cfg.seed))
            batches = (x for i, x in enumerate(loader) if i < cfg.bn_recal_batches)
            n = recalibrate_batchnorm(self.model.encode, self.model.encoder, batches, self.device)
            print(f"[bn] BatchNorm statistics re-estimated on {n} batches of unlabelled frames")
        elif cfg.init == "public":
            raise NotImplementedError("public weights: loaders come with the step-1 evaluation")
        else:
            raise ValueError(f"unknown init {cfg.init!r}")

    # ------------------------------------------------------------------ checkpoints
    def _state(self):
        return {"model": self.model.state_dict(),
                "criterion": self.criterion.state_dict() if self.cfg.task == "pretrain" else None,
                "opt": self.opt.state_dict(), "sched": self.sched.state_dict(), "scaler": self.scaler.state_dict(),
                "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch, "step": self.step,
                "sums": self.sums, "n": self.n,
                "rng": _rng_state(), "config": dataclasses.asdict(self.cfg)}

    def _resume(self):
        s = torch.load(self.last_ckpt, map_location="cpu", weights_only=False)
        new = dataclasses.asdict(self.cfg)
        diff = sorted(k for k in new if k not in RESUME_FREE and new[k] != s["config"].get(k, new[k]))
        if diff:
            raise ValueError(f"{self.last_ckpt}: config differs from the run being resumed in {diff}; "
                             "use another run name, or delete the run directory to start over")
        self.model.load_state_dict(s["model"])
        if s["criterion"] is not None:
            self.criterion.load_state_dict(s["criterion"])
        self.opt.load_state_dict(s["opt"])
        self.sched.load_state_dict(s["sched"])
        self.scaler.load_state_dict(s["scaler"])
        self.epoch, self.batch_in_epoch, self.step = s["epoch"], s["batch_in_epoch"], s["step"]
        self.sums, self.n = s["sums"], s["n"]
        _set_rng_state(s["rng"])
        print(f"[resume] {self.last_ckpt}: epoch {self.epoch}, batch {self.batch_in_epoch}, step {self.step}")

    # ------------------------------------------------------------------ training
    def _loss(self, batch):
        if self.cfg.task == "pretrain":
            out = lejepa_step(self.model.encode, self.criterion, batch.to(self.device, non_blocking=True))
            return out["loss"], {k: v.item() for k, v in out.items()}
        rgb, gt = (t.to(self.device, non_blocking=True) for t in batch)
        with torch.no_grad():
            feats = self.model.encode(rgb)          # frozen encoder
        out = self.model.decode(feats, rgb.shape[-2:])
        pred = self._depth(out, gt.shape[-2:]).clamp(MIN_DEPTH, self.cap)
        valid = (gt > MIN_DEPTH) & (gt < self.cap)
        loss = silog_loss(pred, gt, valid)
        return loss, {"loss": loss.item()}

    def _depth(self, out, hw):
        """Decoder output -> depth in metres at the ground-truth size ``hw``."""
        pred = F.interpolate(out.float(), size=tuple(hw), mode="bilinear", align_corners=False)
        return self.model.to_depth(pred, max_depth=self.cap)

    def fit(self):
        cfg = self.cfg
        bpe = math.ceil(len(self.train_set) / cfg.batch_size)
        if cfg.max_steps_per_epoch:
            bpe = min(bpe, cfg.max_steps_per_epoch)
        print(f"[run] {cfg.name}: {cfg.task} {cfg.model} on {cfg.dataset} | device {self.device} "
              f"precision {self.precision} | {len(self.train_set)} samples, {bpe} batches/epoch, {self.epochs} epochs")
        t_ckpt = time.time()
        while self.epoch < self.epochs:
            self.sampler.set(self.epoch, self.batch_in_epoch * cfg.batch_size)
            loader = DataLoader(self.train_set, batch_size=cfg.batch_size, sampler=self.sampler,
                                num_workers=cfg.num_workers, pin_memory=self.device.type == "cuda",
                                drop_last=True, worker_init_fn=seed_worker, persistent_workers=False)
            self.model.train()
            t0 = time.time()
            for batch in loader:
                if self.batch_in_epoch >= bpe:
                    break
                with autocast(self.device, self.precision):
                    loss, logs = self._loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {self.epoch}, step {self.step}: {logs}")
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.step += 1
                self.batch_in_epoch += 1
                self.n += 1
                for k, v in logs.items():
                    self.sums[k] = self.sums.get(k, 0.0) + v
                if self.step % cfg.log_every == 0:
                    print(f"[train] epoch {self.epoch} batch {self.batch_in_epoch}/{bpe} " +
                          " ".join(f"{k} {v:.4f}" for k, v in logs.items()))
                if (time.time() - t_ckpt) / 60 >= cfg.ckpt_minutes:
                    _save_atomic(self._state(), self.last_ckpt)
                    t_ckpt = time.time()
            row = {"epoch": self.epoch, "step": self.step, "lr": self.opt.param_groups[0]["lr"],
                   "seconds": round(time.time() - t0, 1), **{f"train_{k}": v / max(self.n, 1) for k, v in self.sums.items()}}
            if self.val_set is not None:       # same columns every epoch: empty when the epoch is not validated
                run_val = (self.epoch + 1) % cfg.val_every == 0 or self.epoch + 1 == self.epochs
                val = self.evaluate(self.val_set, ("median",))["median"] if run_val else dict.fromkeys(METRICS, "")
                row.update({f"val_{k}": v for k, v in val.items()})
            self.sched.step()
            self.epoch, self.batch_in_epoch = self.epoch + 1, 0
            self.sums, self.n = {}, 0
            append_csv_row(row, self.dir / "metrics.csv")
            print("[epoch] " + " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
            _save_atomic(self._state(), self.last_ckpt)
            t_ckpt = time.time()
        if self.cfg.task == "pretrain":
            torch.save(self.model.encoder.state_dict(), self.dir / "encoder.pt")
        torch.save(self.model.state_dict(), self.dir / "model.pt")
        print(f"[done] {self.dir}")
        return self.dir

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, dataset: Dataset, alignments=("none", "median", "scale_shift")) -> dict:
        self.model.eval()
        per = {a: [] for a in alignments}
        for i in range(len(dataset)):
            rgb, gt = dataset[i]
            with autocast(self.device, self.precision):
                out = self.model.decode(self.model.encode(rgb[None].to(self.device)), rgb.shape[-2:])
            pred = self._depth(out, gt.shape[-2:])[0, 0].cpu()
            for a in alignments:
                per[a].append(depth_metrics(pred, gt[0], self.cfg.dataset, a))
        self.model.train()
        return {a: mean_metrics(v) for a, v in per.items()}


def evaluate_run(cfg: RunConfig, test_list: str, weights: str | None = None) -> dict:
    """Evaluate a fine-tuned run (``model.pt`` in its run directory, or ``weights``) on a test list."""
    tr = Trainer.__new__(Trainer)
    tr.cfg = cfg
    tr.device = pick_device(cfg.device)
    tr.precision = precision_for(tr.device)
    tr.cap = CAPS[cfg.dataset]
    tr.model = build_model(cfg.model, cfg.variant).to(tr.device)
    run_dir = paths.runs_root() / cfg.name
    tr.model.load_state_dict(torch.load(weights or run_dir / "model.pt", map_location="cpu", weights_only=True))
    test = DepthPairs(_resolve(test_list), paths.data_root(), cfg.image_size, DEPTH_SCALE[cfg.dataset])
    res = tr.evaluate(test)
    save_json({"test_list": str(test_list), "n": len(test), **res}, run_dir / f"test_metrics_{Path(test_list).stem}.json")
    return res
