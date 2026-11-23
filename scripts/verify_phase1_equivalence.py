"""Check that the unified LeJEPA code reproduces the phase-1 course code exactly.

For classic (root setup, ``src/train.py``) and dense (``experimental_setup/src/train.py``) it builds
the phase-1 ``LeJEPAMeter`` and the new ``METER`` + ``LeJEPALoss`` with the ``*_phase1`` recipe,
copies the phase-1 weights into the new modules, and compares, for the same seed and images:
the views, the loss and its terms, and the encoder gradients.

    git clone https://github.com/Zunza5/lejepa-monocular-depth-estimation /tmp/phase1    # the exam version (original repository)
    python scripts/verify_phase1_equivalence.py --phase1 /tmp/phase1

Needs the phase-1 dependencies (lightning, h5py, opencv, scipy). CPU only.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.views import make_views  # noqa: E402
from models.baselines import METER  # noqa: E402
from training.lejepa import LeJEPALoss, load_recipe  # noqa: E402

NORMALISE = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def import_phase1(base: Path):
    """Import ``src.train`` from a phase-1 setup folder, then hide its ``src`` package again."""
    for m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[m]
    sys.path.insert(0, str(base))
    try:
        train = importlib.import_module("src.train")
    finally:
        sys.path.remove(str(base))
        for m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
            sys.modules[f"_phase1_{base.name}.{m}"] = sys.modules.pop(m)
    return train


def report(name: str, a: torch.Tensor, b: torch.Tensor, tol: float) -> bool:
    diff = (a.detach().double() - b.detach().double()).abs().max().item()
    ok = diff <= tol
    print(f"    {name:28s} max |diff| = {diff:.3e}  {'OK' if ok else 'MISMATCH'}")
    return ok


def grad_vector(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for p in module.parameters() if p.grad is not None])


def classic(phase1: Path, dataset: str, seed: int, tol: float) -> bool:
    train = import_phase1(phase1)
    old = train.LeJEPAMeter(arch_type="xxs", proj_dim=64, num_slices=256, lamb=0.05, dataset=dataset).train()
    new = METER("xxs").train()
    new.encoder.load_state_dict(old.encoder.state_dict())
    crit = LeJEPALoss(new.deepest_channels(), load_recipe("classic_phase1", dataset=dataset)).train()
    crit.projector.mlp.load_state_dict(old.projector.state_dict())
    x = torch.rand(4, 3, 288, 384)

    torch.manual_seed(seed)                     # phase-1 views: 2 global then 6 local, per image
    g_old = [torch.stack([old.global_aug(im) for im in x]) for _ in range(2)]
    l_old = [torch.stack([old.local_aug(im) for im in x]) for _ in range(6)]
    torch.manual_seed(seed)
    g_new, l_new = make_views(x, crit.recipe)
    ok = all(report(f"view {i}", NORMALISE(a), b, 1e-5) for i, (a, b) in enumerate(zip(g_new + l_new, g_old + l_old)))

    proj = torch.cat([torch.stack([old(v)[1] for v in g_old]), torch.stack([old(v)[1] for v in l_old])])
    loss_old, inv_old, sig_old = old._lejepa_loss(proj)
    out = crit([new.encode(v)[-1] for v in g_new], [new.encode(v)[-1] for v in l_new])
    ok &= report("loss", out["loss"], loss_old, tol)
    ok &= report("invariance", out["inv"], inv_old, tol)
    ok &= report("SIGReg", out["sigreg"], sig_old, tol)
    loss_old.backward()
    out["loss"].backward()
    ok &= report("encoder gradients", grad_vector(new.encoder), grad_vector(old.encoder), tol)
    ok &= report("projector gradients", grad_vector(crit.projector), grad_vector(old.projector), tol)
    return ok


def dense(phase1: Path, seed: int, tol: float) -> bool:
    train = import_phase1(phase1 / "experimental_setup")
    old = train.LeJEPAMeter(arch_type="xxs", proj_dim=64, num_slices=256, lamb=0.02, alpha=0.5, lr=5e-3).train()
    new = METER("xxs").train()
    new.encoder.load_state_dict(old.encoder.state_dict())
    crit = LeJEPALoss(new.deepest_channels(), load_recipe("dense_phase1")).train()
    crit.projector.mlp.load_state_dict(old.projector.state_dict())
    x = torch.rand(4, 3, 288, 384)

    torch.manual_seed(seed)                     # phase-1 order: global crop, 2 colours, then 3 local crops
    gc = old.global_spatial(x)
    views_old = [old.global_color(gc), old.global_color(gc)]
    for _ in range(3):
        lc = old.local_spatial(x)
        views_old += [old.local_color(lc), old.local_color(lc)]
    torch.manual_seed(seed)
    g_new, l_new = make_views(x, crit.recipe)
    ok = all(report(f"view {i}", NORMALISE(a), b, 1e-5) for i, (a, b) in enumerate(zip(g_new + l_new, views_old)))

    b = x.shape[0]
    _, gp = old(torch.cat(views_old[:2]))
    _, lp = old(torch.cat(views_old[2:]))
    proj = torch.cat([gp.reshape(2, b, *gp.shape[1:]), lp.reshape(6, b, *lp.shape[1:])])
    loss_old, dense_old, global_old, sig_old = old.lejepa_loss(proj)
    deep_g = list(torch.split(new.encode(torch.cat(g_new))[-1], b))
    deep_l = list(torch.split(new.encode(torch.cat(l_new))[-1], b))
    out = crit(deep_g, deep_l)
    ok &= report("loss", out["loss"], loss_old, tol)
    ok &= report("dense invariance", out["dense_inv"], dense_old, tol)
    ok &= report("global invariance", out["global_inv"], global_old, tol)
    ok &= report("SIGReg", out["sigreg"], sig_old, tol)
    loss_old.backward()
    out["loss"].backward()
    ok &= report("encoder gradients", grad_vector(new.encoder), grad_vector(old.encoder), tol)
    ok &= report("projector gradients", grad_vector(crit.projector), grad_vector(old.projector), tol)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1", type=Path, required=True, help="checkout of the exam version (original repository)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()
    results = {}
    for ds in ("kitti", "nyu"):
        print(f"== classic ({ds} crop scales)")
        results[f"classic/{ds}"] = classic(args.phase1, ds, args.seed, args.tol)
    print("== dense")
    results["dense"] = dense(args.phase1, args.seed, args.tol)
    print("\n" + "\n".join(f"{k:14s} {'IDENTICAL' if v else 'DIFFERENT'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
