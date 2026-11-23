"""Device and numeric-precision policy shared by every entry point.

The same code runs on the ALCOR Tower GTX (GTX 1080 / 1080 Ti, Pascal, sm_61) and on a local
RTX 5070 (Blackwell, sm_120):

* Pascal has no tensor cores, runs fp16 at a small fraction of its fp32 throughput and has no
  bf16, so it trains in **fp32**.
* Newer GPUs train with **fp16 autocast** (``fp16-mixed``).
* **bf16 is never used**: in phase 1 it made SIGReg diverge (NaN).
"""
from __future__ import annotations

import contextlib

import torch

PASCAL_OR_OLDER = (6, 9)  # compute capability below 7.0 -> no tensor cores


def pick_device(name: str = "auto") -> torch.device:
    """``auto`` picks CUDA when available, else CPU. ``cuda:1`` etc. are passed through."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def precision_for(device: torch.device) -> str:
    """Return ``"fp32"`` or ``"fp16-mixed"`` for the given device."""
    if device.type != "cuda":
        return "fp32"
    major, minor = torch.cuda.get_device_capability(device)
    return "fp32" if (major, minor) <= PASCAL_OR_OLDER else "fp16-mixed"


def autocast(device: torch.device, precision: str):
    """Autocast context for the chosen precision (a no-op in fp32)."""
    if precision == "fp16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision != "fp32":
        raise ValueError(f"unsupported precision {precision!r} (bf16 is not allowed with SIGReg)")
    return contextlib.nullcontext()


def describe() -> str:
    """One line per visible GPU: index, name, compute capability, memory, precision policy."""
    lines = [f"torch {torch.__version__} | CUDA build {torch.version.cuda} | "
             f"cuda available: {torch.cuda.is_available()}"]
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        dev = torch.device("cuda", i)
        lines.append(f"  cuda:{i} {p.name} | sm_{p.major}{p.minor} | {p.total_memory / 2**30:.1f} GiB | "
                     f"precision {precision_for(dev)}")
    return "\n".join(lines)
