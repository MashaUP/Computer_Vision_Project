"""Where data, runs and caches live, read from environment variables.

Inside the lab container ``scripts/alcor/run.sh`` mounts
``/mnt/hdd2/<user>/{data,runs,cache}`` as ``/work/{data,runs,cache}`` and sets the variables
below. Locally, export them (or rely on the defaults next to the repository).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _root(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, default)).expanduser()


def data_root() -> Path:
    """Datasets (never inside the repository): ``LG_DATA_ROOT``."""
    return _root("LG_DATA_ROOT", REPO_ROOT.parent / "data")


def runs_root() -> Path:
    """Checkpoints, logs and per-run outputs: ``LG_RUNS_ROOT``."""
    return _root("LG_RUNS_ROOT", REPO_ROOT.parent / "runs")


def cache_root() -> Path:
    """pip / torch hub / matplotlib caches: ``LG_CACHE_ROOT``."""
    return _root("LG_CACHE_ROOT", REPO_ROOT.parent / "cache")
