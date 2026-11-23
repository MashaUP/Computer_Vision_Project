"""Tests run on CPU (never on the lab GPUs): ``python -m pytest -q``."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def images():
    """A small batch of random RGB images in [0, 1]."""
    return torch.rand(2, 3, 256, 320)
