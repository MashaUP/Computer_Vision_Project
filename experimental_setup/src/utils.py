import csv
import json
import math
import random
import time
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.config import MODEL_SAVE_PATH

# ==========================================
# Time
# ==========================================
@dataclass
class Timer:
    # Simple wall-clock timer for training and evaluation loops
    start_time: float = time.time()

    def reset(self) -> None:
        self.start_time = time.time()

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def end(self) -> str:
        elapsed = self.elapsed_seconds()
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins}m {secs}s"

# ==========================================
# Checkpoints
# ==========================================
def _checkpoint_dir() -> Path:
    path = Path(MODEL_SAVE_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    filename: str = "checkpoint.pt",
    extra: Optional[Dict[str, Any]] = None
) -> Path:
    ckpt_dir = _checkpoint_dir()
    filepath = ckpt_dir / filename
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    state: Dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        state["extra"] = extra
    torch.save(state, tmp_path)
    tmp_path.replace(filepath)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Checkpoint saved: {filepath}")
    print(f"[INFO] Epoch: {epoch} | Params: {total_params:,}")
    return filepath

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    filename: str = "checkpoint.pt",
    map_location: Optional[str] = None,
    strict: bool = True
) -> Tuple[int, Dict[str, Any]]:
    ckpt_path = _checkpoint_dir() / filename
    if not ckpt_path.exists():
        print(f"[WARNING] Checkpoint not found: {ckpt_path}")
        return 0, {}
    if map_location is None:
        map_location = "cpu"
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    msg = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict
    )
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    missing = getattr(msg, "missing_keys", [])
    unexpected = getattr(msg, "unexpected_keys", [])
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    print(f"[INFO] Resuming from epoch: {start_epoch}")
    if not strict:
        print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    extra = checkpoint.get("extra", {})
    return start_epoch, extra

# ==========================================
# CSV metrics
# ==========================================
def log_metrics_to_csv(
    filename: str,
    epoch: int,
    metrics: Dict[str, Any],
    root: Optional[Path] = None
) -> Path:
    if root is None:
        root = Path(MODEL_SAVE_PATH).parent
    root.mkdir(parents=True, exist_ok=True)
    filepath = root / filename
    file_exists = filepath.exists()
    row = {"epoch": int(epoch)}
    row.update(metrics)
    fieldnames = list(row.keys())
    with filepath.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    if epoch == 0 or epoch % 5 == 0:
        keys = ", ".join(row.keys())
        print(f"[INFO] Logged metrics to {filepath} ({keys})")
    return filepath

# ==========================================
# JSON Helper
# ==========================================
def save_dict_to_json(data_dict: Dict[str, Any], file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_dict = {k: (v.item() if hasattr(v, 'item') else v) for k, v in data_dict.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable_dict, f, indent=4)
    print(f"[INFO] Dictionary successfully saved to JSON: {path}")

def load_dict_from_json(file_path: str | Path) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        print(f"[WARNING] File not found: {path}")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==========================================
# Reproducibility
# ==========================================
def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# Checkpoint discovery
# ==========================================
def find_checkpoint(ckpt_subdir: str, dataset: str, arch: str) -> Path | None:
    """
    Return the most appropriate checkpoint for (dataset, arch) in ckpt_subdir.

    Search priority:
      1. checkpoint filename contains dataset, architecture, and best;
      2. any best-like checkpoint;
      3. most recent checkpoint.
    """
    folder = Path(MODEL_SAVE_PATH) / ckpt_subdir
    if not folder.exists():
        return None

    all_ckpts = list(folder.glob("*.ckpt"))
    if not all_ckpts:
        return None

    def score(path: Path) -> tuple[bool, bool, bool, float]:
        name = path.stem.lower()
        return (
            dataset.lower() in name,
            arch.lower() in name,
            "best" in name,
            path.stat().st_mtime,
        )

    all_ckpts.sort(key=score, reverse=True)
    return all_ckpts[0]


def find_ssl_ckpt(dataset: str) -> Path | None:
    """Find the SSL checkpoint for a dataset using current and legacy paths."""
    candidates = [
        Path(MODEL_SAVE_PATH) / "pretrain" / f"last_{dataset}.ckpt",
        Path(MODEL_SAVE_PATH) / "pretrain" / "best.ckpt",
        Path(MODEL_SAVE_PATH) / "ssl" / "last.ckpt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    folder = Path(MODEL_SAVE_PATH) / f"ssl_{dataset}"
    if folder.exists():
        best_files = sorted(
            folder.glob("best-*.ckpt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if best_files:
            return best_files[0]
    return None

####################################################
# METER-specific utilities
####################################################
def load_meter_pretrained_weights(model, weight_path, device="cpu"):
    if not Path(weight_path).exists():
        print(f"[WARNING] Checkpoint not found: {weight_path}")
        return model
    ckpt = torch.load(weight_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    clean_dict = {}
    model_state = model.state_dict() 
    skipped_shapes = []
    for k, v in state_dict.items():
        # Remove optional 'model.' prefix but keep 'encoder.' or 'decoder.'
        new_key = k.replace("model.", "")
        # Fix decoder nesting issue from original checkpoint
        if new_key.startswith("decoder.decoder."):
            new_key = new_key.replace("decoder.decoder.", "decoder.")
        
        # Special handling for MobileViT weights (encoder)
        if "depthwise.weight" in new_key and new_key in model_state:
            target_shape = model_state[new_key].shape
            if v.shape != target_shape:
                if len(target_shape) > 1 and target_shape[1] == 1:
                    if v.shape[0] == v.shape[1]:
                        indices = torch.arange(v.shape[0])
                        v = v[indices, indices].unsqueeze(1)
                    elif v.dim() == 4 and len(target_shape) == 4:
                        if v.numel() == math.prod(target_shape):
                            v = v.reshape(target_shape)
        
        if "mv2" in new_key and "conv" in new_key and "weight" in new_key and new_key in model_state:
             target_shape = model_state[new_key].shape
             if v.shape != target_shape and v.numel() == math.prod(target_shape):
                 v = v.reshape(target_shape)

        # Generic shape check
        if new_key in model_state:
            if v.shape != model_state[new_key].shape:
                skipped_shapes.append(f"{new_key} (expected {model_state[new_key].shape}, got {v.shape})")
                continue
            clean_dict[new_key] = v
        else:
            # Optional: add anyway to see what's missing in load_state_dict via msg.unexpected_keys
            clean_dict[new_key] = v

    msg = model.load_state_dict(clean_dict, strict=False)
    print(f"[LOAD] {Path(weight_path).name}")
    if skipped_shapes:
        print(f"       Skipped (shape mismatch): {len(skipped_shapes)} layers")
        for s in skipped_shapes:
            print(f"         - {s}")
    print(f"       Missing (randomly initialized): {len(msg.missing_keys)} layers")
    if msg.missing_keys:
        print(f"       [DEBUG] Missing keys: {msg.missing_keys[:10]} {'...' if len(msg.missing_keys) > 10 else ''}")
    print(f"       Unexpected (ignored): {len(msg.unexpected_keys)} layers")
    if msg.unexpected_keys:
        print(f"       [DEBUG] Unexpected keys: {msg.unexpected_keys[:10]} {'...' if len(msg.unexpected_keys) > 10 else ''}")
    
    return model
