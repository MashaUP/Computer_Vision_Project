"""Small JSON / CSV helpers for results (final tables go to ``results/``, run outputs to the runs root)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float))
    return path


def append_csv_row(row: Mapping[str, Any], path: str | Path) -> Path:
    """Append one row, writing the header the first time. Columns must stay the same."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    return path
