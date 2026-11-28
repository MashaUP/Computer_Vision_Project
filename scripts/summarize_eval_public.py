"""Collect the ``main.py eval-public`` JSON files of a folder into one CSV (one row per run and alignment).

    python scripts/summarize_eval_public.py results/step2_public            # writes results/step2_public/summary.csv

Numbers are copied from the JSON files as they are; nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
ALIGNMENTS = ("none", "median", "scale_shift")
FIELDS = ("model", "dataset", "alignment", "n", "image_h", "image_w", "weights", "test_list", "source", *METRICS)


def rows_of(path: Path) -> list[dict]:
    res = json.loads(path.read_text())
    h, w = res["image_size"]
    rows = []
    for alignment in ALIGNMENTS:
        if alignment not in res:
            continue
        rows.append({
            "model": res["model"], "dataset": res["dataset"], "alignment": alignment, "n": res["n"],
            "image_h": h, "image_w": w, "weights": res["weights"], "test_list": res["test_list"],
            "source": path.name, **{m: res[alignment][m] for m in METRICS},
        })
    return rows


def summarize(folder: Path, out: Path | None = None) -> Path:
    files = sorted(p for p in folder.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no JSON file in {folder}")
    rows = [r for p in files for r in rows_of(p)]
    rows.sort(key=lambda r: (r["dataset"], r["model"], ALIGNMENTS.index(r["alignment"])))
    out = out or folder / "summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder with the eval-public JSON files")
    ap.add_argument("--out", type=Path, default=None, help="default <folder>/summary.csv")
    args = ap.parse_args(argv)
    print(f"wrote {summarize(args.folder, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
