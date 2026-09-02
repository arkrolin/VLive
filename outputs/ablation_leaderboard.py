"""Summarise every ablation run into one leaderboard.

Reads ``outputs/train_runs/<run>/metrics.csv`` produced by the trainer and prints
a table sorted by the primary criterion: best val ``abs_mae`` against the exem
prior measured on the *same* shared-action val subset.

Usage
-----
    .venv/bin/python outputs/ablation_leaderboard.py            # all runs
    .venv/bin/python outputs/ablation_leaderboard.py --scale 285

Notes
-----
* ``abs_mae`` is only comparable **within the same data scale**: the exem prior
  itself is 0.7379 at subset=60 but 2.4405 at subset=285 (3.3x harder task),
  because the val set is larger and more diverse.
* Runs are bucketed by scale automatically from the sample count recorded in
  the run log when available; otherwise use --scale to filter.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "outputs" / "train_runs"

# scale -> exem prior abs_mae on that scale's shared val subset
KNOWN_SCALE = {
    "samples=761": 60,      # subset=60  -> 761 train samples
    "samples=6626": 285,    # subset=285 -> 6626 train samples
}
EXEM_PRIOR = {60: 0.7379, 285: 2.4405}


def detect_scale(run_dir: Path) -> int | None:
    """Infer the data scale from the run log header (train samples=...)."""
    log = run_dir.parent / f"{run_dir.name.replace('_reg45', '').replace('_ep60', '')}.log"
    # try a few plausible log names
    cands = [run_dir.parent / f"{run_dir.name}.log"]
    stem = re.sub(r"^abl_", "abl_", run_dir.name)
    first = stem.split("_")[1] if "_" in stem else stem
    cands.append(run_dir.parent / f"abl_{first}.log")
    for c in cands:
        if not c.exists():
            continue
        try:
            head = c.read_text(encoding="utf-8", errors="ignore")[:200_000]
        except OSError:
            continue
        m = re.search(r"train samples=(\d+)", head)
        if m:
            for key, scale in KNOWN_SCALE.items():
                if key in head:
                    return scale
            return 285 if int(m.group(1)) > 3000 else 60
    return None


def best_row(rows: list[dict]) -> dict | None:
    """Row with the lowest finite val_abs_mae."""
    ok = [r for r in rows if r["abs"] == r["abs"]]  # drop NaN
    return min(ok, key=lambda r: r["abs"]) if ok else None


def load_run(run_dir: Path) -> dict | None:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        return None
    rows = []
    with csv_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "epoch": int(r["epoch"]),
                    "train": float(r["train_loss"]),
                    "val": float(r["val_loss"]),
                    "abs": float(r["val_abs_mae"]),
                    "rel_f": float(r["val_rel_mae_f"]),
                    "exem_abs": float(r["exem_abs_mae"]),
                    "exem_rel_f": float(r["exem_rel_mae_f"]),
                })
            except (TypeError, ValueError, KeyError):
                continue
    if not rows:
        return None
    best = best_row(rows)
    finished = "GATE:" in (run_dir.name, "")  # placeholder, set below
    log = run_dir.parent / f"abl_{run_dir.name.split('_')[1]}.log"
    done = False
    if log.exists():
        try:
            done = "TRAINING GATE" in log.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
    return {
        "name": run_dir.name,
        "scale": detect_scale(run_dir),
        "epochs_run": max(r["epoch"] for r in rows),
        "done": done,
        "best": best,
        "last": rows[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=None,
                    help="only show runs at this data scale (60 or 285)")
    args = ap.parse_args()

    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith(("abl_", "x0_")):
            continue
        info = load_run(d)
        if info and info["best"] is not None:
            runs.append(info)
    if not runs:
        print("no runs with metrics.csv found")
        return 1

    if args.scale:
        runs = [r for r in runs if r["scale"] == args.scale or r["scale"] is None]

    # group by scale
    by_scale: dict[int | None, list[dict]] = {}
    for r in runs:
        by_scale.setdefault(r["scale"], []).append(r)

    for scale in sorted(by_scale, key=lambda s: (s is None, s)):
        group = by_scale[scale]
        prior = EXEM_PRIOR.get(scale) or group[0]["best"]["exem_abs"]
        print(f"\n=== data scale: {scale} | exem prior abs_mae = {prior:.4f} ===")
        print(f"{'run':<26}{'ep':>4}{'abs_mae':>10}{'vs exem':>10}"
              f"{'rel_f':>9}{'train':>9}{'status':>10}")
        print("-" * 78)
        for r in sorted(group, key=lambda x: x["best"]["abs"]):
            b = r["best"]
            gain = (prior - b["abs"]) / prior * 100 if prior else float("nan")
            mark = "" if b["abs"] < prior else "  <-- worse than prior"
            print(f"{r['name']:<26}{b['epoch']:>4}{b['abs']:>10.4f}"
                  f"{gain:>9.1f}%{b['rel_f']:>9.4f}{b['train']:>9.4f}"
                  f"{('done' if r['done'] else 'running'):>10}{mark}")
    print("\nNote: abs_mae is NOT comparable across scales (different exem prior).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
