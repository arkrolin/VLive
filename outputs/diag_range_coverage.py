"""How much of the val set falls back to the GLOBAL range, and would a
full-corpus range computation fix it?

Background
----------
``mean_range_vecs`` seeds every lo/hi with the GLOBAL range and only overwrites
params present in ``per_lo``/``per_hi``. Those dicts are built by
``compute_range(train_ds, n=400)`` - i.e. from just ~6% of the 6626 train
samples. Any param that does not appear in those 400 samples silently gets
span = g_hi - g_lo (= 1130.2 here) instead of its own range.

Why it matters
--------------
``abs_mae = e_f * span`` equals the TRUE raw-unit error ``|pred_raw-target_raw|``
regardless of which span was used to normalise, so the 4.16 raw units of error
the model shows on those instances is real, not a metric illusion. But it is an
artefact of the NORMALISATION: with span 1130 the model only has to be accurate
to ~0.004 in normalised units, so it never learns to be precise on them. If
those params really have a small range, normalising correctly would make them
near-trivially predictable - and would stop them from dominating abs_mae
(they are 15.1% of instances but 49.6% of the model's total abs_mae).

Usage
-----
    .venv/bin/python outputs/diag_range_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))

from config import PipelineConfig  # noqa: E402
from dataset import Live2DDataset  # noqa: E402
from train import mean_range_vecs  # noqa: E402


def per_param_range(ds, n=None):
    """Same statistic as compute_range, but returns only the per-param dicts."""
    per_lo, per_hi = {}, {}
    total = len(ds) if n is None else min(n, len(ds))
    for i in range(total):
        try:
            s = ds[i]
        except Exception:
            continue
        x = np.asarray(s["target"], dtype=np.float32)
        names = s["names"]
        for j, nm in enumerate(names):
            if j >= x.shape[0]:
                break
            col = x[j].astype(np.float32)
            lo, hi = float(col.min()), float(col.max())
            if nm not in per_lo:
                per_lo[nm], per_hi[nm] = lo, hi
            else:
                per_lo[nm] = min(per_lo[nm], lo)
                per_hi[nm] = max(per_hi[nm], hi)
    return per_lo, per_hi


def val_param_names(cfg) -> list[str]:
    val_ds = Live2DDataset(cfg, split="val")
    seen: list[str] = []
    got: set[str] = set()
    for i in range(len(val_ds)):
        for nm in val_ds[i]["names"]:
            if nm not in got:
                got.add(nm)
                seen.append(nm)
    return seen


def main() -> int:
    cfg = PipelineConfig()
    cfg.subset_models = 285
    cfg.action_cond = "id"
    cfg.eval_shared_min_models = 5

    print("building train corpus (this is a full pass, please wait) ...")
    train_ds = Live2DDataset(cfg, split="train")
    print(f"  train samples: {len(train_ds)}")

    names = val_param_names(cfg)
    print(f"  distinct val param names: {len(names)}")

    for n in (400, None):
        per_lo, per_hi = per_param_range(train_ds, n)
        known = [nm for nm in names if nm in per_lo]
        span_full = np.array([per_hi[nm] - per_lo[nm] for nm in known],
                             dtype=np.float64)
        label = f"n={n}" if n else "n=ALL"
        print(f"\n=== compute_range({label}) ===")
        print(f"  params with a real range : {len(per_lo)}")
        print(f"  val params covered       : {len(known)}/{len(names)} "
              f"({len(known) / len(names) * 100:.1f}%)")
        print(f"  val params MISSING       : {len(names) - len(known)} "
              f"({(len(names) - len(known)) / len(names) * 100:.1f}%)")
        if len(span_full):
            print(f"  span distribution of covered params: "
                  f"min {span_full.min():.3f}  median {np.median(span_full):.3f}  "
                  f"mean {span_full.mean():.3f}  max {span_full.max():.3f}")
            print(f"  share with span < 1.0    : "
                  f"{(span_full < 1.0).mean() * 100:.1f}%")
            print(f"  share with span < 22.6   : "
                  f"{(span_full < 22.6).mean() * 100:.1f}%")

    # what range do the currently-fallback val params ACTUALLY have?
    per_lo_all, per_hi_all = per_param_range(train_ds, None)
    per_lo_400, _ = per_param_range(train_ds, 400)
    recovered = [nm for nm in names
                 if nm not in per_lo_400 and nm in per_lo_all]
    if recovered:
        sp = np.array([per_hi_all[nm] - per_lo_all[nm] for nm in recovered])
        print(f"\n=== params that n=400 misses but n=ALL covers: {len(recovered)} ===")
        print(f"  their TRUE span: min {sp.min():.4f}  median {np.median(sp):.4f}  "
              f"mean {sp.mean():.4f}  max {sp.max():.4f}")
        print(f"  share with true span < 1.0 : {(sp < 1.0).mean() * 100:.1f}%")
        print(f"  share with true span < 22.6: {(sp < 22.6).mean() * 100:.1f}%")
        print("  -> if these are small, the 1130-wide fallback was inflating them "
              "by orders\n     of magnitude, and a full-corpus range fixes both the "
              "metric and the\n     model's normalisation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
