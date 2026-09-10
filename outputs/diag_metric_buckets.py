"""Why is the trained model (rel_mae~0.070) WORSE than its own exem prior (0.049)?

Hypothesis: the training LOSS floors the per-param span at 2% of the global
range, which downweights near-constant params; the reported rel_mae METRIC uses
the RAW per-param range (floor 1e-3), so those same near-constant params dominate
the metric. Training therefore optimises a different quantity than it reports.

This script buckets params by range magnitude and shows how much of the
rel_mae comes from near-constant params, for the exem prior.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(".")
sys.path.insert(0, "src/live2d_vla")
sys.path.insert(0, ".")

from schema import load_whitelist, load_gen_mask          # noqa: E402
from io_motion import list_motions, load_target_curves    # noqa: E402

T, FPS, SEED = 48, 30.0, 1234


def main(subset_n=60, val_frac=0.12):
    wl = json.loads((ROOT / "outputs" / "model_whitelist.json").read_bytes()).get("kept", [])
    gm = json.loads((ROOT / "outputs" / "gen_mask.json").read_bytes()).get("models", {})
    subset = wl[:subset_n]
    targets = {m: gm[m]["target"] for m in subset if m in gm}

    cache = {}
    for m in subset:
        if m not in targets:
            continue
        for a in list_motions(ROOT / "standrad-live-2d" / m):
            c = load_target_curves(ROOT / "standrad-live-2d" / m, a, targets[m],
                                   fps=FPS, T=T)
            if c:
                cache[(m, a)] = {p: np.asarray(v, np.float32).ravel() for p, v in c.items()}

    rng = random.Random(SEED)
    holdout = set(rng.sample(subset, max(1, int(round(len(subset) * val_frac)))))
    train_s = [s for s in cache if s[0] not in holdout]
    val_s = [s for s in cache if s[0] in holdout]

    per_lo, per_hi = {}, {}
    for s in train_s:
        for p, a in cache[s].items():
            lo, hi = float(a.min()), float(a.max())
            if p not in per_lo:
                per_lo[p], per_hi[p] = lo, hi
            else:
                per_lo[p] = min(per_lo[p], lo)
                per_hi[p] = max(per_hi[p], hi)
    g_lo, g_hi = min(per_lo.values()), max(per_hi.values())
    g_range = g_hi - g_lo

    # exem prior = (action, param) mean over ALL subset models (as in training)
    acc = {}
    for s in cache:
        for p, c in cache[s].items():
            acc.setdefault((s[1], p), []).append(c)
    am = {k: np.mean(np.stack(v, 0), 0) for k, v in acc.items()}

    min_span_loss = max(g_range, 1.0) * 0.02      # flooring used by the LOSS

    buckets = [("large  (>10% glob)", 0.10, 1e9),
               ("mid    (2-10% glob)", 0.02, 0.10),
               ("near-const (<2% glob)", -1.0, 0.02)]
    agg = {b[0]: [0.0, 0, 0.0, 0.0] for b in buckets}   # relsum, cnt, loss-sum, abs-sum

    for s in val_s:
        for p, tgt in cache[s].items():
            pp = am.get((s[1], p))
            if pp is None:
                continue
            lo = per_lo.get(p, g_lo)
            hi = per_hi.get(p, g_hi)
            rng_p = hi - lo
            e = float(np.abs(np.asarray(pp, np.float32) - tgt).mean())
            frac = rng_p / g_range if g_range > 0 else 0.0
            name = next(b[0] for b in buckets if b[1] <= frac < b[2])
            a = agg[name]
            a[0] += min(e / max(rng_p, 1e-3), 2.0)      # raw-span metric
            a[1] += 1
            a[2] += (e / max(rng_p, min_span_loss)) ** 2  # floored-span loss (MSE)
            a[3] += e                                     # absolute error

    tot_rel, tot_cnt, tot_loss, tot_abs = 0.0, 0, 0.0, 0.0
    for b in buckets:
        tot_rel += agg[b[0]][0]; tot_cnt += agg[b[0]][1]
        tot_loss += agg[b[0]][2]; tot_abs += agg[b[0]][3]

    print(f"\nglobal range = {g_range:.3f}   loss span floor = {min_span_loss:.3f}")
    print(f"val samples={len(val_s)}  param-instances={tot_cnt}\n")
    print(f"  {'bucket':26s} {'share':>7s} {'rel_mae':>9s} {'%of total rel':>14s} "
          f"{'loss(MSE)':>10s} {'abs_mae':>9s}")
    for b in buckets:
        rel, cnt, ls, ab = agg[b[0]]
        print(f"  {b[0]:26s} {cnt/max(tot_cnt,1):>6.1%} {rel/max(cnt,1):>9.4f} "
              f"{rel/max(tot_rel,1e-9):>13.1%} {ls/max(cnt,1):>10.5f} {ab/max(cnt,1):>9.4f}")
    print(f"\n  TOTAL rel_mae (raw span, = reported metric) = {tot_rel/max(tot_cnt,1):.4f}")
    print(f"  TOTAL loss   (2%-floored span, = optimised) = {tot_loss/max(tot_cnt,1):.5f}")
    print(f"  TOTAL abs_mae (raw param units)              = {tot_abs/max(tot_cnt,1):.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
