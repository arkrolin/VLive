"""Channel-mask diagnostic: is "shape loss only on moving channels" a real saving?

Motivation
----------
§14.6 originally proposed direction (c) as: "constant channels (43%) are pure
amplitude, moving channels (57%) are shape -> restricting the shape term to
moving channels avoids wasting lambda on constant channels."

That premise is testable, and this script tests it. For every (instance, param)
slot we compute the peak-to-peak motion of the TARGET and ask:

  1. how bimodal is the ptp distribution (is any threshold robust)?
  2. how much of the *target* shape energy lives in low-motion channels?
     (a constant channel has d_target == 0, so its target-side contribution is 0)
  3. how much does masking change the achieved shape loss?
     (a channel with a flat target but a jittery prediction still contributes
      (d_pred - 0)^2, so masking is NOT automatically a saving)
  4. what does masking do to the loss DENOMINATOR (this is what forces a lambda
     rescale)?

Usage
-----
    .venv/bin/python outputs/diag_channel_mask.py \
        outputs/render_data/abl_R_all_v9_best.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def analyze(items, thrs=(0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)) -> None:
    ptps = []
    for it in items:
        tgt = np.asarray(it["target"], dtype=np.float64)
        ptps.append(tgt.max(axis=-1) - tgt.min(axis=-1))
    ptp = np.concatenate(ptps)
    print(f"slots (instance x param) : {ptp.size}")
    print(f"const_frac (ptp == 0)    : {(ptp <= 0).mean():.4f}")
    print()
    print("threshold robustness (keep = fraction of slots above threshold):")
    for t in thrs:
        keep = (ptp > t).mean()
        print(f"  ptp > {t:<9g}  keep={keep*100:5.1f}%   drop={(1-keep)*100:5.1f}%")
    print()
    print("ptp percentiles:")
    for q in (5, 10, 25, 50, 75, 90, 95, 99):
        print(f"  p{q:<3d} = {np.percentile(ptp, q):.6f}")

    print()
    print("=== target-side shape energy in low-motion channels ===")
    print("(a constant target channel has d_target == 0, so it can only ever")
    print(" contribute its own (d_pred - 0)^2; its 'target energy' is 0)")
    tot = 0.0
    parts = {t: 0.0 for t in thrs}
    for it in items:
        tgt = np.asarray(it["target"], dtype=np.float64)
        e = (np.diff(tgt, axis=-1) ** 2).mean(axis=-1)
        tot += e.sum()
        pt = tgt.max(-1) - tgt.min(-1)
        for t in thrs:
            parts[t] += e[(pt <= t)].sum()
    for t in thrs:
        print(f"  ptp <= {t:<9g}: {parts[t]/tot*100:5.1f}% of target shape energy")

    print()
    print("=== effect on the achieved shape loss ===")
    for t in (1e-3,):
        num_all = num_mov = 0.0
        cnt_all = cnt_mov = 0
        aps_all = aps_low = 0.0
        for it in items:
            tgt = np.asarray(it["target"], dtype=np.float64)
            prd = np.asarray(it["pred"], dtype=np.float64)
            dt = np.diff(tgt, axis=-1)
            dp = np.diff(prd, axis=-1)
            se = (dp - dt) ** 2
            mov = (tgt.max(-1) - tgt.min(-1)) > t          # (n,)
            mh = np.repeat(mov[:, None], se.shape[1], axis=1)   # (n, T-1)
            num_all += se.sum()
            cnt_all += se.size
            num_mov += se[mh].sum()
            cnt_mov += int(mh.sum())
            aps_all += np.abs(dp).sum()
            aps_low += np.abs(dp[~mh]).sum()
        l_all = num_all / cnt_all
        l_mov = num_mov / cnt_mov
        print(f"  unmasked: loss = {l_all:.6f}  (denominator {cnt_all})")
        print(f"  masked  : loss = {l_mov:.6f}  (denominator {cnt_mov})")
        print(f"  ratio   : {l_mov/l_all:.4f}   <= lambda is effectively scaled by this")
        print(f"  mean |d_pred| over ALL channels    : {aps_all/cnt_all:.6f}")
        print(f"  mean |d_pred| over LOW-motion chans: {aps_low/(cnt_all-cnt_mov):.6f}")
        print()
        print("  -> If ratio > 1, masking makes the loss LARGER (re-weighting, not")
        print("     saving) and lambda must be rescaled down by ~1/ratio.")

    print()
    print("=== prior (exem) amplitude vs target, on moving channels ===")
    print("Relevant to the multiplicative-gain head x0 = exem*(1+alpha):")
    print("where |exem| is ~0 the gain cannot move the curve at all.")
    movv = []
    tg_amp = []
    ex_amp = []
    for it in items:
        ex = np.asarray(it["exem"], dtype=np.float64)
        tg = np.asarray(it["target"], dtype=np.float64)
        mov = (tg.max(-1) - tg.min(-1)) > 1e-3
        movv.append(np.abs(ex)[mov])
        tg_amp.append((tg.max(-1) - tg.min(-1))[mov])
        ex_amp.append((ex.max(-1) - ex.min(-1))[mov])
    movv = np.concatenate(movv)
    tg_amp = np.concatenate(tg_amp)
    ex_amp = np.concatenate(ex_amp)
    print(f"  moving slots                        : {movv.size}")
    print(f"  |exem| < 0.01                       : {(movv < 0.01).mean()*100:5.2f}%")
    print(f"  |exem| < 0.10                       : {(movv < 0.10).mean()*100:5.2f}%")
    print(f"  exem ptp median / target ptp median : {np.median(ex_amp):.4f} / "
          f"{np.median(tg_amp):.4f}")
    print(f"  exem ptp < 0.5 x target ptp         : {(ex_amp < 0.5*tg_amp).mean()*100:5.2f}%")
    print()
    print("  -> A multiplicative gain is a poor fit where the prior under-shoots")
    print("     or is near zero. Prefer an ADDITIVE amplitude term.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", type=Path)
    ap.add_argument("--thr", type=float, default=1e-3,
                    help="motion threshold used for the masking analysis")
    args = ap.parse_args()
    items = pickle.load(open(args.pkl, "rb"))["items"]
    print(f"=== channel-mask diagnostic | {args.pkl.stem} | n_items={len(items)} ===")
    print()
    analyze(items)
    return 0


if __name__ == "__main__":
    sys.exit(main())
