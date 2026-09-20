"""Capacity vs structure: which one actually bounds the shape error?

Motivation
----------
The V11e four-arm read (report 15.12) shows the structured head reaches
shape corr 0.7124 while the DENSE head (V11b/V11c) sits at ~0.58, even though
the dense head can represent ANY curve. That asymmetry says the binding
constraint is probably not raw capacity. This script makes that measurable.

The structured head (model.py) is
    out = exem * (1 + alpha) + coef @ B_r
with B_r = the first r DCT-II modes over T, FIXED (not learned). So per
(instance, channel) the family is: low-frequency content free, high-frequency
content only as a scaled copy of exem. The best fit in that family is analytic:

    P     = B_r^T B_r                       (projector onto the r low-freq modes)
    beta  = <(I-P)t, (I-P)e> / ||(I-P)e||^2      (per channel; 1 if denom ~ 0)
    pred  = P t + beta (I - P) e

Scanning r gives the REPRESENTATIONAL CEILING of the structured family. Compare
it with what B1b actually achieves (0.7124 at r=8):

  * ceiling(r=8) >> 0.7124  -> the head is nowhere near its capacity; the
                               bottleneck is information / optimisation.
  * ceiling(r=8) ~= 0.7124  -> the band-limit IS the binding constraint, and
                               raising r (B1c) should help.

Metric definitions are copied verbatim from diag_shape.py / diag_shape_floor.py
so the numbers are directly comparable with the report.

Usage
-----
    .venv/bin/python outputs/diag_capacity_vs_structure.py \
        outputs/render_data/abl_V11e_B1b_best.pkl
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import numpy as np

RS = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48)


def dct_basis(r: int, T: int) -> np.ndarray:
    """First r orthonormal DCT-II modes, (r, T) -- identical to model._dct_basis."""
    if r <= 0:
        return np.zeros((0, T), dtype=np.float64)
    n = np.arange(T, dtype=np.float64)[None, :]
    k = np.arange(r, dtype=np.float64)[:, None]
    B = np.cos(math.pi * k * (n + 0.5) / T)
    return B / np.linalg.norm(B, axis=1, keepdims=True)


def metrics(pred: np.ndarray, tgt: np.ndarray) -> dict[str, float]:
    ae = np.abs(pred - tgt)
    de = np.abs(np.diff(pred, axis=-1) - np.diff(tgt, axis=-1))
    ptp = tgt.max(axis=-1) - tgt.min(axis=-1)
    const = ptp < 1e-6
    out: dict[str, float] = {"point": float(ae.mean()), "delta": float(de.mean())}
    out["point_var"] = float(ae[~const].mean()) if (~const).any() else float("nan")
    out["delta_var"] = float(de[~const].mean()) if (~const).any() else float("nan")
    if (~const).any():
        a = np.diff(pred, axis=-1)[~const].ravel()
        b = np.diff(tgt, axis=-1)[~const].ravel()
        out["corr"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    else:
        out["corr"] = float("nan")
    return out


def oracle_pred(exem: np.ndarray, tgt: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Best fit of `exem*(1+alpha) + coef @ B` per (instance, channel)."""
    r = B.shape[0]
    if r == 0:
        par = np.zeros_like(tgt)
        perp_t, perp_e = tgt, exem
    else:
        coef_t = tgt @ B.T                      # (P, r)
        coef_e = exem @ B.T
        par = coef_t @ B                        # P t
        perp_t = tgt - par
        perp_e = exem - coef_e @ B              # (I - P) e
    den = (perp_e ** 2).sum(axis=-1)            # (P,)
    num = (perp_t * perp_e).sum(axis=-1)
    beta = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 1.0)
    return par + beta[:, None] * perp_e


def main() -> int:
    pkl = Path(sys.argv[1] if len(sys.argv) > 1 else
               "outputs/render_data/abl_V11e_B1b_best.pkl")
    blob = pickle.load(open(pkl, "rb"))
    items = blob["items"]
    T = int(items[0]["T"])
    print(f"=== capacity vs structure | {blob['run']} {blob['ckpt']} ep{blob['epoch']} "
          f"| n={blob['n']} T={T} ===")
    print()

    # ---- A. the family's representational ceiling, as a function of r ---- #
    print("A. structured-head ceiling  (out = exem*(1+alpha) + coef @ DCT_r)")
    print(f"{'r':>4}{'point':>10}{'delta':>10}{'corr':>9}{'delta|move':>12}")
    print("-" * 45)
    for r in RS:
        B = dct_basis(r, T)
        acc: dict[str, list[float]] = {}
        for it in items:
            for k, v in metrics(oracle_pred(it["exem"].astype(np.float64),
                                            it["target"].astype(np.float64), B),
                                it["target"].astype(np.float64)).items():
                acc.setdefault(k, []).append(v)
        m = {k: float(np.nanmean(v)) for k, v in acc.items()}
        tag = "  <- r=8 (B1b)" if r == 8 else ("  <- B1c" if r == 16 else
              ("  <- B1a" if r == 2 else ""))
        print(f"{r:>4}{m['point']:>10.4f}{m['delta']:>10.4f}{m['corr']:>9.4f}"
              f"{m['delta_var']:>12.4f}{tag}")
    print()
    print("   reference: the exem prior itself has corr 0.5740 / delta|move 0.2235;")
    print("   the DENSE head family can represent any curve (ceiling corr = 1.0),")
    print("   yet V11b/V11c only achieve ~0.58 -> dense is not capacity-bound either.")
    print()

    # ---- B. what the real model achieved, same pickle / same metrics ---- #
    print("B. what the trained arms actually achieve (same metric code)")
    print(f"{'arm':>26}{'point':>10}{'delta':>10}{'corr':>9}{'delta|move':>12}")
    print("-" * 67)
    for name in ("abl_V11e_B1b_best", "abl_V11e_D1_best", "abl_V11e_B2_best",
                 "abl_V11b_bank_best", "abl_V11c_best", "abl_V11a_charloo_best"):
        p = Path("outputs/render_data") / f"{name}.pkl"
        if not p.exists():
            continue
        b2 = pickle.load(open(p, "rb"))
        acc = {}
        for it in b2["items"]:
            for k, v in metrics(it["pred"].astype(np.float64),
                                it["target"].astype(np.float64)).items():
                acc.setdefault(k, []).append(v)
        m = {k: float(np.nanmean(v)) for k, v in acc.items()}
        print(f"{name:>26}{m['point']:>10.4f}{m['delta']:>10.4f}{m['corr']:>9.4f}"
              f"{m['delta_var']:>12.4f}")
    print()

    # ---- C. intrinsic rank of what must be ADDED to the prior ---- #
    rows, rows_move = [], []
    for it in items:
        t = it["target"].astype(np.float64)
        e = it["exem"].astype(np.float64)
        ptp = t.max(axis=-1) - t.min(axis=-1)
        mov = ptp >= 1e-6
        rows.append((t - e))
        if mov.any():
            rows_move.append((t - e)[mov])
    R = np.concatenate(rows, axis=0)
    Rm = np.concatenate(rows_move, axis=0)
    print("C. intrinsic rank of the residual  r = target - exem")
    print(f"   stacked shape: all channels {R.shape}, moving only {Rm.shape}")
    for name, M in (("all", R), ("moving", Rm)):
        s = np.linalg.svd(M, compute_uv=False) ** 2
        cs = np.cumsum(s) / s.sum()
        k = {p: int(np.searchsorted(cs, p / 100.0) + 1) for p in (80, 90, 95, 99)}
        print(f"   {name:>7}: modes for 80/90/95/99% energy = "
              f"{k[80]}/{k[90]}/{k[95]}/{k[99]}  (of {M.shape[1]})")
    print()

    # ---- D. how much of that residual energy the first r DCT modes capture ---- #
    print("D. energy of the (moving-channel) residual captured by first r DCT modes")
    line = []
    for r in (1, 2, 4, 8, 12, 16, 24, 32):
        B = dct_basis(r, T)
        cap = float((((Rm @ B.T) @ B) ** 2).sum() / (Rm ** 2).sum())
        line.append(f"r={r}:{cap*100:.1f}%")
    print("   " + "  ".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
