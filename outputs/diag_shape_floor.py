"""Is the delta-MAE win degenerate?

delta MAE = mean |Δpred - Δtarget|. A model that outputs a CONSTANT curve has
Δpred == 0 everywhere, giving delta MAE = mean |Δtarget| - a nonzero "floor"
that has nothing to do with shape. If an arm's delta MAE sits at or below that
floor, its apparent shape win is just flattening, not tracking shape.

This script measures, on the same pickles diag_shape.py uses:
  * floor_const   = mean |Δtarget|           (per-channel-constant predictor)
  * floor_exem    = mean |Δexem - Δtarget|   (the prior's shape error)
  * model deltaMAE                          (from the pkl's own pred)
  * shape corr of the model vs corr of exem
and reports where each arm sits relative to the floor. Also reports, among
moving channels, the mean |Δtarget| against mean |Δpred| so a flattening model
(pred delta magnitude far below target's) is visible directly.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

KEYS = ("run", "ckpt", "ep")


def stats(pkl: Path) -> dict[str, float]:
    blob = pickle.load(open(pkl, "rb"))
    acc: dict[str, list[float]] = {}
    for it in blob["items"]:
        tgt = it["target"].astype(np.float64)
        pred = it["pred"].astype(np.float64)
        exm = it["exem"].astype(np.float64)
        dt, dp, de = np.diff(tgt, axis=-1), np.diff(pred, axis=-1), np.diff(exm, axis=-1)

        ptp = tgt.max(axis=-1) - tgt.min(axis=-1)
        mov = ptp >= 1e-6
        if not mov.any():
            continue
        dt, dp, de = dt[mov], dp[mov], de[mov]

        rec = {
            # delta MAE of a per-channel-constant predictor == |dtarget|
            "floor_const": float(np.abs(dt).mean()),
            "exem_delta": float(np.abs(de - dt).mean()),
            "model_delta": float(np.abs(dp - dt).mean()),
            # magnitude of the model's own motion vs the target's
            "tgt_dmag": float(np.abs(dt).mean()),
            "pred_dmag": float(np.abs(dp).mean()),
            "exem_dmag": float(np.abs(de).mean()),
            # correlation of the *magnitudes* (flattening shows up as pred<pred)
            "corr_model": float(np.corrcoef(dp.ravel(), dt.ravel())[0, 1])
            if dp.std() > 0 and dt.std() > 0 else float("nan"),
            "corr_exem": float(np.corrcoef(de.ravel(), dt.ravel())[0, 1])
            if de.std() > 0 and dt.std() > 0 else float("nan"),
        }
        for k, v in rec.items():
            acc.setdefault(k, []).append(v)
    out = {k: float(np.nanmean(v)) for k, v in acc.items()}
    out["n"] = len(blob["items"])
    out["ep"] = blob.get("epoch", -1)
    return out


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: diag_shape_floor.py <pkl> [pkl ...]")
        return 1
    rows = [(p.stem, stats(p)) for p in paths]
    hdr = (f"{'arm':<26}{'ep':>4}{'deltaMAE':>10}{'floor':>9}{'exem':>9}"
           f"{'|dpred|':>9}{'|dtgt|':>9}{'corr':>8}{'corr_ex':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, s in rows:
        print(f"{name:<26}{s['ep']:>4}{s['model_delta']:>10.4f}"
              f"{s['floor_const']:>9.4f}{s['exem_delta']:>9.4f}"
              f"{s['pred_dmag']:>9.4f}{s['tgt_dmag']:>9.4f}"
              f"{s['corr_model']:>8.4f}{s['corr_exem']:>9.4f}")
    print("\nmoving channels only. 'floor' = mean|dtarget| = delta MAE of a")
    print("per-channel-CONSTANT predictor. An arm at/below 'floor' did not")
    print("track shape, it flattened. |dpred| << |dtgt| => flattening.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
