"""Where does the model's advantage actually come from? (CPU-only, uses the dumps)

The w-sweep (§17) showed the residual overshoots; the per-character diagnostic
(§18) showed the model only wins on 54% of characters and seems to have an
error floor. Both point at the same thing: the model applies a residual of
roughly constant magnitude whether or not it is needed.

This script quantifies that directly on the per-param-instance error dumps
(``outputs/eval_dumps/*.npz``, produced by ``eval_ckpt.py --dump``), and answers
the question that actually decides the next step:

    How much is left on the table if the model could simply choose, per
    param-instance, between its own prediction and the exem prior?

That "oracle gate" is the CEILING for any mechanism that learns when not to
apply the residual (a shrinkage gate, a confidence head, ...). If the ceiling is
far below both, gating is the highest-value next move; if it is close to the
model's own number, gating is not worth it.

Usage
-----
    .venv/bin/python outputs/diag_error_floor.py [run ...]
    .venv/bin/python outputs/diag_error_floor.py            # defaults to H
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DUMPS = ROOT / "outputs" / "eval_dumps"
DEFAULT = "abl_H_regress285_ep45"


def analyse(run: str) -> None:
    path = DUMPS / f"{run}.npz"
    if not path.exists():
        print(f"  !! no dump for {run} (expected {path})")
        return
    z = np.load(path)
    m = z["err"].astype(np.float64)        # model,     per param-instance abs error
    e = z["exem_err"].astype(np.float64)   # exem prior, same instances, same order
    n = min(len(m), len(e))
    m, e = m[:n], e[:n]

    mean_m, mean_e = m.mean(), e.mean()
    oracle = np.minimum(m, e)              # perfect per-instance choice
    win = (m < e).mean()

    print(f"\n=== {run} ===   instances: {n}")
    print(f"model        abs_mae = {mean_m:.4f}")
    print(f"exem prior   abs_mae = {mean_e:.4f}")
    print(f"model vs prior       = {(mean_e - mean_m) / mean_e * 100:+.1f}%")
    print(f"instance win rate    = {win * 100:.1f}%  "
          f"(model closer than prior on this share of param-instances)")
    print(f"ORACLE GATE  abs_mae = {oracle.mean():.4f}   "
          f"<-- ceiling for 'learn when not to apply the residual'")
    print(f"oracle vs prior      = {(mean_e - oracle.mean()) / mean_e * 100:+.1f}%")
    print(f"oracle vs model      = {(mean_m - oracle.mean()) / mean_m * 100:+.1f}%"
          f"   <-- how much gating could still buy")

    # ---- bucket by how good the prior already is ----------------------------
    print(f"\n  {'exem_err decile':>18}{'n':>7}{'exem':>9}{'model':>9}"
          f"{'vs exem':>10}{'win%':>7}{'oracle':>9}")
    order = np.argsort(e)
    k = 10
    for i in range(k):
        lo, hi = i * n // k, (i + 1) * n // k
        idx = order[lo:hi]
        me, mm = e[idx], m[idx]
        g = (me.mean() - mm.mean()) / me.mean() * 100 if me.mean() else float("nan")
        print(f"  {i + 1:>18}{len(idx):>7}{me.mean():>9.4f}{mm.mean():>9.4f}"
              f"{g:>9.1f}%{(mm < me).mean() * 100:>7.1f}"
              f"{np.minimum(mm, me).mean():>9.4f}")

    # ---- the error-floor test ----------------------------------------------
    # If the model adds a residual of roughly CONSTANT magnitude regardless of
    # whether one is needed, its error should flatten out in the low-exem
    # deciles (a floor) instead of tracking the prior's error down.
    lows = [e[order[i * n // k:(i + 1) * n // k]].mean() for i in range(4)]
    lowm = [m[order[i * n // k:(i + 1) * n // k]].mean() for i in range(4)]
    print("\n  error-floor test (bottom 4 deciles by prior error):")
    print(f"    prior : {'  '.join(f'{v:.4f}' for v in lows)}")
    print(f"    model : {'  '.join(f'{v:.4f}' for v in lowm)}")
    spread_e = (max(lows) - min(lows)) / max(lows) * 100
    spread_m = (max(lowm) - min(lowm)) / max(lowm) * 100
    print(f"    relative spread across these deciles: prior {spread_e:.0f}%  "
          f"model {spread_m:.0f}%")
    print("    -> model spread much SMALLER than prior spread => the model's error "
          "has a\n       floor; it cannot go below it even where the prior is "
          "already accurate.")


def main() -> int:
    runs = sys.argv[1:] or [DEFAULT]
    for r in runs:
        analyse(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
