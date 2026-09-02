"""Paired comparison of checkpoints that were scored in SEPARATE processes.

eval_ckpt.py --dump saves the per-param-instance error vector of each run. Two
runs scored on the same val subset (same order, same batch size) produce vectors
that align element-wise, so the difference can be tested with far more precision
than either mean - even though the runs never shared a process.

Usage
-----
    .venv/bin/python outputs/compare_dumps.py outputs/eval_dumps
    .venv/bin/python outputs/compare_dumps.py outputs/eval_dumps --metric rel_f
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def load_all(d: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(d.glob("*.npz")):
        z = np.load(f)
        out[f.stem] = {k: z[k] for k in z.files}
    return out


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """mean(a-b), its standard error, and t. Positive => a is WORSE."""
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    d = a[:n] - b[:n]
    m = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    return float(m), float(se), float(m / se) if se > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--metric", default="err", choices=["err", "rel_f"],
                    help="'err' = per-instance |error| in raw param units; "
                         "'rel_f' = loss-aligned relative error")
    ap.add_argument("--base", default=None,
                    help="run to compare everything against (default: lowest mean)")
    args = ap.parse_args()

    runs = load_all(Path(args.dump_dir))
    if not runs:
        print(f"no .npz dumps in {args.dump_dir}")
        return 1

    print(f"\n=== paired comparison | metric={args['metric'] if False else args.metric} ===")
    rows = []
    for name, z in runs.items():
        v = z[args.metric] if z[args.metric].ndim else z[args.metric]
        ex = z["exem_err"] if args.metric == "err" else z["exem_rel_f"]
        exem = float(np.atleast_1d(ex).mean())
        rows.append((name, float(np.atleast_1d(v).mean()), exem,
                     np.atleast_1d(v), np.atleast_1d(ex)))
    rows.sort(key=lambda r: r[1])

    base_name = args.base or rows[0][0]
    base = next(r for r in rows if r[0] == base_name)

    print(f"{'run':<30}{'mean':>10}{'exem':>10}{'vs exem':>10}"
          f"{'diff vs base':>15}{'t':>9}  verdict")
    print("-" * 92)
    for name, m, exem, vec, exvec in rows:
        gain = (exem - m) / exem * 100 if exem else float("nan")
        if name == base_name:
            print(f"{name:<30}{m:>10.4f}{exem:>10.4f}{gain:>9.1f}%"
                  f"{'-- base --':>15}{'':>9}")
        else:
            diff, se, t = paired(vec, base[3])
            verdict = "significant" if abs(t) > 1.96 else "not significant"
            print(f"{name:<30}{m:>10.4f}{exem:>10.4f}{gain:>9.1f}%"
                  f"{diff:>+15.4f}{t:>9.2f}  {verdict}")
    print(f"\n(base = {base_name}; positive diff = worse than base; |t|>1.96 ~ p<0.05)")
    print("NOTE: means come from separate processes, but the error vectors are")
    print("      element-wise aligned, so the paired diff is tightly estimated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
