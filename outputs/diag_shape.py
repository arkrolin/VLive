"""Shape-vs-amplitude diagnostic over a render-data pickle.

Answers the question the abs_mae number cannot: did the model actually learn
*shape*, or only amplitude?

For every sample we compare pred / exem against target on:
    point MAE    - raw-unit mean |pred - target|          (amplitude-dominated)
    delta MAE    - mean |d(pred) - d(target)| along T     (pure shape)
    shape corr   - corr(d(pred), d(target))               (shape agreement)
and split channels by whether the target is (nearly) constant, because constant
channels are pure amplitude and hide the shape story.

Usage
-----
    .venv/bin/python outputs/diag_shape.py outputs/render_data/abl_R_all_v9_best.pkl
    .venv/bin/python outputs/diag_shape.py outputs/render_data/abl_V11a_charloo_best.pkl \
                    --ref outputs/render_data/abl_R_all_v9_best.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def delta(x: np.ndarray) -> np.ndarray:
    return np.diff(x, axis=-1)


def metrics(pred: np.ndarray, tgt: np.ndarray) -> dict[str, float]:
    """All three metrics, plus the constant / non-constant channel split."""
    out: dict[str, float] = {}
    ae = np.abs(pred - tgt)
    de = np.abs(delta(pred) - delta(tgt))

    # a channel is "constant" if its target barely moves across time
    ptp = tgt.max(axis=-1) - tgt.min(axis=-1)
    const = ptp < 1e-6

    out["point"] = float(ae.mean())
    out["delta"] = float(de.mean())
    out["point_const"] = float(ae[const].mean()) if const.any() else float("nan")
    out["point_var"] = float(ae[~const].mean()) if (~const).any() else float("nan")
    out["delta_var"] = float(de[~const].mean()) if (~const).any() else float("nan")

    # shape correlation over non-constant channels (constant ones carry no shape)
    if (~const).any():
        a = delta(pred)[~const].ravel()
        b = delta(tgt)[~const].ravel()
        if a.std() > 0 and b.std() > 0:
            out["shape_corr"] = float(np.corrcoef(a, b)[0, 1])
        else:
            out["shape_corr"] = float("nan")
    else:
        out["shape_corr"] = float("nan")
    out["const_frac"] = float(const.mean())
    return out


def avg(xs: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(np.nanmean(v)) for k, v in xs.items()}


def load_metrics(pkl: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Metrics for the model and for the exem prior stored in the same pickle.

    Returning both matters: exem is the baseline to beat, and on SHAPE it is a
    genuinely strong baseline (the cross-character action mean already carries
    the temporal pattern). A model that loses to it on delta MAE has not solved
    shape at all, no matter how good its point MAE looks.
    """
    blob = pickle.load(open(pkl, "rb"))
    acc_m: dict[str, list[float]] = {}
    acc_e: dict[str, list[float]] = {}
    for it in blob["items"]:
        for k, v in metrics(it["pred"], it["target"]).items():
            acc_m.setdefault(k, []).append(v)
        for k, v in metrics(it["exem"], it["target"]).items():
            acc_e.setdefault(k, []).append(v)
    return avg(acc_m), avg(acc_e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", type=Path)
    ap.add_argument("--ref", type=Path, default=None,
                    help="second pickle to print side by side")
    args = ap.parse_args()

    a, a_ex = load_metrics(args.pkl)
    name_a = args.pkl.stem
    b, _ = load_metrics(args.ref) if args.ref else (None, None)
    name_b = args.ref.stem if args.ref else None

    keys = ["point", "delta", "shape_corr", "point_const", "point_var",
            "delta_var", "const_frac"]
    labels = {
        "point": "point MAE (amplitude-dominated)",
        "delta": "delta MAE (pure shape)",
        "shape_corr": "shape corr (higher=better)",
        "point_const": "  point MAE | constant channels",
        "point_var": "  point MAE | moving channels",
        "delta_var": "  delta MAE | moving channels",
        "const_frac": "constant-channel fraction",
    }

    if b is None:
        print(f"=== shape diagnostic | {name_a} ===")
        print(f"{'metric':42s}{name_a:>22s}{'exem prior':>22s}")
        print("-" * 86)
        for k in keys:
            print(f"{labels[k]:42s}{a[k]:22.4f}{a_ex[k]:22.4f}")
        return 0

    print("=== shape diagnostic ===")
    print(f"{'metric':38s}{name_a:>20s}{name_b:>20s}{'delta':>11s}{'exem':>11s}")
    print("-" * 100)
    for k in keys:
        d = a[k] - b[k]
        print(f"{labels[k]:38s}{a[k]:20.4f}{b[k]:20.4f}{d:11.4f}{a_ex[k]:11.4f}")
    print("-" * 100)
    print("delta = A - B (negative better for MAE, positive better for corr).")
    print("'exem' column = the exem prior scored in A's own pickle.")

    if b["point"] > 0 and b["delta"] > 0:
        print()
        print(f"point MAE improvement : {(b['point'] - a['point']) / b['point'] * 100:6.1f}%")
        print(f"delta MAE improvement : {(b['delta'] - a['delta']) / b['delta'] * 100:6.1f}%")

    # The decisive question: does A beat the prior on SHAPE?
    print()
    print("vs EXEM PRIOR (shape):")
    for k, better_low in (("delta", True), ("delta_var", True), ("shape_corr", False)):
        win = (a[k] < a_ex[k]) if better_low else (a[k] > a_ex[k])
        print(f"  {labels[k].strip():34s} {a[k]:8.4f} vs {a_ex[k]:8.4f}  "
              f"{'BEATS prior' if win else 'LOSES to prior'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
