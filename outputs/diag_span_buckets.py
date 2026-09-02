"""Bucket the dumped per-instance errors by the PARAMETER'S OWN SPAN.

The decile table in ``diag_error_floor.py`` showed ~30% of param-instances have
an exem-prior error of exactly 0, and the model injects ~1 raw unit of error
into them. Two very different stories fit that observation, and they call for
opposite fixes:

  (a) those instances are NEAR-CONSTANT params (raw span ~ 0, floored to
      2% of the global range = 22.66). Then the ~1 unit of model error is
      22.66 * |residual|, i.e. the model is making a LARGE normalised mistake
      on params that simply should not move -> fix the model / drop them.

  (b) those instances are LARGE-span params the prior happens to predict well.
      Then 1 raw unit is a tiny normalised error and nothing is broken; the
      aggregate `abs_mae` is just dominated by a few big-range params.

This script settles it by replaying the val set in the SAME order that
``recon_metrics`` iterates it (CPU only, no model forward) and recording the
floored span of every param-instance. It then joins that against the dumped
errors so every error can be bucketed by span.

Usage
-----
    .venv/bin/python outputs/diag_span_buckets.py [run ...]
    .venv/bin/python outputs/diag_span_buckets.py                 # defaults to H
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))

from config import PipelineConfig  # noqa: E402
from dataset import Live2DDataset, collate  # noqa: E402
from train import compute_range, mean_range_vecs  # noqa: E402

DUMPS = ROOT / "outputs" / "eval_dumps"
CACHE = ROOT / "outputs" / "eval_dumps" / "_val_spans_cache.pkl"
DEFAULT = "abl_H_regress285_ep45"


def build_val_spans(cfg) -> np.ndarray:
    """Floored span of every param-instance, in recon_metrics iteration order."""
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            blob = pickle.load(f)
        if blob.get("n_pad") == cfg.max_tokens:
            return blob["spans"]

    train_ds = Live2DDataset(cfg, split="train")
    val_ds = Live2DDataset(cfg, split="val")
    g_lo, g_hi, per_lo, per_hi = compute_range(train_ds, n=400)
    min_span = max(g_hi - g_lo, 1.0) * 0.02

    loader = DataLoader(val_ds, batch_size=4, sampler=SequentialSampler(val_ds),
                        collate_fn=lambda b: collate(b, cfg.max_tokens),
                        num_workers=0)
    spans: list[float] = []
    for batch in loader:
        names = batch["names"]
        token_mask = batch["token_mask"]
        lo_v, hi_v = mean_range_vecs(names, per_lo, per_hi, g_lo, g_hi,
                                     cfg.max_tokens)
        span_v = np.clip(hi_v - lo_v, min_span, None)     # (B, n_pad)
        B, n = token_mask.shape
        for b in range(B):
            for j in range(n):
                if token_mask[b, j] < 0.5:
                    continue
                spans.append(float(span_v[b, j]))
    arr = np.asarray(spans, dtype=np.float64)
    with open(CACHE, "wb") as f:
        pickle.dump({"n_pad": cfg.max_tokens, "spans": arr}, f)
    print(f"[cached val spans for {len(arr)} param-instances -> {CACHE.name}]")
    return arr


def analyse(run: str, spans: np.ndarray) -> None:
    path = DUMPS / f"{run}.npz"
    if not path.exists():
        print(f"  !! no dump for {run}")
        return
    z = np.load(path)
    m = z["err"].astype(np.float64)
    e = z["exem_err"].astype(np.float64)
    n = min(len(m), len(e), len(spans))
    m, e, sp = m[:n], e[:n], spans[:n]
    if len(spans) != len(z["err"]):
        print(f"  !! span count {len(spans)} != err count {len(z['err'])} "
              f"- joining on the first {n}")

    print(f"\n=== {run} ===   instances: {n}")
    print(f"model {m.mean():.4f} | exem {e.mean():.4f} | "
          f"vs exem {(e.mean() - m.mean()) / e.mean() * 100:+.1f}%")

    print(f"\n  {'span decile':>14}{'n':>7}{'span':>10}{'exem':>9}{'model':>9}"
          f"{'vs exem':>10}{'win%':>7}{'model_e_f':>11}")
    order = np.argsort(sp)
    k = 10
    for i in range(k):
        lo, hi = i * n // k, (i + 1) * n // k
        idx = order[lo:hi]
        s, me, mm = sp[idx], e[idx], m[idx]
        g = (me.mean() - mm.mean()) / me.mean() * 100 if me.mean() else float("nan")
        # e_f = normalised error = abs error / span  -> comparable across spans
        ef = (mm / s).mean()
        print(f"  {i + 1:>14}{len(idx):>7}{s.mean():>10.2f}{me.mean():>9.4f}"
              f"{mm.mean():>9.4f}{g:>9.1f}%{(mm < me).mean() * 100:>7.1f}"
              f"{ef:>11.5f}")

    # ---- the crux: are the "prior error == 0" instances constant or big? -----
    zero = e < 1e-6
    print(f"\n  instances with EXACTLY zero prior error: {zero.sum()} "
          f"({zero.mean() * 100:.1f}%)")
    if zero.sum():
        print(f"    their mean span        : {sp[zero].mean():.2f}")
        print(f"    overall mean span      : {sp.mean():.2f}")
        print(f"    floored-span floor     : {sp.min():.2f}")
        print(f"    model error on them    : {m[zero].mean():.4f} abs, "
              f"{(m[zero] / sp[zero]).mean():.5f} normalised")
        print(f"    model error elsewhere  : {m[~zero].mean():.4f} abs, "
              f"{(m[~zero] / sp[~zero]).mean():.5f} normalised")
        verdict = ("(a) NEAR-CONSTANT params -> the model makes a LARGE "
                   "normalised\nmistake on params that should not move"
                   if (m[zero] / sp[zero]).mean() > (m[~zero] / sp[~zero]).mean()
                   else "(b) LARGE-span params the prior predicts well -> the "
                        "normalised\nerror is small; nothing is broken, abs_mae "
                        "is just span-weighted")
        print(f"    VERDICT: {verdict}")


def main() -> int:
    cfg = PipelineConfig()
    cfg.subset_models = 285
    cfg.action_cond = "id"
    cfg.eval_shared_min_models = 5
    spans = build_val_spans(cfg)
    for r in (sys.argv[1:] or [DEFAULT]):
        analyse(r, spans)
    return 0


if __name__ == "__main__":
    sys.exit(main())
