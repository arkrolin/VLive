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


def build_val_spans(cfg):
    """Floored span + param name of every instance, in recon_metrics order.

    Returns ``(spans, names)``. CPU only - no model forward is needed because
    the span depends solely on (param name, per-param range stats).
    """
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            blob = pickle.load(f)
        if blob.get("n_pad") == cfg.max_tokens and "names" in blob:
            return blob["spans"], blob["names"], blob["g_range"]

    train_ds = Live2DDataset(cfg, split="train")
    val_ds = Live2DDataset(cfg, split="val")
    g_lo, g_hi, per_lo, per_hi, fb_span = compute_range(
        train_ds, n=getattr(cfg, 'range_stats_n', None))
    if getattr(cfg, 'fb_span', None) is None:
        cfg.fb_span = fb_span
    min_span = max(g_hi - g_lo, 1.0) * 0.02

    loader = DataLoader(val_ds, batch_size=4, sampler=SequentialSampler(val_ds),
                        collate_fn=lambda b: collate(b, cfg.max_tokens),
                        num_workers=0)
    spans: list[float] = []
    pnames: list[str] = []
    for batch in loader:
        names = batch["names"]
        token_mask = batch["token_mask"]
        lo_v, hi_v = mean_range_vecs(names, per_lo, per_hi, g_lo, g_hi, cfg.max_tokens,
                             fb_span=getattr(cfg, 'fb_span', None))
        span_v = np.clip(hi_v - lo_v, min_span, None)     # (B, n_pad)
        B, n = token_mask.shape
        for b in range(B):
            for j in range(n):
                if token_mask[b, j] < 0.5:
                    continue
                spans.append(float(span_v[b, j]))
                pnames.append(names[b][j])
    arr = np.asarray(spans, dtype=np.float64)
    with open(CACHE, "wb") as f:
        pickle.dump({"n_pad": cfg.max_tokens, "spans": arr, "names": pnames,
                     "g_range": float(g_hi - g_lo)}, f)
    print(f"[cached val spans for {len(arr)} param-instances -> {CACHE.name}]")
    return arr, np.asarray(pnames), float(g_hi - g_lo)


def analyse(run: str, spans: np.ndarray, pnames: np.ndarray,
            g_range: float) -> None:
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

    # ---- THE BIG ONE: is the top-span decile a range-fallback artefact? -----
    # mean_range_vecs() seeds lo/hi with the GLOBAL range and only overwrites
    # params present in per_lo/per_hi (built from just 400 samples). Any param
    # missing from that dict silently gets span = g_hi - g_lo, i.e. it is
    # treated as if it traversed the entire global range even when it is a
    # constant. abs_mae = normalised_error * span, so those instances are
    # inflated by ~50x and can dominate the whole metric.
    fb = np.abs(sp - g_range) < 1e-3
    print(f"\n  !! range-fallback check (span == global range {g_range:.1f}): "
          f"{fb.sum()} instances ({fb.mean() * 100:.1f}%)")
    if fb.sum():
        print(f"     their share of the model's total abs_mae : "
              f"{m[fb].sum() / m.sum() * 100:.1f}%")
        print(f"     their share of the prior's total abs_mae: "
              f"{e[fb].sum() / e.sum() * 100:.1f}%")
        print(f"     mean abs error on them   : model {m[fb].mean():.4f} vs "
              f"prior {e[fb].mean():.4f}")
        print(f"     mean NORMALISED error    : model "
              f"{(m[fb] / sp[fb]).mean():.5f} vs prior "
              f"{(e[fb] / sp[fb]).mean():.5f}   <-- both tiny")
        print(f"     distinct params affected : {len(set(pnames[fb]))}")
        excl_m = m[~fb].mean()
        excl_e = e[~fb].mean()
        print(f"     EXCLUDING them: model {excl_m:.4f} vs prior {excl_e:.4f} "
              f"= {(excl_e - excl_m) / excl_e * 100:+.1f}%   "
              f"<-- vs {((e.mean() - m.mean()) / e.mean() * 100):+.1f}% with them")

    # ---- which params make up the losing top-span decile? -------------------
    # Decides whether the damage is visually important: params like ParamMouseX
    # span the whole canvas but barely move the character, whereas ParamAngleZ
    # is small-range yet highly visible.
    order_s = np.argsort(sp)
    top = order_s[9 * n // 10:]
    uniq, counts = np.unique(pnames[top], return_counts=True)
    o = np.argsort(-counts)
    print(f"\n  top-span decile composition ({len(top)} instances, "
          f"mean span {sp[top].mean():.1f}) - top 12 params:")
    for u, c in list(zip(uniq[o], counts[o]))[:12]:
        sel = top[pnames[top] == u]
        g = ((e[sel].mean() - m[sel].mean()) / e[sel].mean() * 100
             if e[sel].mean() else float("nan"))
        print(f"    {u:<28} n={c:<6} span={sp[sel].mean():>8.1f} "
              f"exem={e[sel].mean():>8.3f} model={m[sel].mean():>8.3f} "
              f"({g:+.1f}%)")


def main() -> int:
    cfg = PipelineConfig()
    cfg.subset_models = 285
    cfg.action_cond = "id"
    cfg.eval_shared_min_models = 5
    spans, pnames, g_range = build_val_spans(cfg)
    for r in (sys.argv[1:] or [DEFAULT]):
        analyse(r, spans, pnames, g_range)
    return 0


if __name__ == "__main__":
    sys.exit(main())
