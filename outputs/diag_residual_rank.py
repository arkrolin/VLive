"""Is the residual (target - exem prior) low-rank structure, or noise?

This decides whether a AuK-style frozen curve-autoencoder (compress the
(P, T) curve matrix into a low-dim latent, regress/generate there) can buy
anything.  If the residual's singular-value spectrum is indistinguishable
from a matched random Gaussian matrix, there is no manifold to compress and
the idea is dead -- we would be at the error floor.

Reports, over val samples:
  * R2 of the exem prior            -- how much is already explained
  * k90 / PR of the residual        -- effective rank (vs min(P, T))
  * the same for a matched-noise baseline (per-row std preserved)
  * top-1 share of residual energy  -- a rank-1 residual is an amplitude
                                       error, not a shape error, and wants a
                                       different fix (a gain, not a VAE)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/work/nlp/xjzhao13/lijie_llama/VLive")
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))

from config import PipelineConfig  # noqa: E402
from dataset import Live2DDataset   # noqa: E402

ARMS = {
    "Q_std_v9": dict(
        data_root="standrad-live-2d",
        whitelist_path="outputs/model_whitelist.json",
        gen_mask_path="outputs/gen_mask.json",
        val_holdout_path="outputs/val_holdout_std.json",
        dedup_skip_path="outputs/dedup_skip_std.json",
        subset_models=285, cache_tag="v9std",
    ),
    "R_all_v9": dict(
        data_root="data/all",
        whitelist_path="outputs/model_whitelist_all.json",
        gen_mask_path="outputs/gen_mask_all.json",
        val_holdout_path="outputs/val_holdout_all.json",
        dedup_skip_path="outputs/dedup_skip_all.json",
        subset_models=580, cache_tag="v9all",
    ),
}
COMMON = dict(action_map_path="outputs/action_semantic_map.json")


def spectrum(m: np.ndarray):
    """Return (k90, participation ratio, top1 share) of a 2-D matrix."""
    m = np.asarray(m, dtype=np.float64)
    if m.size == 0 or not np.isfinite(m).all():
        return None
    s = np.linalg.svd(m, compute_uv=False)
    e = s ** 2
    tot = e.sum()
    if tot <= 0:
        return None
    frac = e / tot
    k90 = int(np.searchsorted(np.cumsum(frac), 0.9) + 1)
    pr = float(tot ** 2 / (e ** 2).sum())
    return k90, pr, float(frac[0])


def main() -> None:
    rng = np.random.default_rng(0)
    for name, kw in ARMS.items():
        cfg = PipelineConfig()
        for k, v in kw.items():
            setattr(cfg, k, ROOT / v if k.endswith(("_root", "_path")) else v)
        for k, v in COMMON.items():
            setattr(cfg, k, v)
        va = Live2DDataset(cfg, split="val")
        print(f"\n=== {name} ===  val samples={len(va.samples)}", flush=True)

        r2s, k90r, prr, k90n, prn, top1, ranks = [], [], [], [], [], [], []
        for i in range(len(va.samples)):
            it = va[i]
            tgt = np.asarray(it["target"], dtype=np.float64)
            exm = np.asarray(it["exem"], dtype=np.float64)
            if tgt.ndim != 2 or tgt.shape[0] < 4:
                continue
            res = tgt - exm
            r2 = 1.0 - (res ** 2).sum() / max((tgt ** 2).sum(), 1e-12)
            r2s.append(r2)

            a = spectrum(res)
            # matched noise baseline: same shape, per-row std preserved
            sd = res.std(axis=1, keepdims=True)
            sd = np.where(sd < 1e-9, 1e-9, sd)
            noise = rng.standard_normal(res.shape) * sd
            b = spectrum(noise)
            if a is None or b is None:
                continue
            k90r.append(a[0]); prr.append(a[1]); top1.append(a[2])
            k90n.append(b[0]); prn.append(b[1])
            ranks.append(min(res.shape))

        if not r2s:
            print("  no usable samples")
            continue
        r2s = np.array(r2s); k90r = np.array(k90r); prr = np.array(prr)
        k90n = np.array(k90n); prn = np.array(prn)
        top1 = np.array(top1); ranks = np.array(ranks, dtype=float)

        print(f"  exem prior R2           : median {np.median(r2s):.3f}"
              f"   (p10 {np.percentile(r2s,10):.3f} / p90 {np.percentile(r2s,90):.3f})")
        print(f"  residual k90            : median {np.median(k90r):.0f}"
              f"   / min(P,T) median {np.median(ranks):.0f}"
              f"   -> {np.median(k90r)/np.median(ranks):.1%} of full rank")
        print(f"  noise  baseline k90     : median {np.median(k90n):.0f}"
              f"   -> {np.median(k90n)/np.median(ranks):.1%} of full rank")
        print(f"  residual PR (eff. rank) : median {np.median(prr):.2f}")
        print(f"  noise  baseline PR      : median {np.median(prn):.2f}")
        print(f"  structure ratio PR_res/PR_noise : {np.median(prr)/np.median(prn):.3f}"
              f"   (<0.7 = compressible manifold; ~1.0 = pure noise)")
        print(f"  top-1 share of residual : median {np.median(top1):.3f}"
              f"   (>0.5 = mostly an amplitude/gain error)")


if __name__ == "__main__":
    main()
