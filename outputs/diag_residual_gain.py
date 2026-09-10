"""Follow-up to diag_residual_rank.py: the residual is ~rank-1 (87.5% of its
energy in one singular direction).  Is that direction "the exemplar prior,
rescaled"?

If  R ~= (alpha - 1) * E  then the whole remaining error is a SCALAR GAIN on
the prior: predicting one number per sample (or per param) recovers most of
it, and a learned curve-autoencoder (AuK-style VAE latent) would be massive
overkill.  If not, the residual is a genuinely different shape and needs a
real generative head.

Reports, per sample:
  cos   = <R,E> / (||R|| ||E||)          alignment with the prior
  a*    = <R,E> / <E,E>                  best scalar gain
  gain_R2 = 1 - ||R - a* E||^2/||R||^2   energy explained by that one scalar
and the same after removing the per-param gain (alpha per row).
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


def main() -> None:
    for name, kw in ARMS.items():
        cfg = PipelineConfig()
        for k, v in kw.items():
            setattr(cfg, k, ROOT / v if k.endswith(("_root", "_path")) else v)
        for k, v in COMMON.items():
            setattr(cfg, k, v)
        va = Live2DDataset(cfg, split="val")
        print(f"\n=== {name} ===  val samples={len(va.samples)}", flush=True)

        cos, g1, gp, norm_e = [], [], [], []
        for i in range(len(va.samples)):
            it = va[i]
            E = np.asarray(it["exem"], dtype=np.float64)
            T = np.asarray(it["target"], dtype=np.float64)
            if E.ndim != 2 or E.shape[0] < 4:
                continue
            R = T - E
            ne = (E ** 2).sum()
            nr = (R ** 2).sum()
            if ne <= 1e-12 or nr <= 1e-12:
                continue
            cos.append(float((R * E).sum() / np.sqrt(nr * ne)))
            a = (R * E).sum() / ne                       # global scalar gain
            g1.append(1.0 - ((R - a * E) ** 2).sum() / nr)
            # per-param (row) gain
            den = (E ** 2).sum(axis=1)
            av = np.where(den > 1e-12, (R * E).sum(axis=1) / np.where(den > 1e-12, den, 1), 0.0)
            gp.append(1.0 - ((R - av[:, None] * E) ** 2).sum() / nr)
            norm_e.append(np.sqrt(ne / E.size))

        cos, g1, gp = np.array(cos), np.array(g1), np.array(gp)
        print(f"  cos(R, E)                 : median {np.median(cos):+.3f}"
              f"   (p10 {np.percentile(cos,10):+.3f} / p90 {np.percentile(cos,90):+.3f})")
        print(f"  |cos| > 0.8               : {np.mean(np.abs(cos) > 0.8):.1%} of samples")
        print(f"  R2 from ONE global gain   : median {np.median(g1):.3f}"
              f"   (p10 {np.percentile(g1,10):.3f} / p90 {np.percentile(g1,90):.3f})")
        print(f"  R2 from per-param gain    : median {np.median(gp):.3f}")
        print(f"  gain R2 >= 0.5            : {np.mean(g1 >= 0.5):.1%} of samples"
              f"   | >= 0.8: {np.mean(g1 >= 0.8):.1%}")
        print(f"  -> per-param gain buys {np.median(gp) - np.median(g1):+.3f} over global gain")


if __name__ == "__main__":
    main()
