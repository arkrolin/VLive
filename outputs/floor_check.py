"""Compute the range-normalized exem floor + exem-only reconstruction baseline.

Range normalization: x0 = (target - lo) / (hi - lo). In these units the loss is
~rel_mae^2, so the floor (predict the corpus exem mean) should be ~exem rel^2.
This confirms the new VAL_TARGET / REL_MAE_TARGET are calibrated correctly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))

import numpy as np
from config import PipelineConfig
from dataset import Live2DDataset, collate
from train import compute_range, mean_range_vecs, arm_weight
from torch.utils.data import DataLoader, SequentialSampler

cfg = PipelineConfig()
train_ds = Live2DDataset(cfg, split="train")
val_ds = Live2DDataset(cfg, split="val")
g_lo, g_hi, per_lo, per_hi = compute_range(train_ds, n=400)
print(f"[floor_check] global range=({g_lo:.3f},{g_hi:.3f}) per-param={len(per_lo)}")

arm_w = 3.0
vloader = DataLoader(
    val_ds, batch_size=cfg.batch_size, sampler=SequentialSampler(val_ds),
    collate_fn=lambda b: collate(b, cfg.max_tokens))

floor_sum, floor_cnt = 0.0, 0.0
ra_sum, rr_sum, rc = 0.0, 0.0, 0
for batch in vloader:
    target = np.asarray(batch["target"], np.float32)
    exem = np.asarray(batch["exem"], np.float32)
    names = batch["names"]
    token_mask = np.asarray(batch["token_mask"], np.float32)
    B, n, T = target.shape
    lo_v, hi_v = mean_range_vecs(names, per_lo, per_hi, g_lo, g_hi, cfg.max_tokens,
                             fb_span=getattr(cfg, 'fb_span', None))
    min_span = max(g_hi - g_lo, 1.0) * 0.02
    span = (hi_v - lo_v)
    span = np.maximum(span, min_span)
    x0 = (target - lo_v[:, :, None]) / span[:, :, None]
    exem_s = (exem - lo_v[:, :, None]) / span[:, :, None]
    se = (exem_s - x0) ** 2
    aw = arm_weight(names, arm_w, n_pad=cfg.max_tokens)[:, :, None]
    valid = token_mask[:, :, None] * aw
    floor_sum += (se * valid).sum()
    floor_cnt += valid.sum()
    # exem-only reconstruction in ORIGINAL param units
    x0_hat_u = exem_s * span[:, :, None] + lo_v[:, :, None]
    err = np.abs(x0_hat_u - target)
    for b in range(B):
        for j in range(n):
            if token_mask[b, j] < 0.5:
                continue
            nm = names[b][j]
            lo = per_lo.get(nm, g_lo)
            hi = per_hi.get(nm, g_hi)
            sp = max(hi - lo, 1e-3)
            e = float(err[b, j].mean())
            ra_sum += e
            rr_sum += min(e / sp, 2.0)
            rc += 1

print(f"[floor_check] RANGE-NORM exem floor (val, weighted arm_w={arm_w}): "
      f"{floor_sum / floor_cnt:.5f}")
print(f"[floor_check] EXEM-ONLY recon  abs_mae={ra_sum / rc:.4f}  "
      f"rel_mae={rr_sum / rc:.4f}  (n={rc})")
print(f"[floor_check] => VAL_TARGET=0.020 is ~{0.020 / (floor_sum / floor_cnt):.1f}x the floor; "
      f"REL_MAE_TARGET=0.15 vs exem rel {rr_sum / rc:.4f}")
