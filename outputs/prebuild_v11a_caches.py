"""Pre-build the V11a caches ONCE, before any distributed launch.

Why: build_model_identity (v3) and build_char_param_stats each do a full corpus
pass (12083 motions). If 4 DDP ranks start cold they all race to write the same
.pkl and the loser loads a half-written file. Run this first, single process.

Also prints a sanity check that the leave-one-out rig really differs from the
leaky one, and that char stats carry the rest value.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "live2d_vla"
sys.path.insert(0, str(SRC))

from dataset import (Live2DDataset, build_model_identity,          # noqa: E402
                     build_char_param_stats, CHAR_STATS_NAMES)

cfg = SimpleNamespace(
    whitelist_path=ROOT / "outputs" / "model_whitelist_all.json",
    gen_mask_path=ROOT / "outputs" / "gen_mask_all.json",
    subset_models=580,
    data_root=ROOT / "data" / "all",
    T=48, fps=30.0, rig_sig_dim=96, max_tokens=128,
    action_map_path="outputs/action_semantic_map.json",
    dedup_skip_path="outputs/dedup_skip_all.json",
    val_holdout_path="outputs/val_holdout_all.json",
    cache_tag="v9all",
    rig_loo=True,
    char_stats="mlp",
    eval_shared_min_models=5,
)
import json                                                         # noqa: E402
from io_motion import set_action_map                                # noqa: E402
amap = json.loads((ROOT / "outputs" / "action_semantic_map.json").read_text(encoding="utf-8"))
rev = {}
for g, names in amap.get("groups", {}).items():
    for n in names:
        rev[str(n).lower()] = g
set_action_map(rev)

t0 = time.time()
ds_train = Live2DDataset(cfg, split="train")
print(f"[ok] train dataset {len(ds_train)} samples  ({time.time()-t0:.0f}s)")
t0 = time.time()
ds_val = Live2DDataset(cfg, split="val")
print(f"[ok] val dataset {len(ds_val)} samples  ({time.time()-t0:.0f}s)")

# ---- sanity: LOO rig must differ from the leaky rig ----
blob = build_model_identity(cfg, ds_val.subset, ds_val.targets)
n = 0
diff = []
for i in range(0, len(ds_val), 37):
    it = ds_val[i]
    m, a = ds_val.samples[i]
    from io_motion import load_target_curves
    curves = load_target_curves(ROOT / cfg.data_root / m, a, ds_val.targets[m],
                                fps=cfg.fps, T=cfg.T)
    cfg.rig_loo = False
    v_leak = ds_val._rig_vec(m, curves)
    cfg.rig_loo = True
    v_loo = ds_val._rig_vec(m, curves)
    if np.abs(v_leak - v_loo).sum() > 0:
        n += 1
    diff.append(float(np.abs(v_leak - v_loo).mean()))
print(f"[check] rig LOO vs leaky: mean|delta| {np.mean(diff):.4f}, "
      f"{n}/{len(diff)} samples changed")

# ---- sanity: char stats coverage + rest value spread ----
tab = build_char_param_stats(cfg, ds_val.subset, ds_val.targets)
print(f"[check] charstats LOO entries: {len(tab)}")
miss = sum(1 for s in ds_val.samples if s not in tab)
print(f"[check] val samples WITHOUT char stats: {miss}/{len(ds_val.samples)}")
rest = []
for i in range(0, len(ds_val), 17):
    it = ds_val[i]
    cs = it.get("cstats")
    if cs is not None and len(cs):
        rest.append(cs[:, 0])
rest = np.concatenate(rest)
print(f"[check] rest column: n={rest.size} nonzero={int((np.abs(rest)>1e-8).sum())} "
      f"mean={rest.mean():.4f} std={rest.std():.4f}")
print("[done] caches ready:",
      sorted(p.name for p in (ROOT / "outputs").glob("*v3*.pkl"))
      + sorted(p.name for p in (ROOT / "outputs").glob("charstats_loo*.pkl")))
