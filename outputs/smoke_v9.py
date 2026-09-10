"""Smoke-test the two V9 arms' corpora before spending GPU time.

Builds the train/val datasets for each arm (which also warms the exem/ident
caches) and prints exactly the numbers the gate is written against:
train samples, val characters, val samples, action vocabulary size.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path("/root/work/nlp/xjzhao13/lijie_llama/VLive")
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))

from config import PipelineConfig  # noqa: E402
from dataset import Live2DDataset   # noqa: E402

ARMS = {
    "arm1_std_v9": dict(
        data_root="standrad-live-2d",
        whitelist_path="outputs/model_whitelist.json",
        gen_mask_path="outputs/gen_mask.json",
        val_holdout_path="outputs/val_holdout_std.json",
        dedup_skip_path="outputs/dedup_skip_std.json",
        subset_models=285,
        cache_tag="v9std",
    ),
    "arm2_all_v9": dict(
        data_root="data/all",
        whitelist_path="outputs/model_whitelist_all.json",
        gen_mask_path="outputs/gen_mask_all.json",
        val_holdout_path="outputs/val_holdout_all.json",
        dedup_skip_path="outputs/dedup_skip_all.json",
        subset_models=580,
        cache_tag="v9all",
    ),
}

COMMON = dict(action_map_path="outputs/action_semantic_map.json")


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, kw in ARMS.items():
        if only and only not in name:
            continue
        t0 = time.time()
        cfg = PipelineConfig()
        for k, v in kw.items():
            setattr(cfg, k, ROOT / v if k.endswith(("_root", "_path")) else v)
        for k, v in COMMON.items():
            setattr(cfg, k, v)
        print(f"\n=== {name} ===", flush=True)
        tr = Live2DDataset(cfg, split="train")
        va = Live2DDataset(cfg, split="val")
        va_all = Live2DDataset(cfg, split=None)
        val_chars = len({m for m, _ in va.samples})
        print(f"  models(subset)   : {len(tr.subset)}")
        print(f"  train samples    : {len(tr.samples)}")
        print(f"  val   samples    : {len(va.samples)}  (chars={val_chars})")
        print(f"  val+train        : {len(va_all.samples)}")
        print(f"  action vocab     : {len(tr.action2idx)}")
        print(f"  param vocab      : {len(tr.param2idx)}")
        print(f"  overlap train/val: "
              f"{len(set(va.samples) & set(tr.samples))}")
        # how many val samples survive eval_shared_min_models=5
        k = cfg.eval_shared_min_models
        kept = [s for s in va.samples if tr.action_nmodels.get(s[1], 0) >= k]
        print(f"  val w/ action in >={k} models : {len(kept)}")
        print(f"  build time       : {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
