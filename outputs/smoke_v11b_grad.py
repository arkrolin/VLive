"""V11b: verify the ZERO-INIT bank branch actually OPENS during training.

At init every gradient in the bank branch is zero, because BOTH the residual
head and the bank branch are zero-init (out == exem, so nothing upstream of the
head receives gradient). That is the same mechanism the existing zero-init head
uses and is not a bug - but it MUST un-block after the first step, otherwise
V11b would silently train the baseline for 70 epochs.

This test runs a few optimiser steps on real data and asserts that
  * head.weight          leaves zero   (step 1)
  * in_proj 3rd channel  leaves zero   (step 1, via d loss/d h)
  * bank_gate            leaves zero   (step 1, via d loss/d tok)
  * bank_q / bank_curve_enc leave zero (step 2+, once gate != 0)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))

from io_motion import set_action_map                                # noqa: E402
from dataset import Live2DDataset, collate                          # noqa: E402
from model import Live2DModel                                       # noqa: E402

amap = json.loads((ROOT / "outputs" / "action_semantic_map.json").read_text(encoding="utf-8"))
rev = {}
for g, names in amap.get("groups", {}).items():
    for n in names:
        rev[str(n).lower()] = g
set_action_map(rev)

cfg = SimpleNamespace(
    whitelist_path=ROOT / "outputs" / "model_whitelist_all.json",
    gen_mask_path=ROOT / "outputs" / "gen_mask_all.json",
    data_root=ROOT / "data" / "all",
    subset_models=580, T=48, fps=30.0, rig_sig_dim=96, max_tokens=128,
    action_map_path="outputs/action_semantic_map.json",
    dedup_skip_path="outputs/dedup_skip_all.json",
    val_holdout_path="outputs/val_holdout_all.json",
    cache_tag="v9all", eval_shared_min_models=5,
    action_cond="id", action_name_max_len=32, val_frac=0.12, seed=1234,
    n_layers=2, d_model=128, n_heads=4, mlp_ratio=4, dropout=0.0,
    name_dropout=0.0, k_exemplars=3, num_diff_steps=1000,
    beta_schedule="cosine", gen_mode="regress",
    rig_loo=True, char_stats="mlp", bank_cond="attn", bank_k=8,
)

ds = Live2DDataset(cfg, split="val")
dev = torch.device("cuda:6")
torch.manual_seed(0)
mdl = Live2DModel(cfg, ds.word2idx, len(ds.action2idx), cfg.max_tokens,
                  ds.action_char2idx, getattr(ds, "param2idx", None)).to(dev)
opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
b = collate([ds[i] for i in range(6)], cfg.max_tokens)


def step():
    B, n, T = b["target"].shape
    lo = torch.zeros(B, n, device=dev)
    sp = torch.ones(B, n, device=dev)
    bank = (b["bank"].to(dev) - lo[:, None, :, None]) / sp[:, None, :, None]
    x0 = b["target"].to(dev)
    out = mdl(torch.zeros(B, n, T, device=dev),
              torch.zeros(B, dtype=torch.long, device=dev),
              b["names"], b["rig"].to(dev), b["action_id"].to(dev),
              b["token_mask"].to(dev), b["exem"].to(dev), training=True,
              cstats=b["cstats"].to(dev), span=sp, bank=bank,
              bank_mask=b["bank_mask"].to(dev), bank_act=b["bank_act"].to(dev))
    loss = ((out - x0) ** 2 * b["token_mask"].to(dev).unsqueeze(-1)).sum()
    loss = loss / b["token_mask"].to(dev).unsqueeze(-1).sum().clamp(min=1)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def mag(name):
    for nm, p in mdl.named_parameters():
        if nm == name:
            return (p.detach().abs().sum().item()
                    if p.ndim else abs(p.detach().item()))
    return float("nan")


WATCH = ["head.weight", "in_proj.weight", "bank_gate", "bank_q.weight",
         "bank_curve_enc.weight", "char_stats_enc.weight"]
print(f"{'step':>5s} " + " ".join(f"{w.split('.')[0][:12]:>14s}" for w in WATCH))
for s in range(6):
    ls = step()
    print(f"{s:5d} " + " ".join(f"{mag(w):14.3e}" for w in WATCH) + f"   loss={ls:.5f}")

# in_proj 3rd channel specifically
w = dict(mdl.named_parameters())["in_proj.weight"]
print(f"\nin_proj channel magnitudes: x_t={w[:, 0].abs().sum():.3e}  "
      f"exem={w[:, 1].abs().sum():.3e}  bank={w[:, 2].abs().sum():.3e}")
ok = (mag("head.weight") > 0 and w[:, 2].abs().sum() > 0
      and mag("bank_gate") > 0 and mag("bank_q.weight") > 0
      and mag("bank_curve_enc.weight") > 0 and mag("char_stats_enc.weight") > 0)
print("[PASS] bank branch opens" if ok else "[FAIL] branch stayed dead")
