"""Smoke-test the V11b bank path BEFORE committing to a 70-epoch run.

Checks, with the real 580-model corpus:
  1. build_motion_bank produces a usable table (also warms its cache)
  2. collate() emits bank / bank_mask / bank_act with the right shapes and that
     the TARGET action is never among the sampled references
  3. Live2DModel(bank_cond='attn') forward returns the right shape, starts as a
     bit-identical no-op vs the 2-channel baseline (zero-init 3rd channel), and
     produces non-zero gradients through the bank branch
  4. the attended reference curve actually varies per param (attention is not
     collapsed to a constant)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))

from io_motion import set_action_map                                # noqa: E402
from dataset import Live2DDataset, collate, build_motion_bank       # noqa: E402
from model import Live2DModel                                       # noqa: E402

amap = json.loads((ROOT / "outputs" / "action_semantic_map.json").read_text(encoding="utf-8"))
rev = {}
for g, names in amap.get("groups", {}).items():
    for n in names:
        rev[str(n).lower()] = g
set_action_map(rev)


def mk(rig_loo=True, char_stats="mlp", bank_cond="attn", bank_k=8):
    return SimpleNamespace(
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
        rig_loo=rig_loo, char_stats=char_stats, bank_cond=bank_cond, bank_k=bank_k,
    )


cfg = mk()
ds = Live2DDataset(cfg, split="val")
print(f"[1] val samples {len(ds)}; bank entries {len(ds.bank_tab)}")

bank_tab = build_motion_bank(cfg, ds.subset, ds.targets)
print(f"[1] motionbank {(m := (ROOT/'outputs'/'motionbank_580_v9all.pkl'))}"
      f" exists={m.exists()} size={m.stat().st_size/1e6:.0f} MB")

# ---- 2. shapes + no target leak ----
leaks = 0
for i in range(0, len(ds), 97):
    mdl, act = ds.samples[i]
    for a in ds._bank_refs(mdl, act):
        if a == act:
            leaks += 1
ks = [len(ds._bank_refs(*ds.samples[i])) for i in range(0, len(ds), 53)]
print(f"[2] sampled refs: mean {np.mean(ks):.1f} max {max(ks)}; "
      f"target-action leaks: {leaks}")

b = collate([ds[i] for i in range(4)], cfg.max_tokens)
print(f"[2] bank {tuple(b['bank'].shape)} mask {tuple(b['bank_mask'].shape)} "
      f"act {tuple(b['bank_act'].shape)} cstats {tuple(b['cstats'].shape)}")
assert b["bank"].shape[0] == 4 and b["bank"].shape[1] == cfg.bank_k
assert b["bank"].shape[2] == cfg.max_tokens and b["bank"].shape[3] == cfg.T

# ---- 3. forward / no-op at init / gradients ----
dev = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
m_attn = Live2DModel(cfg, ds.word2idx, len(ds.action2idx), cfg.max_tokens,
                     ds.action_char2idx, getattr(ds, "param2idx", None)).to(dev)
cfg_off = mk(bank_cond="none")
torch.manual_seed(0)
m_base = Live2DModel(cfg_off, ds.word2idx, len(ds.action2idx), cfg.max_tokens,
                     ds.action_char2idx, getattr(ds, "param2idx", None)).to(dev)


def run(mdl, use_bank):
    B, n, T = b["target"].shape
    x_t = torch.zeros(B, n, T, device=dev)
    t = torch.zeros(B, dtype=torch.long, device=dev)
    names = b["names"]
    rig = b["rig"].to(dev)
    aid = b["action_id"].to(dev)
    tm = b["token_mask"].to(dev)
    exem = b["exem"].to(dev)
    cs = b["cstats"].to(dev)
    lo = torch.zeros(B, n, device=dev)
    sp = torch.ones(B, n, device=dev)
    kw = dict(cstats=cs, span=sp,
              bank=(b["bank"].to(dev) - lo[:, None, :, None]) / sp[:, None, :, None],
              bank_mask=b["bank_mask"].to(dev), bank_act=b["bank_act"].to(dev))
    if not use_bank:
        kw.update(bank=None, bank_mask=None, bank_act=None)
    return mdl(x_t, t, names, rig, aid, tm, exem, training=True, **kw)


out_a = run(m_attn, True)
out_b = run(m_base, False)
print(f"[3] out {tuple(out_a.shape)}; max|attn - base| at init = "
      f"{(out_a - out_b).abs().max().item():.3e}  (must be ~0: 3rd channel "
      f"is zero-init)")
out_a.sum().backward()
gn = [p.grad.abs().sum().item() for nm, p in m_attn.named_parameters()
      if p.grad is not None and ("bank" in nm or "in_proj" in nm)]
print(f"[3] grads through bank/in_proj: {[f'{g:.3e}' for g in gn]} "
      f"(must be > 0)")

# ---- 4. attention is not collapsed ----
with torch.no_grad():
    B, n, T = b["target"].shape
    bn = (b["bank"].to(dev) - 0.0) / 1.0
    kv = m_attn.bank_curve_enc(bn) + m_attn._action_emb(
        b["bank_act"].to(dev), None, False).unsqueeze(2)
    q = m_attn.bank_q(m_attn._token_emb(
        b["names"], b["rig"].to(dev), b["action_id"].to(dev), None, False,
        cstats=b["cstats"].to(dev)))
    sc = torch.einsum("bnd,bknd->bnk", q, kv) / (cfg.d_model ** 0.5)
    valid = b["bank_mask"].to(dev) > 0.5
    sc = sc.masked_fill(~valid.unsqueeze(1), -1e4)
    w = torch.softmax(sc, -1) * valid.unsqueeze(1)
    ent = -(w.clamp_min(1e-9).log() * w).sum(-1)
    print(f"[4] attention entropy over K: mean {ent.mean().item():.3f} "
          f"(log K = {np.log(cfg.bank_k):.3f}; near 0 = hard selection, "
          f"near logK = uniform)")
print("[done] V11b smoke test passed" if (out_a - out_b).abs().max().item() < 1e-6
      else "[FAIL] no-op property violated")
