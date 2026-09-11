"""Does V11b actually USE the motion bank, or is the gain coming from cstats?

Runs the same val subset three times on one checkpoint:
    full      - cstats + bank (what was trained)
    no-bank   - cstats only, bank channels zeroed + mask off
    no-cstats - bank only, cstats zeroed
If "full" beats both ablations, both branches carry real signal. If "no-bank"
matches "full", the bank is dead weight and the win is all cstats.

Zeroing is the honest test: the branches are zero-initialised, so a zeroed
input is exactly the model's own starting point for that branch.

Usage
-----
    .venv/bin/python outputs/diag_bank_ablation.py abl_V11b_bank --ckpt best --device cuda:5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/work/nlp/xjzhao13/lijie_llama/VLive")
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))
sys.path.insert(0, str(ROOT / "outputs"))

from train import recon_metrics, collate  # noqa: E402
import eval_ckpt  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


class ZeroBank:
    """Wrap a collate fn and blank the bank tensors (and their mask)."""

    def __init__(self, cfg, drop_bank=False, drop_cstats=False):
        self.cfg, self.db, self.dc = cfg, drop_bank, drop_cstats

    def __call__(self, batch):
        out = collate(batch, self.cfg.max_tokens)
        if self.db:
            for k in ("bank", "bank_mask", "bank_act"):
                if k in out and out[k] is not None:
                    b = out[k]
                    out[k] = (torch.zeros_like(b) if k == "bank" else b * 0)
        if self.dc and out.get("cstats") is not None:
            out["cstats"] = torch.zeros_like(out["cstats"])
        return out


def score(ctx, device, drop_bank, drop_cstats, tag):
    cfg = ctx["cfg"]
    ds = ctx["loader"].dataset
    loader = DataLoader(ds, batch_size=16, shuffle=False,
                        collate_fn=ZeroBank(cfg, drop_bank, drop_cstats),
                        num_workers=0)
    # recon_metrics already returns per-instance MEANS (it divides by cnt).
    out = recon_metrics(ctx["model"], loader, device, cfg,
                        ctx["per_lo"], ctx["per_hi"], ctx["g_lo"], ctx["g_hi"],
                        ctx["alphabar"], n_samples=ctx["cap"])
    print(f"{tag:12s} abs_mae={out['abs']:.4f}  exem={out['exem_abs']:.4f}  "
          f"rel_f={out['rel_f']:.5f}  exem_rel_f={out['exem_rel_f']:.5f}")
    return float(out["abs"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--device", default="cuda:5")
    args = ap.parse_args()

    device = torch.device(args.device)
    ctx = eval_ckpt.load_ctx(args.run, device, 0, args.ckpt, 0)
    print(f"cab: bank_cond={getattr(ctx['cfg'],'bank_cond','?')} "
          f"char_stats={getattr(ctx['cfg'],'char_stats','?')}\n")

    full = score(ctx, device, False, False, "full")
    nb = score(ctx, device, True, False, "no-bank")
    nc = score(ctx, device, False, True, "no-cstats")

    print()
    print(f"bank contributes   : {full:.4f} vs {nb:.4f}  ->  "
          f"{'BANK IS USED' if nb > full * 1.01 else 'bank looks dead'}")
    print(f"cstats contributes : {full:.4f} vs {nc:.4f}  ->  "
          f"{'CSTATS IS USED' if nc > full * 1.01 else 'cstats looks dead'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
