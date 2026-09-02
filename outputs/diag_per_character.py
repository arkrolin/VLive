"""Per-character breakdown of model vs exem prior on the held-out val set.

Why this exists
---------------
Aggregate numbers hid a contradiction:

  * biased 5-character prefix (64 samples) : exem 2.4405, model 1.1189  -> +54%
  * full 34-character val set  (561 samples): exem 1.3453, model 1.2630  -> +6.1%

Back-solving the remaining 29 characters gives exem ~= 1.204 vs model ~= 1.281,
i.e. the model would be *worse* than its own prior there. If true, "the model
beats the prior" is an artefact of a handful of characters, and the honest
generalisation claim collapses.

This script settles it: it scores EVERY val character separately, so we can see
how many of the 34 held-out characters the model actually wins on.

Usage
-----
    .venv/bin/python outputs/diag_per_character.py abl_H_regress285_ep45 \
        [--device cuda:3] [--ckpt best] [--min_samples 4] [--residual_scale 1.0]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, SequentialSampler, Subset  # noqa: E402

from config import PipelineConfig  # noqa: E402
from dataset import Live2DDataset, collate  # noqa: E402
from model import Live2DModel  # noqa: E402
from train import compute_range, recon_metrics  # noqa: E402

RUNS = ROOT / "outputs" / "train_runs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run directory name under outputs/train_runs/")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--ckpt", type=str, default="best", choices=["best", "latest"])
    ap.add_argument("--min_samples", type=int, default=1,
                    help="skip characters with fewer than this many val samples "
                         "(a single sample is far too noisy to rank)")
    ap.add_argument("--residual_scale", type=float, default=1.0,
                    help="rewrite the prediction as exem + w*residual")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = RUNS / args.run / f"ckpt_{args.ckpt}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = PipelineConfig()
    cfg.__dict__.update(sd.get("cfg", {}))

    train_ds = Live2DDataset(cfg, split="train")
    val_ds = Live2DDataset(cfg, split="val")
    g_lo, g_hi, per_lo, per_hi = compute_range(train_ds, n=400)

    model = Live2DModel(cfg, train_ds.word2idx, len(train_ds.action2idx),
                        cfg.max_tokens, train_ds.action_char2idx)
    model.to(device)
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    if missing:
        print(f"  strict=False: missing {sorted({m.split('.')[0] for m in missing})} "
              f"(untrained in the original run - faithful)")
    if unexpected:
        print(f"  unexpected: {sorted({u.split('.')[0] for u in unexpected})}")
    model.eval()

    alphabar = model.ddpm_schedule(
        cfg.num_diff_steps, cosine=(cfg.beta_schedule == "cosine"))[2].to(device)

    # group val indices by held-out character
    by_model: dict[str, list[int]] = defaultdict(list)
    for i, (m, _a) in enumerate(val_ds.samples):
        by_model[m].append(i)

    rows = []
    for name in sorted(by_model):
        idx = by_model[name]
        if len(idx) < args.min_samples:
            continue
        loader = DataLoader(
            Subset(val_ds, idx), batch_size=4, sampler=SequentialSampler(idx),
            collate_fn=lambda b: collate(b, cfg.max_tokens), num_workers=0)
        with torch.no_grad():
            m = recon_metrics(model, loader, device, cfg, per_lo, per_hi,
                              g_lo, g_hi, alphabar, n_samples=len(idx), steps=50,
                              residual_scale=args.residual_scale)
        rows.append({
            "char": name,
            "n": len(idx),
            "abs": m["abs"], "exem": m["exem_abs"],
            "rel_f": m["rel_f"], "exem_rel_f": m["exem_rel_f"],
        })

    if not rows:
        print("no characters had enough val samples; lower --min_samples")
        return 1

    w = args.residual_scale
    print(f"\n=== per-character breakdown | {args.run} | ckpt_{args.ckpt} | w={w} ===")
    print(f"{'character':<34}{'n':>5}{'abs':>9}{'exem':>9}{'vs exem':>10}"
          f"{'rel_f':>9}{'exem_rel_f':>12}")
    print("-" * 88)
    for r in sorted(rows, key=lambda x: (x["abs"] - x["exem"]) / x["exem"]):
        gain = (r["exem"] - r["abs"]) / r["exem"] * 100 if r["exem"] else float("nan")
        print(f"{r['char']:<34}{r['n']:>5}{r['abs']:>9.4f}{r['exem']:>9.4f}"
              f"{gain:>9.1f}%{r['rel_f']:>9.4f}{r['exem_rel_f']:>12.4f}")

    # aggregate: sample-weighted (what the reported number is) and per-character
    tot_n = sum(r["n"] for r in rows)
    w_abs = sum(r["abs"] * r["n"] for r in rows) / tot_n
    w_exem = sum(r["exem"] * r["n"] for r in rows) / tot_n
    w_rel = sum(r["rel_f"] * r["n"] for r in rows) / tot_n
    w_exem_rel = sum(r["exem_rel_f"] * r["n"] for r in rows) / tot_n
    wins_abs = sum(1 for r in rows if r["abs"] < r["exem"])
    wins_rel = sum(1 for r in rows if r["rel_f"] < r["exem_rel_f"])

    print("-" * 88)
    print(f"characters scored: {len(rows)}   val samples: {tot_n}")
    print(f"sample-weighted  abs_mae {w_abs:.4f} vs exem {w_exem:.4f} "
          f"({(w_exem - w_abs) / w_exem * 100:+.1f}%)")
    print(f"sample-weighted  rel_f   {w_rel:.4f} vs exem {w_exem_rel:.4f} "
          f"({(w_exem_rel - w_rel) / w_exem_rel * 100:+.1f}%)")
    print(f"model beats prior on {wins_abs}/{len(rows)} characters by abs_mae "
          f"({wins_abs / len(rows) * 100:.0f}%)")
    print(f"model beats prior on {wins_rel}/{len(rows)} characters by rel_f   "
          f"({wins_rel / len(rows) * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
