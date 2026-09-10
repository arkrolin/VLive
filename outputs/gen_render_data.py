"""Export val-set predictions in RAW parameter units so they can be rendered.

eval_ckpt.py only reports aggregate metrics; to actually LOOK at the output we
need per-sample curves. This replays recon_metrics' forward pass but keeps the
curves instead of averaging them, and undoes the range normalisation so the
numbers are the same units the Live2D runtime expects
(param = x0_hat * span + lo).

Also dumps a lightweight manifest (character, action, energies, abs error) so
the renderer can pick representative samples without loading every curve.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

ROOT = Path("/root/work/nlp/xjzhao13/lijie_llama/VLive")
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))
sys.path.insert(0, str(ROOT / "outputs"))

from train import mean_range_vecs, collate  # noqa: E402
import eval_ckpt  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="abl_R_all_v9")
    ap.add_argument("--ckpt", default="best", choices=["best", "latest"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs/render_data")
    ap.add_argument("--limit", type=int, default=0, help="0 = all val samples")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ctx = eval_ckpt.load_ctx(args.run, device, 0, args.ckpt, 0)
    cfg, model = ctx["cfg"], ctx["model"]
    val_ds = ctx["loader"].dataset
    per_lo, per_hi, g_lo, g_hi = ctx["per_lo"], ctx["per_hi"], ctx["g_lo"], ctx["g_hi"]
    min_span = max(g_hi - g_lo, 1.0) * 0.02

    loader = DataLoader(val_ds, batch_size=4, sampler=SequentialSampler(val_ds),
                        collate_fn=lambda b: collate(b, cfg.max_tokens), num_workers=0)

    samples = val_ds.samples
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    items, manifest = [], []
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["target"].to(device)
            rig = batch["rig"].to(device)
            action_id = batch["action_id"].to(device)
            ac = batch.get("action_chars")
            action_chars = ac.to(device) if ac is not None else None
            token_mask = batch["token_mask"].to(device)
            names = batch["names"]
            B, n, T = target.shape
            lo_v, hi_v = mean_range_vecs(
                names, per_lo, per_hi, g_lo, g_hi, cfg.max_tokens,
                fb_span=getattr(cfg, "fb_span", None))
            lo_t = torch.from_numpy(lo_v).to(device)
            hi_t = torch.from_numpy(hi_v).to(device)
            span_t = (hi_t - lo_t).clamp(min=min_span)
            exem_s = ((batch["exem"].to(device) - lo_t.unsqueeze(-1))
                      / span_t.unsqueeze(-1)).clamp(-0.5, 1.5)
            x0 = ((target - lo_t.unsqueeze(-1)) / span_t.unsqueeze(-1)).clamp(-0.5, 1.5)

            t0 = torch.zeros(B, device=device, dtype=torch.long)
            x0_hat = model(torch.zeros_like(x0), t0, names, rig, action_id,
                           token_mask, exem_s, training=False,
                           action_chars=action_chars, span=span_t)
            x0_hat = x0_hat.clamp(-0.5, 1.5)

            span_np = span_t.cpu().numpy()
            lo_np = lo_t.cpu().numpy()
            pred_np = x0_hat.cpu().numpy()
            tgt_np = x0.cpu().numpy()
            exm_np = exem_s.cpu().numpy()
            mask_np = token_mask.cpu().numpy()

            for b in range(B):
                idx = n_done
                n_done += 1
                if args.limit and n_done > args.limit:
                    break
                m = mask_np[b] > 0.5
                nm = [names[b][j] for j in range(n) if m[j]]
                if not nm:
                    continue
                sp, lo = span_np[b][m], lo_np[b][m]
                pred = pred_np[b][m] * sp[:, None] + lo[:, None]
                tgt = tgt_np[b][m] * sp[:, None] + lo[:, None]
                exm = exm_np[b][m] * sp[:, None] + lo[:, None]
                model_id, action = samples[idx]
                ptp = float(np.abs(tgt).max())
                items.append(dict(
                    idx=idx, model=model_id, action=action, names=nm,
                    pred=pred.astype(np.float32), target=tgt.astype(np.float32),
                    exem=exm.astype(np.float32), T=int(T),
                ))
                manifest.append(dict(
                    idx=idx, model=model_id, action=action, n_params=len(nm),
                    energy=ptp,
                    span_sum=float(np.abs(tgt.max(axis=1) - tgt.min(axis=1)).sum()),
                    err_pred=float(np.abs(pred - tgt).mean()),
                    err_exem=float(np.abs(exm - tgt).mean()),
                ))
            if args.limit and n_done > args.limit:
                break

    out = out_dir / f"{args.run}_{args.ckpt}.pkl"
    with open(out, "wb") as f:
        pickle.dump(dict(run=args.run, ckpt=args.ckpt,
                         epoch=ctx["sd"].get("epoch"),
                         n=len(items), items=items, manifest=manifest), f,
                    protocol=4)
    mf = manifest
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB, {len(items)} samples)")
    ep = np.array([x["err_pred"] for x in mf])
    ee = np.array([x["err_exem"] for x in mf])
    print(f"mean |pred-target| = {ep.mean():.4f}   |exem-target| = {ee.mean():.4f}")
    top = sorted(mf, key=lambda x: -x["span_sum"])[:12]
    print("\nlargest-motion samples (good render candidates):")
    for x in top:
        print(f"  idx={x['idx']:4d} {x['model']:34s} {x['action'][:28]:28s} "
              f"P={x['n_params']:3d} span={x['span_sum']:8.2f} "
              f"err {x['err_pred']:.3f} vs exem {x['err_exem']:.3f}")


if __name__ == "__main__":
    main()
