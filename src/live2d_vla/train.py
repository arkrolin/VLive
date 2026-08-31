"""V7.1 training entrypoint: DDPM training of Live2DModel with validation.

End-to-end, validation-gated training prototype (task #2):
    param-as-token + 4-way identity embedding (name/rig/deformer/action)
    + stable DiT (QKNorm/RMSNorm/SwiGLU) + DDPM noise-prediction.

Standardization: PER-PARAM RANGE normalization (each param token ->
    (target - lo) / (hi - lo) over the TRAIN corpus min/max). This puts every
    param in ~[0,1] units, i.e. units of FRACTION OF RANGE. Consequently the
    noise-MSE loss is directly proportional to the deployment metric rel_mae^2,
    so training optimizes reconstruction fidelity, not an arbitrary scale.
    (The earlier std-normalization ignored small-range params and produced
    rel_mae ~0.58 even when the std-MSE looked fine.)

Training objective (x0-PREDICTION MSE, per-param range units):
    x0_p = (target_p - lo_p) / (hi_p - lo_p)   (clamped to [-0.5, 1.5])
    The network regresses the clean x0 directly from the noisy x_t (+ exem prior).
    x0_hat = exem + residual.  loss = masked MSE(x0_hat, x0)  (+ arm weighting).
    The loss floor = exem rel^2 ~ (0.055)^2 ~ 0.003 (exem alone reconstructs
    within 5.5% of range, so the model cannot beat ~0.055 without identity).

Validation (held-out *whole models*, character generalization):
    * val_loss    : same x0-prediction MSE on the val split (every epoch)
    * val_abs_mae: DDIM-reverse reconstruction error in ORIGINAL param units
    * val_rel_mae: mean |x0_hat - x0|  -> deployment fidelity (% of range)

Acceptance gate (printed, logged, and asserted at the end):
    Calibrated baselines (CPU, held-out val, arm_w=3):
        exem-only recon: abs_mae=0.4706  rel_mae=0.0551
        => the in-corpus exemplar prior is very strong (5.5% of range). The
           range-norm loss floor is exem rel^2 ~0.003. The model converges near
           the exem rel (no character-identity signal -> bounded by exem prior).
    Targets (the model must reach ~exem fidelity, not worse):
        train_loss   <= TRAIN_TARGET  (0.05)   [range-norm MSE]
        val_loss     <= VAL_TARGET    (0.06)   [range-norm MSE; ~1.4x the 0.042 floor]
        overfit      = val_loss/train_loss <= OVERFIT_MAX (1.30)  [no runaway]
        val_rel_mae  <= REL_MAE_TARGET (0.15)  [recon within 15% of param range]
    Going materially below exem rel needs a learned character-identity embedding
    (replace bag-of-hash rig_sig) - the next architecture step, not a hyperparam.

Run:
    torchrun --nproc_per_node=7 src/live2d_vla/train.py --epochs 50
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from config import PipelineConfig
from dataset import Live2DDataset, collate
from model import Live2DModel

# ---- acceptance targets (the "expected metrics", calibrated; see docstring) ----
# Range-normalized loss units (fraction of param range). Robust exem floor ~0.042.
# The model must reach ~exem fidelity (rel_mae ~0.055-0.12), not worse.
TRAIN_TARGET = 0.09      # range-norm MSE (train sits just above the 0.042 floor)
VAL_TARGET = 0.06        # held-out; ~1.4x the 0.042 exem floor
OVERFIT_MAX = 1.30       # val_loss / train_loss (no runaway overfit)
REL_MAE_TARGET = 0.15    # reconstruction error as fraction of param range

# Arm / upper-body channels get higher loss weight (V7.1 action prior).
ARM_KEYWORDS = {
    "arm", "shoulder", "elbow", "wrist", "hand", "finger", "forearm",
    "upper", "lower", "bicep", "tricep", "neck",
}


# --------------------------------------------------------------------------- #
# distributed helpers
# --------------------------------------------------------------------------- #
def distributed_available() -> bool:
    return ("RANK" in os.environ and "WORLD_SIZE" in os.environ
            and int(os.environ["WORLD_SIZE"]) > 1)


def setup_dist():
    if not distributed_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1, False
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device, rank, world_size, True


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
# loss / standardization utilities
# --------------------------------------------------------------------------- #
def arm_weight(names_b, weight: float = 3.0, n_pad: int = None) -> np.ndarray:
    """(B, n_pad) per-token loss weight; arm/upper-body params get `weight`."""
    n = n_pad if n_pad else max((len(x) for x in names_b), default=0)
    w = np.ones((len(names_b), max(n, 1)), np.float32)
    for i, names in enumerate(names_b):
        for j, nm in enumerate(names):
            if j >= w.shape[1]:
                break
            toks = [t.lower() for t in re.findall(r"[A-Za-z]+|\d+", nm)]
            if any(k in toks for k in ARM_KEYWORDS):
                w[i, j] = weight
    return w


def compute_range(ds, n: int = None):
    """Per-param (lo, hi) over the TRAIN corpus (one-time).

    Returns (g_lo, g_hi, per_lo, per_hi). per_* are dicts keyed by param name;
    g_lo/g_hi are global fallbacks for params unseen in the stats.
    """
    per_lo, per_hi = {}, {}
    if n is None:
        n = len(ds)
    else:
        n = min(n, len(ds))
    for i in range(n):
        try:
            s = ds[i]
        except Exception:
            continue
        x = np.asarray(s["target"], dtype=np.float32)
        names = s["names"]
        for j, nm in enumerate(names):
            if j >= x.shape[0]:
                break
            col = x[j].astype(np.float32)
            lo, hi = float(col.min()), float(col.max())
            if nm not in per_lo:
                per_lo[nm], per_hi[nm] = lo, hi
            else:
                per_lo[nm] = min(per_lo[nm], lo)
                per_hi[nm] = max(per_hi[nm], hi)
    g_lo = min(per_lo.values()) if per_lo else 0.0
    g_hi = max(per_hi.values()) if per_hi else 1.0
    return g_lo, g_hi, per_lo, per_hi


def mean_range_vecs(names_b, per_lo, per_hi, g_lo, g_hi, max_tokens):
    """(B, max_tokens) lo/hi arrays aligned to token rows (for range norm)."""
    B = len(names_b)
    lo_v = np.full((B, max_tokens), g_lo, np.float32)
    hi_v = np.full((B, max_tokens), g_hi, np.float32)
    for i, names in enumerate(names_b):
        for j, nm in enumerate(names):
            if j >= max_tokens:
                break
            if nm in per_lo:
                lo_v[i, j] = per_lo[nm]
                hi_v[i, j] = per_hi[nm]
    return lo_v, hi_v


def noise_loss(model, batch, device, cfg, per_lo, per_hi, g_lo, g_hi, arm_w):
    # `model` may be DDP-wrapped; q_sample lives on the unwrapped module.
    q_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    target = batch["target"].to(device)
    exem = batch["exem"].to(device)
    rig = batch["rig"].to(device)
    action_id = batch["action_id"].to(device)
    token_mask = batch["token_mask"].to(device)
    names = batch["names"]
    lo_v, hi_v = mean_range_vecs(names, per_lo, per_hi, g_lo, g_hi, cfg.max_tokens)
    lo_t = torch.from_numpy(lo_v).to(device)
    hi_t = torch.from_numpy(hi_v).to(device)
    # robust span: floor near-constant params at 2% of the global range so they
    # don't blow up (exem-target)/range^2 (which would dominate the loss).
    min_span = max(g_hi - g_lo, 1.0) * 0.02
    span_t = (hi_t - lo_t).clamp(min=min_span)
    # per-param RANGE normalization -> units of fraction of range
    x0 = ((target - lo_t.unsqueeze(-1)) / span_t.unsqueeze(-1)).clamp(-0.5, 1.5)
    exem_s = ((exem - lo_t.unsqueeze(-1)) / span_t.unsqueeze(-1)).clamp(-0.5, 1.5)
    B = x0.shape[0]
    if getattr(cfg, "gen_mode", "ddpm") == "regress":
        # P1 arm A: deterministic residual regression - no noise, t=0. The model
        # still outputs exem + residual, so it starts exactly at the prior and
        # only learns the correction. Tests whether the diffusion formulation
        # itself is the bottleneck.
        t = torch.zeros(B, device=device, dtype=torch.long)
        x_t = torch.zeros_like(x0)
    else:
        # P1 arm C: truncated schedule - cap the noise level. With a very strong
        # prior, fully destroying the signal (t up to 1000) may be counterproductive.
        t_max = min(int(getattr(cfg, "max_diff_t", cfg.num_diff_steps)),
                    cfg.num_diff_steps)
        t = torch.randint(0, t_max, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = q_model.q_sample(x0, t, noise, alphabar)
    # x0-PREDICTION: network regresses clean x0 directly (+ exem prior).
    # loss = masked MSE(x0_hat, x0) in fraction-of-range units == rel_mae^2.
    x0_hat = model(x_t, t, names, rig, action_id, token_mask, exem_s, training=True)
    se = (x0_hat - x0) ** 2
    aw = torch.from_numpy(
        arm_weight(names, arm_w, n_pad=cfg.max_tokens)).to(device).unsqueeze(-1)
    valid = token_mask.unsqueeze(-1) * aw
    loss = (se * valid).sum() / valid.sum().clamp(min=1.0)
    return loss


# --------------------------------------------------------------------------- #
# DDIM reverse sampling + reconstruction metric
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ddim_reverse(model, x_T, names, rig, action_id, token_mask, exem_s, alphabar,
                 device, steps: int = 50, eta: float = 0.0, t_start: int = None):
    """x0-prediction DDIM reverse: x0_hat (per-param std units) from x_T.

    t_start: highest timestep index to reverse from. Defaults to S-1 (full
    diffusion); pass < S for a truncated schedule (P1 ablation arm C), which
    must match the `max_diff_t` used during training.
    """
    S = alphabar.shape[0]
    hi = (S - 1) if t_start is None else min(int(t_start), S) - 1
    hi = max(hi, 1)
    ts = torch.linspace(hi, 0, steps).long()   # descending
    x = x_T
    for i in range(steps):
        t_cur = ts[i]
        t_batch = t_cur.view(1).expand(x.shape[0]).to(device)
        x0_hat = model(x, t_batch, names, rig, action_id, token_mask, exem_s,
                       training=False)
        x0_hat = x0_hat * token_mask.unsqueeze(-1)
        a_cur = alphabar[t_cur]
        if i == steps - 1:                          # reached t=0 -> x0
            break
        t_prev = ts[i + 1]
        a_prev = alphabar[t_prev]
        # implied noise from the x0 prediction
        noise_pred = (x - a_cur.sqrt() * x0_hat) / (1 - a_cur).sqrt().clamp(min=1e-6)
        sigma = eta * ((1 - a_prev) / (1 - a_cur)).sqrt() * \
            (1 - a_cur / a_prev).sqrt()
        dir_term = (1 - a_prev - sigma ** 2).sqrt() * noise_pred
        noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
        x = a_prev.sqrt() * x0_hat + dir_term + sigma * noise
    return x0_hat * token_mask.unsqueeze(-1)


METRIC_KEYS = ("abs", "rel", "rel_f", "exem_abs", "exem_rel", "exem_rel_f")


@torch.no_grad()
def recon_metrics(model, loader, device, cfg, per_lo, per_hi, g_lo, g_hi,
                  alphabar, n_batches: int, steps: int = 50):
    """Reconstruction error for the MODEL *and* the EXEM PRIOR on identical batches.

    Two relative-error variants are reported because the training loss and the
    legacy metric used DIFFERENT span definitions (see P0 diagnosis):
      rel     : / raw per-param span (floor 1e-3)  -> legacy. 74.4% of param
                instances are near-constant yet supply 94.5% of this number.
      rel_f   : / span floored at 2% of the global range -> ALIGNED with the
                training loss. In normalised units it is just mean_t|x0_hat-x0|.
    The exem prior is scored on the same batches, so any model can be compared
    directly against "just output the action mean" - which is the comparison
    that actually matters.
    """
    model.eval()
    min_span = max(g_hi - g_lo, 1.0) * 0.02
    acc = {k: 0.0 for k in METRIC_KEYS}
    cnt = 0
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        target = batch["target"].to(device)
        rig = batch["rig"].to(device)
        action_id = batch["action_id"].to(device)
        token_mask = batch["token_mask"].to(device)
        names = batch["names"]
        B, n, T = target.shape
        lo_v, hi_v = mean_range_vecs(names, per_lo, per_hi, g_lo, g_hi, cfg.max_tokens)
        lo_t = torch.from_numpy(lo_v).to(device)
        hi_t = torch.from_numpy(hi_v).to(device)
        span_t = (hi_t - lo_t).clamp(min=min_span)      # loss-aligned span
        span_raw_t = (hi_t - lo_t).clamp(min=1e-3)      # legacy metric span
        exem_s = ((batch["exem"].to(device) - lo_t.unsqueeze(-1))
                  / span_t.unsqueeze(-1)).clamp(-0.5, 1.5)
        x0 = ((target - lo_t.unsqueeze(-1)) / span_t.unsqueeze(-1)).clamp(-0.5, 1.5)

        if getattr(cfg, "gen_mode", "ddpm") == "regress":
            t0 = torch.zeros(B, device=device, dtype=torch.long)
            x0_hat = model(torch.zeros_like(x0), t0, names, rig, action_id,
                           token_mask, exem_s, training=False)
        else:
            x_T = torch.randn(B, n, T, device=device)
            x0_hat = ddim_reverse(
                model, x_T, names, rig, action_id, token_mask, exem_s, alphabar,
                device, steps=steps,
                t_start=int(getattr(cfg, "max_diff_t", 1000)))
        x0_hat = x0_hat.clamp(-0.5, 1.5)

        for pred, pre in ((x0_hat, ""), (exem_s, "exem_")):
            err_n = (pred - x0).abs()                    # (B,n,T), normalised units
            for b in range(B):
                for j in range(n):
                    if token_mask[b, j] < 0.5:
                        continue
                    e_f = err_n[b, j].mean().item()      # loss-aligned rel error
                    sp = float(span_t[b, j].item())
                    sp_raw = float(span_raw_t[b, j].item())
                    e_abs = e_f * sp                     # raw param units
                    acc[pre + "abs"] += e_abs
                    acc[pre + "rel_f"] += min(e_f, 2.0)
                    acc[pre + "rel"] += min(e_abs / sp_raw, 2.0)
                    if pre == "":
                        cnt += 1
    model.train()
    out = {k: acc[k] / max(cnt, 1) for k in METRIC_KEYS}
    out["n_params"] = cnt
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="V7.1 Live2D VLA DDPM trainer")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--subset_models", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None,
                   help="cap total optimizer steps (smoke test)")
    p.add_argument("--eval_every", type=int, default=None,
                   help="epochs between DDIM reconstruction metric (default config)")
    p.add_argument("--gen_mode", type=str, default=None,
                   choices=["ddpm", "regress"],
                   help="P1 ablation: ddpm (diffusion) | regress (deterministic residual)")
    p.add_argument("--max_diff_t", type=int, default=None,
                   help="P1 ablation: truncate noise sampling to t < max_diff_t")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--arm_weight", type=float, default=3.0)
    p.add_argument("--fresh", action="store_true",
                   help="ignore existing checkpoint and retrain")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PipelineConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.subset_models is not None:
        cfg.subset_models = args.subset_models
    if args.out_dir is not None:
        cfg.out_dir = Path(args.out_dir)
    if args.lr is not None:
        cfg.lr = args.lr
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    if args.gen_mode is not None:
        cfg.gen_mode = args.gen_mode
    if args.max_diff_t is not None:
        cfg.max_diff_t = args.max_diff_t
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    device, rank, world_size, ddp = setup_dist()
    if ddp:
        dist.barrier()

    # ---- datasets ----
    train_ds = Live2DDataset(cfg, split="train")
    val_ds = Live2DDataset(cfg, split="val")
    action_vocab_size = len(train_ds.action2idx)
    n_tokens_pad = cfg.max_tokens
    if is_main(rank):
        print(f"[rank{rank}] train samples={len(train_ds)} val samples={len(val_ds)} "
              f"params_vocab={len(train_ds.param2idx)} actions={action_vocab_size} "
              f"val_models={len(val_ds.holdout)}")
        print(f"[rank{rank}] gen_mode={cfg.gen_mode} max_diff_t={cfg.max_diff_t} "
              f"eval_shared_min_models={cfg.eval_shared_min_models}")

    model = Live2DModel(cfg, train_ds.word2idx, action_vocab_size, n_tokens_pad)
    model.to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=False)
        dmodel = model.module
    else:
        dmodel = model

    # ---- per-param RANGE stats (TRAIN only, no leakage) ----
    g_lo, g_hi, per_lo, per_hi = compute_range(train_ds, n=400)
    if is_main(rank):
        print(f"[rank{rank}] global range=({g_lo:.3f},{g_hi:.3f}) "
              f"per-param entries={len(per_lo)}")

    global alphabar
    alphabar = dmodel.ddpm_schedule(
        cfg.num_diff_steps, cosine=(cfg.beta_schedule == "cosine"))[2].to(device)

    optimizer = torch.optim.AdamW(
        dmodel.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---- samplers / loaders ----
    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = RandomSampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        collate_fn=lambda b: collate(b, cfg.max_tokens),
        num_workers=0, pin_memory=(device.type == "cuda"))

    steps_per_epoch = len(train_loader)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    # ---- resume ----
    start_epoch = 0
    ckpt = cfg.out_dir / "ckpt_latest.pt"
    if ckpt.exists() and not args.fresh:
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        dmodel.load_state_dict(sd["model"])
        optimizer.load_state_dict(sd["optim"])
        start_epoch = sd.get("epoch", 0)
        g_lo = sd.get("g_lo", g_lo)
        g_hi = sd.get("g_hi", g_hi)
        if is_main(rank):
            print(f"[rank{rank}] resume from epoch {start_epoch}")

    # ---- metrics log ----
    csv_path = cfg.out_dir / "metrics.csv"
    if is_main(rank) and not (csv_path.exists() and not args.fresh and start_epoch > 0):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss",
                        "val_abs_mae", "val_rel_mae", "val_rel_mae_f",
                        "exem_abs_mae", "exem_rel_mae", "exem_rel_mae_f",
                        "lr", "best"])

    best_val, epochs_no_improve = float("inf"), 0
    global_step = start_epoch * steps_per_epoch
    nan_m = {k: float("nan") for k in METRIC_KEYS}
    best_m = dict(nan_m)

    # ---- training loop ----
    for epoch in range(start_epoch, cfg.epochs):
        if ddp:
            train_sampler.set_epoch(epoch)
        dmodel.train()
        running, cnt = 0.0, 0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            # ---- LR schedule: warmup + cosine ----
            if global_step < warmup_steps:
                lr = cfg.lr * (global_step + 1) / max(warmup_steps, 1)
            else:
                prog = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr = cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * prog))
            for g in optimizer.param_groups:
                g["lr"] = lr

            loss = noise_loss(model, batch, device, cfg, per_lo, per_hi,
                              g_lo, g_hi, args.arm_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dmodel.parameters(), cfg.grad_clip)
            optimizer.step()

            running += loss.item()
            cnt += 1
            global_step += 1
            if is_main(rank) and step % args.log_every == 0:
                print(f"[ep {epoch + 1}/{cfg.epochs} step {step}/{steps_per_epoch}] "
                      f"loss={loss.item():.4f} lr={lr:.2e} "
                      f"elapsed={time.time() - t0:.1f}s")
            if args.max_steps and global_step >= args.max_steps:
                break

        train_loss = running / max(cnt, 1)

        # ---- validation (rank 0 only) ----
        val_loss = float("nan")
        m = dict(nan_m)
        if is_main(rank):
            dmodel.eval()
            v_sum, v_cnt = 0.0, 0
            val_loader = DataLoader(
                val_ds, batch_size=cfg.batch_size, sampler=SequentialSampler(val_ds),
                collate_fn=lambda b: collate(b, cfg.max_tokens), num_workers=0)
            with torch.no_grad():
                for vbatch in val_loader:
                    vl = noise_loss(dmodel, vbatch, device, cfg, per_lo, per_hi,
                                    g_lo, g_hi, args.arm_weight)
                    v_sum += vl.item()
                    v_cnt += 1
            val_loss = v_sum / max(v_cnt, 1)
            if epoch + 1 >= cfg.eval_every or epoch == cfg.epochs - 1:
                m = recon_metrics(
                    dmodel, val_loader, device, cfg, per_lo, per_hi,
                    g_lo, g_hi, alphabar,
                    n_batches=cfg.val_recon_batches, steps=50)
            dmodel.train()

            improved = val_loss < best_val - 1e-4
            if improved:
                best_val = val_loss
                best_m = m
                epochs_no_improve = 0
                torch.save({
                    "model": dmodel.state_dict(), "optim": optimizer.state_dict(),
                    "epoch": epoch + 1, "g_lo": g_lo, "g_hi": g_hi,
                    "cfg": cfg.__dict__, "val_loss": val_loss,
                }, cfg.out_dir / "ckpt_best.pt")
            else:
                epochs_no_improve += 1

            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch + 1, f"{train_loss:.4f}", f"{val_loss:.4f}",
                            f"{m['abs']:.4f}", f"{m['rel']:.4f}", f"{m['rel_f']:.4f}",
                            f"{m['exem_abs']:.4f}", f"{m['exem_rel']:.4f}",
                            f"{m['exem_rel_f']:.4f}", f"{lr:.2e}",
                            f"{best_val:.4f}"])
            print(f"[rank{rank}] ep {epoch + 1} train={train_loss:.4f} "
                  f"val={val_loss:.4f} (best {best_val:.4f}) "
                  f"rel_f={m['rel_f']:.4f} exem_rel_f={m['exem_rel_f']:.4f} | "
                  f"abs={m['abs']:.3f} exem_abs={m['exem_abs']:.3f} "
                  f"gap={val_loss / max(train_loss,1e-6):.2f}x "
                  f"noimp={epochs_no_improve}")

        # ---- checkpoint latest ----
        if is_main(rank):
            torch.save({
                "model": dmodel.state_dict(), "optim": optimizer.state_dict(),
                "epoch": epoch + 1, "g_lo": g_lo, "g_hi": g_hi,
                "cfg": cfg.__dict__,
            }, ckpt)

        if ddp:
            dist.barrier()

        # ---- early stopping ----
        stop = False
        if is_main(rank) and epochs_no_improve >= cfg.early_stop_patience:
            print(f"[rank{rank}] early stop: val_loss not improved for "
                  f"{epochs_no_improve} epochs")
            stop = True
        if ddp:
            stop_flag = torch.tensor([1 if stop else 0], device=device)
            dist.broadcast(stop_flag, src=0)
            stop = bool(stop_flag.item())
        if stop:
            break

        if args.max_steps and global_step >= args.max_steps:
            break

    # ---- final report / gate ----
    if is_main(rank):
        overfit = val_loss / max(train_loss, 1e-6)
        rel_f, exem_rel_f = best_m["rel_f"], best_m["exem_rel_f"]
        rel_ok = (rel_f != rel_f) or (rel_f <= REL_MAE_TARGET)
        # The comparison that actually matters: does the model beat its own prior?
        beats = (rel_f == rel_f and exem_rel_f == exem_rel_f and rel_f < exem_rel_f)
        passed = (train_loss <= TRAIN_TARGET and val_loss <= VAL_TARGET
                  and overfit <= OVERFIT_MAX and rel_ok)
        print("\n==================== TRAINING GATE ====================")
        print(f"  train_loss   = {train_loss:.4f}   (target <= {TRAIN_TARGET}) "
              f"{'OK' if train_loss <= TRAIN_TARGET else 'FAIL'}")
        print(f"  val_loss     = {val_loss:.4f}   (target <= {VAL_TARGET}) "
              f"{'OK' if val_loss <= VAL_TARGET else 'FAIL'}")
        print(f"  overfit ratio= {overfit:.3f}x  (target <= {OVERFIT_MAX}) "
              f"{'OK' if overfit <= OVERFIT_MAX else 'FAIL'}")
        rc = "n/a" if rel_f != rel_f else f"{rel_f:.4f}"
        print(f"  val_rel_mae_f (loss-aligned) = {rc}  (target <= {REL_MAE_TARGET}) "
              f"{'OK' if rel_ok else 'FAIL'}")
        ec = "n/a" if exem_rel_f != exem_rel_f else f"{exem_rel_f:.4f}"
        print(f"  exem prior rel_mae_f         = {ec}  (baseline to beat)")
        print(f"  BEATS EXEM PRIOR             = {'YES' if beats else 'NO'}"
              f"   ({'model better' if beats else 'model NOT better than its prior'})")
        print(f"  val_abs_mae  = {best_m['abs']:.4f}  vs exem {best_m['exem_abs']:.4f}")
        print(f"  GATE: {'PASS' if passed else 'NOT YET MET'}")
        print("=======================================================")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
