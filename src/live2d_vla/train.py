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

Acceptance gate (recalibrated 2026-09-01, printed at the end):
    [1] BEATS EXEM PRIOR - model abs_mae must beat the exem prior by >= 10%.
        (primary criterion; see the note on the old targets below)
    [2] val_rel_mae_f <= RELF_TARGET (0.05)   [loss-aligned rel error]
    [3] train_loss    <= TRAIN_TARGET (0.09)  [sanity: optimisation converged]

    Why the old targets (val_loss<=0.06, overfit<=1.30) were dropped as gate
    criteria: they were calibrated against the pre-P0 metric, which mixed in
    memorisation samples (72% of actions belong to a single model) and used a
    span definition inconsistent with the training loss. They produced a false
    PASS - the ddpm@285 arm printed "GATE: PASS" while "BEATS EXEM PRIOR = NO".
    val_loss is also NOT comparable across generation formulations: a truncated
    schedule makes the denoising task trivially easier, so the t200 arm had the
    lowest val_loss (0.0150) and by far the worst reconstruction (25-87).
    val_loss / overfit are still printed, but as diagnostics only.

    Measured reference points (shared-action held-out val):
        subset=60  : exem prior abs_mae=0.7379
        subset=285 : exem prior abs_mae=2.4405 (harder: 34 held-out characters)
        best so far: regress@285 abs_mae=1.1986 (51% better than its prior)

Run:
    torchrun --nproc_per_node=7 src/live2d_vla/train.py --epochs 50
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

from config import PipelineConfig
from dataset import Live2DDataset, collate
from model import Live2DModel

# ---- acceptance targets ---------------------------------------------------- #
# RECALIBRATED 2026-09-01 against the honest shared-action eval.
#
# The old targets (val_loss<=0.06, overfit<=1.30) were calibrated against the
# pre-P0 metric, which mixed in memorisation samples (72% of actions belong to a
# single model) and used a span definition inconsistent with the training loss.
# They produced a false PASS: the ddpm@285 arm printed "GATE: PASS" while
# "BEATS EXEM PRIOR = NO" - i.e. the gate passed on a model that was worse than
# just outputting the action mean.
#
# The only criterion that survives every ablation is:
#     does the model beat the exem prior it is initialised to?
# val_loss / overfit are still reported, but as DIAGNOSTICS only: val_loss is not
# comparable across generation formulations (a truncated schedule makes the
# denoising task trivially easier - the t200 arm had the lowest val_loss 0.0150
# and by far the worst reconstruction 25-87).
TRAIN_TARGET = 0.09        # range-norm MSE on train (sanity: optimisation works)
RELF_TARGET = 0.05         # loss-aligned rel_mae (fraction of floored span)
BEATS_EXEM_MIN_GAIN = 0.10 # model abs_mae must beat exem by >= 10% to count

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
    ac = batch.get("action_chars")
    action_chars = ac.to(device) if ac is not None else None
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
    x0_hat = model(x_t, t, names, rig, action_id, token_mask, exem_s, training=True,
                   action_chars=action_chars)
    se = (x0_hat - x0) ** 2
    aw = torch.from_numpy(
        arm_weight(names, arm_w, n_pad=cfg.max_tokens)).to(device).unsqueeze(-1)
    valid = token_mask.unsqueeze(-1) * aw
    if getattr(cfg, "span_w", 0.0) > 0:
        # Span-weighted loss: (span / batch-mean-span) ** span_w, clipped so a
        # few extreme-range params cannot monopolise the gradient. span_w=2
        # makes this raw-unit MSE, i.e. the loss then optimises abs_mae.
        ref = span_t[token_mask > 0.5].mean().clamp(min=1e-6)
        cap = float(getattr(cfg, "span_w_cap", 8.0))
        ratio = (span_t / ref).clamp(1.0 / cap, cap)
        valid = valid * ratio.pow(float(cfg.span_w)).unsqueeze(-1)
    loss = (se * valid).sum() / valid.sum().clamp(min=1.0)
    return loss


# --------------------------------------------------------------------------- #
# DDIM reverse sampling + reconstruction metric
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ddim_reverse(model, x_T, names, rig, action_id, token_mask, exem_s, alphabar,
                 device, steps: int = 50, eta: float = 0.0, t_start: int = None,
                 action_chars=None):
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
                       training=False, action_chars=action_chars)
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
                  alphabar, n_samples: int, steps: int = 50,
                  collect: dict | None = None, residual_scale: float = 1.0):
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

    ``residual_scale`` (w) rewrites the prediction as ``exem + w * residual``.
    w=1 is the trained model; w<1 shrinks toward the exem prior. Used offline
    (outputs/eval_ckpt.py --residual_scale) to map the abs_mae / rel_f
    trade-off without retraining.

    If ``collect`` is a dict, the per-param-instance errors are appended to
    ``collect["abs"]`` / ``collect["exem_abs"]`` (aligned element-wise, in
    iteration order) instead of only being averaged. Two runs over the same val
    subset then yield PAIRED vectors, so their difference can be tested far more
    precisely than either mean.
    """
    model.eval()
    min_span = max(g_hi - g_lo, 1.0) * 0.02
    acc = {k: 0.0 for k in METRIC_KEYS}
    cnt = 0
    n_seen = 0
    for _bi, batch in enumerate(loader):
        # Cap on SAMPLES so the metric does not depend on the loader batch size.
        if n_seen >= n_samples:
            break
        n_seen += len(batch["target"])
        target = batch["target"].to(device)
        rig = batch["rig"].to(device)
        action_id = batch["action_id"].to(device)
        ac = batch.get("action_chars")
        action_chars = ac.to(device) if ac is not None else None
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
                           token_mask, exem_s, training=False,
                           action_chars=action_chars)
        else:
            x_T = torch.randn(B, n, T, device=device)
            x0_hat = ddim_reverse(
                model, x_T, names, rig, action_id, token_mask, exem_s, alphabar,
                device, steps=steps,
                t_start=int(getattr(cfg, "max_diff_t", 1000)),
                action_chars=action_chars)
        if residual_scale != 1.0:
            x0_hat = exem_s + residual_scale * (x0_hat - exem_s)
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
                    if collect is not None:
                        collect.setdefault(pre + "abs", []).append(e_abs)
                        collect.setdefault(pre + "rel_f", []).append(min(e_f, 2.0))
                    if pre == "":
                        cnt += 1
    model.train()
    out = {k: acc[k] / max(cnt, 1) for k in METRIC_KEYS}
    out["n_params"] = cnt
    out["n_samples"] = n_seen
    return out


def _stratified_val_indices(val_ds, per_model: int, seed: int) -> list[int]:
    """Pick up to `per_model` val samples per held-out character.

    The val set is ordered by character, so a contiguous prefix evaluates only a
    few characters (first 64 of 561 -> 5 of 34 models). Stratifying guarantees
    every held-out character is represented, which matters because character
    level variance dominates this metric.
    """
    by_model: dict[str, list[int]] = {}
    for i, (m, _a) in enumerate(val_ds.samples):
        by_model.setdefault(m, []).append(i)
    rng = random.Random(seed)
    idx: list[int] = []
    for m in sorted(by_model):
        pool = list(by_model[m])
        rng.shuffle(pool)
        idx.extend(pool[:per_model])
    return sorted(idx)


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
    p.add_argument("--action_cond", type=str, default=None,
                   choices=["id", "name", "both"],
                   help="P2 ablation: id (lookup table) | name (compositional) | both")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--arm_weight", type=float, default=3.0)
    # --- regularisation / capacity / schedule knobs (overfitting ablation) ---
    p.add_argument("--dropout", type=float, default=None,
                   help="transformer block dropout")
    p.add_argument("--name_dropout", type=float, default=None,
                   help="per-param name dropout")
    p.add_argument("--rig_dropout", type=float, default=None,
                   help="identity condition dropout")
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--lr_min", type=float, default=None, help="cosine LR floor")
    p.add_argument("--d_model", type=int, default=None, help="transformer width")
    p.add_argument("--span_w", type=float, default=None,
                   help="weight each param-instance loss by "
                        "(span/mean_span)**span_w; 0 = equal weight (legacy), "
                        "2 = raw-unit MSE (matches abs_mae)")
    p.add_argument("--span_w_cap", type=float, default=None,
                   help="clip the relative span ratio to [1/cap, cap] before pow")
    p.add_argument("--n_layers", type=int, default=None, help="DiT depth")
    p.add_argument("--patience", type=int, default=None,
                   help="early-stopping patience (epochs) on val_loss")
    p.add_argument("--val_recon_samples", type=int, default=None,
                   help="val samples scored per reconstruction eval (cap)")
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
    if args.action_cond is not None:
        cfg.action_cond = args.action_cond
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.name_dropout is not None:
        cfg.name_dropout = args.name_dropout
    if args.rig_dropout is not None:
        cfg.rig_dropout = args.rig_dropout
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.lr_min is not None:
        cfg.lr_min = args.lr_min
    if args.d_model is not None:
        cfg.d_model = args.d_model
    if args.span_w is not None:
        cfg.span_w = args.span_w
    if args.span_w_cap is not None:
        cfg.span_w_cap = args.span_w_cap
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    if args.patience is not None:
        cfg.early_stop_patience = args.patience
    if args.val_recon_samples is not None:
        cfg.val_recon_samples = args.val_recon_samples
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
              f"action_cond={cfg.action_cond} "
              f"eval_shared_min_models={cfg.eval_shared_min_models}")
        print(f"[rank{rank}] dropout={cfg.dropout} name_dropout={cfg.name_dropout} "
              f"rig_dropout={cfg.rig_dropout} weight_decay={cfg.weight_decay} "
              f"lr={cfg.lr} lr_min={cfg.lr_min} d_model={cfg.d_model} "
              f"span_w={cfg.span_w} span_w_cap={cfg.span_w_cap} "
              f"n_layers={cfg.n_layers} batch_size={cfg.batch_size} "
              f"patience={cfg.early_stop_patience}")

    model = Live2DModel(cfg, train_ds.word2idx, action_vocab_size, n_tokens_pad,
                        train_ds.action_char2idx)
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

    # Reconstruction-eval subset: stratified over held-out characters so the
    # metric reflects all of them instead of the first few in dataset order.
    eval_idx = _stratified_val_indices(val_ds, max(1, cfg.val_recon_per_model),
                                       cfg.val_recon_seed)
    eval_loader = DataLoader(
        Subset(val_ds, eval_idx), batch_size=cfg.batch_size,
        sampler=SequentialSampler(eval_idx),
        collate_fn=lambda b: collate(b, cfg.max_tokens), num_workers=0)
    if is_main(rank):
        print(f"[rank{rank}] recon eval subset={len(eval_idx)} samples "
              f"({cfg.val_recon_per_model}/character, stratified) "
              f"val_loss uses all {len(val_ds)}")

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
                    dmodel, eval_loader, device, cfg, per_lo, per_hi,
                    g_lo, g_hi, alphabar,
                    n_samples=len(eval_idx), steps=50)
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
        m_abs, e_abs = best_m["abs"], best_m["exem_abs"]
        train_ok = train_loss <= TRAIN_TARGET
        rel_ok = (rel_f != rel_f) or (rel_f <= RELF_TARGET)
        # The comparison that actually matters, and the only one that survived
        # every ablation: beat the prior the model is initialised to, by a
        # margin that is not explainable by noise.
        gain = (e_abs - m_abs) / e_abs if (e_abs == e_abs and e_abs > 0) else float("nan")
        beats = (gain == gain) and (gain >= BEATS_EXEM_MIN_GAIN)
        passed = train_ok and rel_ok and beats
        rc = "n/a" if rel_f != rel_f else f"{rel_f:.4f}"
        print("\n==================== TRAINING GATE ====================")
        print(f"  [1] BEATS EXEM PRIOR  : {'YES' if beats else 'NO'}   "
              f"abs_mae {m_abs:.4f} vs exem {e_abs:.4f}  "
              f"(gain {gain * 100:+.1f}%, need >= {BEATS_EXEM_MIN_GAIN * 100:.0f}%)")
        print(f"  [2] val_rel_mae_f     : {rc}  (target <= {RELF_TARGET}) "
              f"{'OK' if rel_ok else 'FAIL'}")
        print(f"  [3] train_loss        : {train_loss:.4f}  "
              f"(target <= {TRAIN_TARGET}) {'OK' if train_ok else 'FAIL'}")
        print("  --- diagnostics only (NOT comparable across formulations) ---")
        print(f"      val_loss={val_loss:.4f}  overfit={overfit:.3f}x  "
              f"rel_mae(legacy)={best_m['rel']:.4f} vs exem {best_m['exem_rel']:.4f}")
        print(f"  GATE: {'PASS' if passed else 'NOT YET MET'}")
        print("=======================================================")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
