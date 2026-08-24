"""V7.1 training entrypoint: DDPM training of Live2DModel.

Implements the minimal runnable training prototype for task #2:
    param-as-token + 4-way identity embedding (name/rig/deformer/action)
    + stable DiT (QKNorm/RMSNorm/SwiGLU) + DDPM noise-prediction.

Run:
    # single GPU (smoke test)
    .venv/bin/python src/live2d_vla/train.py --epochs 3 --max_steps 200
    # multi-GPU via torchrun
    torchrun --nproc_per_node=2 src/live2d_vla/train.py --epochs 50

Loss = masked MSE between predicted noise and true noise, with arm-channel
up-weighting. Curves are globally standardized (x0 -> (x0-mean)/std) for
diffusion stability across hetero-scaled Live2D params (angles vs [0,1]).

NOTE (coverage gap, tracked as follow-up): the B1 exemplar signal (`exem`)
is produced by the dataset but the current retrieval index does not cover all
whitelisted packs, so `exem` is frequently all-zero and is intentionally NOT
fed to the model yet (doing so would bias predictions toward zero). The action
condition is provided via the learned `action_id` embedding. Wiring `exem` as
an explicit condition is a clean next step once the retrieval index is rebuilt
for all whitelisted models.
"""
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, RandomSampler

from config import PipelineConfig
from dataset import Live2DDataset, collate
from model import Live2DModel

# Arm / upper-body channels get higher loss weight (V7.1 action prior).
ARM_KEYWORDS = {
    "arm", "shoulder", "elbow", "wrist", "hand", "finger", "forearm",
    "upper", "lower", "bicep", "tricep", "neck",
}


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


def arm_weight(names_b, weight: float = 3.0, n_pad: int = None) -> np.ndarray:
    """(B, n_pad) per-token loss weight; arm/upper-body params get `weight`.

    Width is padded to `n_pad` (the token_mask width) so it broadcasts with the
    padded mask; padding columns keep weight 1.0 (they are masked out anyway).
    """
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


def compute_stats(ds, n: int = 200):
    """Global mean/std of x0 over active token frames (one-time, cheap)."""
    total_sum, total_sq, total_n = 0.0, 0.0, 0
    n = min(n, len(ds))
    for i in range(n):
        try:
            s = ds[i]
        except Exception:
            continue
        x = np.asarray(s["target"], dtype=np.float32)
        total_sum += float(x.sum())
        total_sq += float((x ** 2).sum())
        total_n += x.size
    mean = total_sum / max(total_n, 1)
    var = total_sq / max(total_n, 1) - mean ** 2
    std = math.sqrt(max(var, 1e-6)) + 1e-6
    return float(mean), float(std)


def parse_args():
    p = argparse.ArgumentParser(description="V7.1 Live2D VLA DDPM trainer")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--subset_models", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None,
                   help="cap total optimizer steps (smoke test)")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--stats_samples", type=int, default=200)
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
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    device, rank, world_size, ddp = setup_dist()
    if ddp:
        dist.barrier()

    # ---- dataset + model ----
    ds = Live2DDataset(cfg)
    action_vocab_size = len(ds.action2idx)
    n_tokens_pad = cfg.max_tokens
    print(f"[rank{rank}] dataset samples={len(ds)} "
          f"params_vocab={len(ds.param2idx)} actions={action_vocab_size}")

    model = Live2DModel(cfg, ds.word2idx, action_vocab_size, n_tokens_pad)
    model.to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=False)
        dmodel = model.module
    else:
        dmodel = model

    # ---- standardization stats ----
    mean, std = compute_stats(ds, n=args.stats_samples)
    if is_main(rank):
        print(f"[rank{rank}] x0 stats mean={mean:.4f} std={std:.4f}")

    betas, alphas, alphabar = dmodel.ddpm_schedule(cfg.num_diff_steps)
    alphabar = alphabar.to(device)

    optimizer = torch.optim.AdamW(
        dmodel.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---- sampler / loader ----
    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = RandomSampler(ds)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, sampler=sampler,
        collate_fn=lambda b: collate(b, cfg.max_tokens),
        num_workers=0, pin_memory=(device.type == "cuda"))

    # ---- resume ----
    start_epoch = 0
    ckpt = cfg.out_dir / "ckpt_latest.pt"
    if ckpt.exists() and not args.fresh:
        sd = torch.load(ckpt, map_location=device)
        dmodel.load_state_dict(sd["model"])
        optimizer.load_state_dict(sd["optim"])
        start_epoch = sd.get("epoch", 0)
        mean, std = sd.get("mean", mean), sd.get("std", std)
        if is_main(rank):
            print(f"[rank{rank}] resume from epoch {start_epoch}")

    # ---- training loop ----
    steps_per_epoch = len(loader)
    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, cfg.epochs):
        if ddp:
            sampler.set_epoch(epoch)
        dmodel.train()
        running, cnt = 0.0, 0
        t0 = time.time()
        for step, batch in enumerate(loader):
            target = batch["target"].to(device)
            rig = batch["rig"].to(device)
            action_id = batch["action_id"].to(device)
            token_mask = batch["token_mask"].to(device)
            names = batch["names"]

            x0 = (target - mean) / std
            B = x0.shape[0]
            t = torch.randint(0, cfg.num_diff_steps, (B,),
                              device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = dmodel.q_sample(x0, t, noise, alphabar)
            pred = model(x_t, t, names, rig, action_id, token_mask,
                         training=True)

            se = (pred - noise) ** 2                       # (B, n, T)
            aw = torch.from_numpy(
                arm_weight(names, args.arm_weight, n_pad=cfg.max_tokens)).to(device).unsqueeze(-1)
            valid = token_mask.unsqueeze(-1) * aw          # (B, n, 1)
            loss = (se * valid).sum() / valid.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dmodel.parameters(), cfg.grad_clip)
            optimizer.step()

            running += loss.item()
            cnt += 1
            global_step += 1
            if is_main(rank) and step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"[ep {epoch + 1}/{cfg.epochs} step {step}/{steps_per_epoch}] "
                      f"loss={loss.item():.4f} lr={lr:.2e} "
                      f"elapsed={time.time() - t0:.1f}s")
            if args.max_steps and global_step >= args.max_steps:
                break

        if is_main(rank):
            sd = {
                "model": dmodel.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": epoch + 1,
                "mean": mean,
                "std": std,
                "cfg": cfg.__dict__,
            }
            torch.save(sd, ckpt)
            torch.save(sd, cfg.out_dir / f"ckpt_ep{epoch + 1}.pt")
            print(f"[rank{rank}] save epoch {epoch + 1} "
                  f"avg_loss={running / max(cnt, 1):.4f}")
        if args.max_steps and global_step >= args.max_steps:
            break

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
