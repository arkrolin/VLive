"""Evaluate the saved best checkpoint: compute val_abs_mae / val_rel_mae on the
held-out val split via DDIM-50 reverse (the deployment-relevant recon metric).

Run on server from src/live2d_vla with the project venv:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python ../outputs/eval_recon.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))

import torch
from config import PipelineConfig
from dataset import Live2DDataset, collate
from model import Live2DModel
from train import (
    compute_range, mean_range_vecs, arm_weight, recon_metrics,
)
from torch.utils.data import DataLoader, SequentialSampler

cfg = PipelineConfig()
train_ds = Live2DDataset(cfg, split="train")
val_ds = Live2DDataset(cfg, split="val")
action_vocab_size = len(train_ds.action2idx)
n_tokens_pad = cfg.max_tokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Live2DModel(cfg, train_ds.word2idx, action_vocab_size, n_tokens_pad).to(device)

import os
ckpt_path = os.environ.get("CKPT", str(ROOT / "outputs" / "train_runs" / "x0_exem_res_zinit" / "ckpt_best.pt"))
sd = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(sd["model"])
model.eval()
print(f"[eval] loaded {ckpt_path}  (val_loss@save={sd.get('val_loss')})")

g_lo, g_hi, per_lo, per_hi = compute_range(train_ds, n=400)
alphabar = model.ddpm_schedule(cfg.num_diff_steps, cosine=(cfg.beta_schedule == "cosine"))[2].to(device)

val_loader = DataLoader(
    val_ds, batch_size=cfg.batch_size, sampler=SequentialSampler(val_ds),
    collate_fn=lambda b: collate(b, cfg.max_tokens), num_workers=0)

abs_mae, rel_mae = recon_metrics(
    model, val_loader, device, cfg, per_lo, per_hi, g_lo, g_hi,
    alphabar, n_batches=50, steps=50)
print(f"[eval] val_abs_mae={abs_mae:.4f}  val_rel_mae={rel_mae:.4f}  "
      f"(REL_MAE_TARGET=0.15)")
print(f"[eval] PASS rel_mae <= 0.15: {rel_mae <= 0.15}")
