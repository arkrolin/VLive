"""Pipeline configuration for the Live2D VLA trainer (V7.1 MVP)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/live2d_vla -> project root


@dataclass
class PipelineConfig:
    # ---- data ----
    data_root: Path = ROOT / "standrad-live-2d"
    gen_mask_path: Path = ROOT / "outputs" / "gen_mask.json"
    whitelist_path: Path = ROOT / "outputs" / "model_whitelist.json"
    retrieval_index: Path = ROOT / "outputs" / "retrieval_index.jsonl"
    fps: float = 30.0
    T: int = 48                       # frames per generated clip (~1.6s)
    subset_models: int = 60           # first N whitelisted models (MVP scope)
    k_exemplars: int = 3              # B1 cross-character same-action references
    max_tokens: int = 128             # pad the per-model token dim to this
    rig_sig_dim: int = 64             # bag-of-hash-buckets rig signature (proxy)

    # ---- model ----
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    mlp_ratio: int = 4
    name_dropout: float = 0.2         # V7.1: 10-30% name dropout for unnamed params
    rig_dropout: float = 0.1          # condition dropout (classifier-free guidance prep)

    # ---- diffusion ----
    num_diff_steps: int = 1000

    # ---- training ----
    batch_size: int = 4
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-2
    epochs: int = 50
    seed: int = 1234
    grad_clip: float = 1.0
    out_dir: Path = ROOT / "outputs" / "train_runs"
