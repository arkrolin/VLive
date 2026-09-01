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
    rig_sig_dim: int = 96             # structured per-param identity feature (2*K, K=48)

    # ---- model ----
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1              # transformer block dropout (generalization)
    name_dropout: float = 0.2         # V7.1: 10-30% name dropout for unnamed params
    rig_dropout: float = 0.1          # condition dropout (classifier-free guidance prep)

    # ---- diffusion ----
    num_diff_steps: int = 1000
    beta_schedule: str = "cosine"     # "cosine" | "linear"

    # ---- training ----
    batch_size: int = 4
    lr: float = 2.0e-4
    lr_min: float = 2.0e-5
    warmup_epochs: int = 3
    weight_decay: float = 1.0e-2
    epochs: int = 50
    seed: int = 1234
    grad_clip: float = 1.0

    # ---- validation / early stopping ----
    val_frac: float = 0.12            # hold out this fraction of *models* (whole-character)
    eval_every: int = 10              # epochs between DDIM reconstruction metric
    early_stop_patience: int = 15     # stop if val_loss not improved for N epochs
    val_recon_batches: int = 16       # cap val reconstruction cost per eval

    # ---- honest evaluation (P0) ----
    # Evaluate only on actions shared by >= N models. The deployment task is
    # "known action, new character"; actions unique to one model make the exem
    # prior degenerate (the (action,param) mean IS that model's own curve), so
    # those samples contribute no generalization signal and dilute every metric.
    # Measured: 72% of actions occur in only one model (subset=60).
    eval_shared_min_models: int = 5

    # ---- generation formulation (P1 ablation) ----
    # The exem prior is very strong (abs_mae=0.47 vs the model's 2.02-3.37), so
    # DDPM-from-pure-noise may be actively destroying it. These knobs ablate it.
    gen_mode: str = "ddpm"            # "ddpm" | "regress" (deterministic residual)
    max_diff_t: int = 1000            # sample t in [0, max_diff_t); <1000 truncates

    # ---- action conditioning (P2) ----
    # "id"   : nn.Embedding(action_vocab, d) - a memorisation table at ~4.4
    #          samples/action (1679 actions / 7382 samples), but empirically the
    #          BEST option once the corpus is scaled up.
    # "name" : compositional char-level encoder over the (semantic pinyin)
    #          action name. Measured WORSE than "id" at subset=285:
    #          abs_mae 1.3134 vs 1.1986 (+9.6% error). Mean-pooling over
    #          characters blurs distinctions between actions that share
    #          characters. Kept for reference; do not re-try without changing
    #          the pooling (e.g. learnable / sequence model instead of mean).
    action_cond: str = "id"           # "id" | "name" | "both"
    action_name_max_len: int = 32     # chars kept per action name

    out_dir: Path = ROOT / "outputs" / "train_runs"
