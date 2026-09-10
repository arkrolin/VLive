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
    # ---- V9 data hygiene (all OFF by default = old behaviour) -------------- #
    # Pinned validation characters (JSON list of model names). Needed because
    # val_frac draws a DIFFERENT holdout as soon as the whitelist grows, which
    # makes abs_mae incomparable across data scales.
    val_holdout_path: str = ""
    # {model: [action, ...]} to drop: the same (moc3 md5, motion md5) pair
    # already occurs earlier in the corpus. See outputs/prep_v9.py.
    dedup_skip_path: str = ""
    # Semantic action consolidation map (outputs/action_semantic_map.json).
    action_map_path: str = ""
    # Namespaces outputs/exem_cache_*.pkl, ident_cache_*.pkl, range_cache_*.pkl
    # so runs that share subset_models but not the corpus never share a cache.
    cache_tag: str = ""
    fps: float = 30.0
    T: int = 48                       # frames per generated clip (~1.6s)
    subset_models: int = 60           # first N whitelisted models (MVP scope)
    k_exemplars: int = 3              # B1 cross-character same-action references
    max_tokens: int = 128             # pad the per-model token dim to this
    rig_sig_dim: int = 96             # structured per-param identity feature (2*K, K=48)

    # ---- V11a: character-conditioned features (leave-one-out) ----
    # Measured (outputs/diag_ref_prior3.py): build_model_identity aggregates each
    # model's per-param (min, max, mean) over ALL its motions, INCLUDING the
    # target motion. 29.2% of channels have the target contributing >=50% of the
    # global range and 5.2% have it as the sole contributor -> the 96d rig
    # feature leaks the target's amplitude. That is why the model improves
    # point-wise MAE by +39.8% but shape (1st-difference) MAE by only +2.1%.
    # rig_loo=True excludes the target motion from its own identity vector,
    # which is also exactly what deployment does (the target does not exist yet).
    rig_loo: bool = False
    # Per-token statistics of THIS character's OTHER motions for the same param
    # (rest value, amplitude, p10/p90 envelope, support count). Measured: the
    # character's own motion bank carries +85% real signal over any other
    # character's bank, and the exact rest value is worth 1.31 -> 0.03 abs_mae
    # on the 43.9% of channels whose target is constant. "none" = off,
    # "mlp" = a zero-init Linear(5, d) added to the token embedding (so a
    # char_stats run starts bit-identical to the baseline).
    char_stats: str = "none"
    # V11b: condition on the character's OWN motion bank (its other motions).
    # "attn" = per-param cross-attention over K reference curves; the attended
    # curve also enters in_proj as a 3rd channel. "none" keeps the 2-channel
    # baseline bit-identical.
    bank_cond: str = "none"          # "none" | "attn"
    bank_k: int = 8                  # reference actions sampled per sample

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
    # Cap the reconstruction eval by SAMPLES, not batches: the val loader uses
    # batch_size=cfg.batch_size, so a batch cap silently changes the evaluated
    # subset (16x4=64 samples -> exem 2.4405; 16x8=128 -> exem 2.7254), which
    # makes abs_mae incomparable across runs with different batch sizes.
    val_recon_samples: int = 64       # val samples scored per reconstruction eval
    # The val set is ordered by character, so a contiguous prefix of N samples
    # covers only a few held-out characters (the first 64 of 561 come from just
    # 5 of 34 models, and those 5 are ~1.8x harder than average: exem prior
    # abs_mae 2.4405 there vs ~1.345 over all 561). Any metric taken on that
    # prefix describes 5 characters, not 34. Sample the eval subset STRATIFIED
    # over characters instead (see _stratified_val_indices in train.py).
    val_recon_per_model: int = 4      # val samples scored per held-out character
    val_recon_seed: int = 1234        # fixed -> the subset is identical every run
    # Loss weighting by param span. 0 = every param-instance counts the same
    # (normalised units -> optimises rel_f). p>0 multiplies each instance's
    # squared error by (span / batch-mean-span) ** p, so p=2 makes the loss
    # raw-unit MSE, i.e. aligned with abs_mae. Motivated by the w-sweep:
    # abs_mae wants w~0.75 while rel_f wants w=1.0, i.e. the residual
    # overshoots on large-range params and the loss never charges it for that.
    span_w: float = 0.0             # 0 = off (equal weight, legacy)
    span_w_cap: float = 8.0         # clip the relative span ratio before pow

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

    # ---- learnable residual gate (V8.2) ----
    # x0_hat = exem + g * residual, g = sigmoid(bias + Emb(param_name)[name]).
    #
    # Motivation (diag_error_floor.py): an ORACLE gate that picks model-vs-prior
    # per param-instance scores abs_mae 0.7914, i.e. 37% below the best model,
    # so "learning when NOT to apply the residual" is the highest ceiling left.
    # The largest measured defect is systematic overshoot on the top span
    # decile (that bucket alone is -103%); param NAME correlates with span, so a
    # single lookup table can express most of that.
    #
    # The gate sees ONLY the static param name - never the instance's residual -
    # so it cannot collapse into "predict nothing" per sample; it can only learn
    # "params of this kind should be shrunk". Init bias=4 -> g~0.98 (a no-op),
    # so a gated run starts from the ungated model's behaviour.
    residual_gate: str = "none"       # "none" | "name"

    # ---- per-token span conditioning (V8.3) ----
    # The model never OBSERVES each param's absolute range: span enters only the
    # normalisation (x0/span) and the loss weighting (span_w), so the network
    # has to infer "is this a large-range param?" indirectly from its name.
    # §20/§21 showed residual overshoot is strongly span-correlated, so we give
    # the model the answer directly: inject log(span) as a per-token feature.
    # "none" = legacy. "log" = add Linear(1,d)(log span) to the token embedding.
    # Zero-init makes a span_cond="log" run start bit-identical to "none".
    span_cond: str = "none"           # "none" | "log"

    # ---- checkpoint selection (V8.2) ----
    # WHICH score decides `ckpt_best.pt` / early stopping.
    # "val_loss" is the legacy choice and is WRONG: measured on abl_K_span1 the
    # val_loss-selected checkpoint (ep32) scores abs_mae 1.3654 on the full val
    # set while the LAST epoch (ep45) scores 1.1401 - a 20% swing in the very
    # metric the gate is written against. span-weighted losses make val_loss and
    # abs_mae diverge even harder, so selection must follow the target metric.
    select_metric: str = "val_loss"   # "val_loss" | "abs" | "rel_f"

    # ---- structured latent head (V10) ------------------------------------- #
    # The dense head outputs a P x T residual per sample, but the residual is
    # actually ~rank-2 (diag_residual_rank.py: k90=2/48, PR 1.29 vs 2.46 noise)
    # and is ~orthogonal to the prior (cos -0.012). diag_oracle_ceiling.py shows
    # x0_hat = exem*(1+alpha) + rank-2 shape correction has abs_mae ceiling
    # 0.204 vs the dense head's 0.978. So we replace the PxT head with a
    # structured one: a per-param amplitude gain alpha (P dims) + a low-rank
    # shape correction U_r @ C where U_r is a small set of global time bases.
    #
    # "dense"       = legacy Linear(d,1) -> PxT residual (the R-arm head).
    # "structured"  = exem*(1+alpha) + coef @ basis, alpha in R^P, coef in R^(P,r),
    #                 basis in R^(r,T) (DCT low-freq bases). Output dims collapse
    #                 from P*T (~2688) to P*(1+r) (~56*3=168).
    head_mode: str = "dense"         # "dense" | "structured"
    residual_rank: int = 2           # r = number of global shape bases (rank)

    # ---- per-param range statistics (see outputs/_patch_range_fix.py) ----
    # None = scan the whole train corpus (one pass, ~3.5 min, cached to
    # outputs/range_cache_*.pkl). 400 = the old behaviour, which covered only
    # 29/251 train characters and handed 14.9% of val instances a 1130-wide
    # span (665x their true median span) -> 88% of abs_mae's weighting.
    range_stats_n: int = None
    # Span given to params still absent from the stats: the q-quantile of the
    # per-param span distribution. <0 restores the legacy global range.
    fb_span_q: float = 0.5

    out_dir: Path = ROOT / "outputs" / "train_runs"
