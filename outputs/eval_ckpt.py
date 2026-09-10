"""Offline checkpoint evaluation on the FULL shared val set (+ paired tests).

The trainer scores only ``cfg.val_recon_samples`` (64) val samples per epoch to
keep training cheap. That is too noisy to rank arms that differ by ~7%. This
script re-scores saved checkpoints over all shared val samples and, when two
runs are given, performs a PAIRED comparison (both iterate the same samples in
the same order, so their per-param-instance error vectors align).

Usage
-----
    # score one or more runs on the full val set
    .venv/bin/python outputs/eval_ckpt.py abl_H_regress285_ep45 abl_I_reg45

    # limit to the first N val samples (for a like-for-like check vs the trainer)
    .venv/bin/python outputs/eval_ckpt.py abl_H_regress285_ep45 --n_samples 64

    # choose GPU (default: cuda:0)
    .venv/bin/python outputs/eval_ckpt.py ... --device cuda:5

    # residual-scale sweep: load the checkpoint ONCE, score it at every w.
    # x0_hat = exem + w * residual. w=1 is the trained model, w=0 is the prior.
    .venv/bin/python outputs/eval_ckpt.py abl_H_regress285_ep45 \
        --sweep 0.0,0.25,0.5,0.75,1.0,1.25

Notes
-----
* Uses ``ckpt_best.pt`` (selected on val_loss) unless ``--ckpt latest`` is given.
* The exem prior is scored on identical batches, so "vs exem" is always honest.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, SequentialSampler  # noqa: E402

from config import PipelineConfig  # noqa: E402
from dataset import Live2DDataset, collate  # noqa: E402
from model import Live2DModel  # noqa: E402
from train import compute_range, recon_metrics  # noqa: E402

RUNS = ROOT / "outputs" / "train_runs"

# Cache the (expensive) train corpus + range stats - but ONLY across runs whose
# config produces a bit-identical corpus. Keying on subset_models alone was a
# bug: arms D/E predate the ParamNameEncoder fix, and reusing their cached
# dataset (plus the process state left behind by a strict=False load) shifted a
# later arm's abs_mae by ~6%. Safer default: one process per checkpoint.
_CORPUS_KEYS = ("subset_models", "T", "fps", "k_exemplars", "max_tokens",
                "val_frac", "seed", "eval_shared_min_models", "action_cond",
                "action_name_max_len",
                # The range statistics decide each param's NORMALISATION, so two
                # runs that differ here must never share a cached corpus:
                # reusing a full-corpus (g_lo, per_lo, ...) for a legacy run
                # would silently change 14.9% of spans by up to 665x.
                "range_stats_n", "fb_span_q",
                # V9: corpus-defining settings. Two runs that differ in any of
                # these build a different train/val split, so they must never
                # share a cached corpus.
                "data_root", "val_holdout_path", "dedup_skip_path",
                "action_map_path", "cache_tag",
                # V11a: these change the per-sample tensors (rig is recomputed
                # leave-one-out; cstats is added to the batch only when the
                # feature is on), so a cached corpus must never be reused
                # across runs that differ in them.
                "rig_loo", "char_stats",
                # V11b: adds the K reference-curve tensors to every batch.
                "bank_cond", "bank_k")
_CACHE: dict[tuple, tuple] = {}


def get_corpus(cfg):
    key = tuple(getattr(cfg, k, None) for k in _CORPUS_KEYS)
    if key not in _CACHE:
        train_ds = Live2DDataset(cfg, split="train")
        val_ds = Live2DDataset(cfg, split="val")
        g_lo, g_hi, per_lo, per_hi, fb_span = compute_range(
            train_ds, n=getattr(cfg, 'range_stats_n', None))
        # cfg.fb_span is a DERIVED value (it lives on cfg only so that
        # mean_range_vecs can see it). Fill it in only when the caller has not
        # pinned it, because None is a meaningful value there: it selects the
        # LEGACY fill (g_lo, g_hi), which is NOT equivalent to (0, fb_span)
        # even when the spans coincide - the network is nonlinear in its input,
        # so a constant offset in the normalised units does not cancel out.
        # Measured: forcing the (0, fb_span) fill on a legacy checkpoint moved
        # K from 1.1401 to 1.1871 and H from 1.2630 to 1.2865.
        if getattr(cfg, "fb_span", "UNSET") == "UNSET":
            cfg.fb_span = fb_span
        _CACHE[key] = (train_ds, val_ds, g_lo, g_hi, per_lo, per_hi)
    return _CACHE[key]


def load_ctx(run: str, device: torch.device, n_samples: int,
             which: str = "best", stratified: int = 0,
             override: dict | None = None) -> dict:
    """Build everything needed to score one checkpoint.

    Split out from :func:`evaluate` so a residual-scale sweep can pay the
    expensive part (corpus construction + model load) exactly once and then
    call ``recon_metrics`` repeatedly with different w.
    """
    ckpt_path = RUNS / run / f"ckpt_{which}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = PipelineConfig()
    _saved_cfg = sd.get("cfg", {})
    cfg.__dict__.update(_saved_cfg)                 # restore the run's own config
    if override:
        # Deliberate post-hoc change of a DATA-SIDE setting, e.g. forcing
        # rig_loo=True on a checkpoint that was trained with the leaky rig.
        # This is how we measure the size of the leak without retraining:
        # the model is unchanged, only what it is shown at eval time is.
        for k, v in override.items():
            setattr(cfg, k, v)
        print(f"  [{run}] OVERRIDE {override}")
    _legacy_range = "range_stats_n" not in _saved_cfg
    if _legacy_range:
        # Checkpoints saved before outputs/_patch_range_fix.py carry neither
        # key. They were trained under n=400 stats + GLOBAL-range fallback, and
        # they MUST be scored under exactly that: those stats cover only
        # 29/251 train characters, so 14.9% of val instances were given a
        # 1130-wide span (665x their true median span) and carry ~88% of
        # abs_mae's weighting. Scoring them with the fixed stats would change
        # the metric without changing the model - i.e. compare apples to
        # oranges. fb_span_q < 0 restores the global-range fallback.
        cfg.range_stats_n = 400
        cfg.fb_span_q = -1.0
        cfg.fb_span = None

    train_ds, val_ds, g_lo, g_hi, per_lo, per_hi = get_corpus(cfg)
    print(f"  [{run}] range stats: n={cfg.range_stats_n} "
          f"fb_span_q={cfg.fb_span_q} fb_span="
          f"{'legacy (g_lo,g_hi) fill' if cfg.fb_span is None else f'{cfg.fb_span:.4f}'}"
          f"{'  (LEGACY checkpoint -> forced to n=400 / global fill)' if _legacy_range else ''}")
    model = Live2DModel(cfg, train_ds.word2idx, len(train_ds.action2idx),
                        cfg.max_tokens, train_ds.action_char2idx,
                        getattr(train_ds, "param2idx", None))
    model.to(device)
    # strict=False: arms D/E predate the ParamNameEncoder bug fix (commit 23d0459),
    # so their checkpoints have no "name_enc.*" weights - and during those runs the
    # name branch genuinely WAS an untrained random init, so leaving it freshly
    # initialised here faithfully reproduces them.
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    missing_top = sorted({m.split(".")[0] for m in missing})
    if missing_top:
        print(f"  [{run}] strict=False: missing {missing_top} "
              f"(untrained in the original run - faithful)")
    if unexpected:
        print(f"  [{run}] unexpected keys: {sorted({u.split('.')[0] for u in unexpected})}")
    model.eval()

    alphabar = model.ddpm_schedule(
        cfg.num_diff_steps, cosine=(cfg.beta_schedule == "cosine"))[2].to(device)

    if stratified > 0:
        # same stratified-over-characters subset the trainer uses in-training
        from train import _stratified_val_indices
        sub_idx = _stratified_val_indices(val_ds, stratified,
                                          getattr(cfg, "val_recon_seed", 1234))
        from torch.utils.data import Subset
        loader = DataLoader(
            Subset(val_ds, sub_idx), batch_size=4,
            sampler=SequentialSampler(sub_idx),
            collate_fn=lambda b: collate(b, cfg.max_tokens), num_workers=0)
        cap = len(sub_idx)
    else:
        loader = DataLoader(
            val_ds, batch_size=4, sampler=SequentialSampler(val_ds),
            collate_fn=lambda b: collate(b, cfg.max_tokens), num_workers=0)
        cap = n_samples if n_samples > 0 else len(val_ds)
    label = run if not override else run + "+rigloo"
    return dict(run=label, device=device, model=model, cfg=cfg, loader=loader,
                cap=cap, per_lo=per_lo, per_hi=per_hi, g_lo=g_lo, g_hi=g_hi,
                alphabar=alphabar, sd=sd)


def evaluate(ctx: dict, residual_scale: float = 1.0) -> dict:
    """Score one already-loaded checkpoint.

    ``residual_scale`` (w) rewrites the prediction as ``exem + w * residual``:
    w=1 is the trained model, w=0 is the exem prior, w<1 shrinks toward the
    prior. Sweeping it maps the abs_mae / rel_f trade-off with no retraining.
    Safe to call repeatedly - ``recon_metrics`` re-enters eval mode itself.
    """
    run = ctx["run"]
    collect: dict = {}
    with torch.no_grad():
        m = recon_metrics(ctx["model"], ctx["loader"], ctx["device"], ctx["cfg"],
                          ctx["per_lo"], ctx["per_hi"], ctx["g_lo"], ctx["g_hi"],
                          ctx["alphabar"], n_samples=ctx["cap"], steps=50,
                          collect=collect, residual_scale=residual_scale)
    sd = ctx["sd"]
    return {
        "run": run if residual_scale == 1.0 else f"{run}@w{residual_scale:g}",
        "base": run,
        "w": residual_scale,
        "epoch": sd.get("epoch"),
        "val_loss": sd.get("val_loss"),
        "n_samples": m.get("n_samples"),
        "abs": m["abs"],
        "exem_abs": m["exem_abs"],
        "rel_f": m["rel_f"],
        "exem_rel_f": m["exem_rel_f"],
        "err": collect.get("abs", []),
        "exem_err": collect.get("exem_abs", []),
    }


def dump_errors(res: dict, dump_dir: str) -> None:
    """Persist per-param-instance errors so runs scored in SEPARATE processes can
    still be compared with a paired test later (see outputs/compare_dumps.py)."""
    import numpy as np
    d = Path(dump_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{res['run']}.npz",
             err=np.asarray(res["err"], dtype=np.float64),
             exem_err=np.asarray(res["exem_err"], dtype=np.float64),
             abs=np.asarray([res["abs"]]),
             exem_abs=np.asarray([res["exem_abs"]]),
             rel_f=np.asarray([res["rel_f"]]),
             exem_rel_f=np.asarray([res["exem_rel_f"]]))


def paired(a: dict, b: dict, key: str = "err") -> tuple[float, float, float]:
    """Mean difference (a - b), its standard error, and the t statistic.

    Positive diff means `a` has LARGER error, i.e. is worse.
    """
    xa, xb = a[key], b[key]
    n = min(len(xa), len(xb))
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    d = [xa[i] - xb[i] for i in range(n)]
    mean = sum(d) / n
    var = sum((v - mean) ** 2 for v in d) / (n - 1)
    se = math.sqrt(var / n)
    return mean, se, (mean / se if se > 0 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directory names under outputs/train_runs/")
    ap.add_argument("--n_samples", type=int, default=0, help="0 = full val set")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--ckpt", type=str, default="best", choices=["best", "latest"])
    ap.add_argument("--stratified", type=int, default=0,
                    help="if >0, score this many val samples per held-out CHARACTER "
                         "(the trainer's representative subset) instead of a prefix")
    ap.add_argument("--dump", type=str, default="",
                    help="dir to save per-param-instance errors (.npz) for later "
                         "paired comparison across separately-scored runs")
    ap.add_argument("--residual_scale", type=float, default=1.0,
                    help="rewrite the prediction as exem + w*residual; "
                         "w=1 is the trained model, w=0 is the exem prior")
    ap.add_argument("--sweep", type=str, default="",
                    help="comma-separated w values; loads each checkpoint ONCE and "
                         "scores it at every w (cheap trade-off curve, no retraining)")
    ap.add_argument("--rig_loo", action="store_true",
                    help="V11a: score with the TARGET motion removed from the 96d "
                         "rig feature. The model is NOT retrained - this isolates "
                         "how much of a run's score came from the rig leak.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sweeps = [float(x) for x in args.sweep.split(",") if x.strip()] if args.sweep else []
    override = {"rig_loo": True} if args.rig_loo else None
    res = []
    for r in args.runs:
        ctx = load_ctx(r, device, args.n_samples, args.ckpt, args.stratified,
                       override)
        for w in (sweeps if sweeps else [args.residual_scale]):
            one = evaluate(ctx, w)
            if args.dump:
                dump_errors(one, args.dump)
            res.append(one)

    n = res[0]["n_samples"]
    print(f"\n=== offline eval | val samples scored: {n} | ckpt_{args.ckpt}.pt ===")
    print(f"{'run':<32}{'w':>6}{'ep':>4}{'abs_mae':>10}{'exem_abs':>10}{'vs exem':>10}"
          f"{'rel_f':>9}{'exem_rel_f':>12}{'n_inst':>9}")
    print("-" * 98)
    # sweeping -> group by run and order by w; comparing runs -> rank by abs_mae
    order = (lambda x: (x["base"], x["w"])) if sweeps else (lambda x: x["abs"])
    for r in sorted(res, key=order):
        exem = r["exem_abs"]
        gain = (exem - r["abs"]) / exem * 100 if exem else float("nan")
        print(f"{r['run']:<32}{r['w']:>6.2f}{r['epoch']:>4}{r['abs']:>10.4f}"
              f"{exem:>10.4f}{gain:>9.1f}%{r['rel_f']:>9.4f}"
              f"{r['exem_rel_f']:>12.4f}{len(r['err']):>9}")

    if len(res) > 1:
        print("\n=== paired comparisons (positive = first row is WORSE) ===")
        base = min(res, key=lambda x: x["abs"])
        for r in res:
            if r is base:
                continue
            mean, se, t = paired(r, base, "err")
            verdict = "significant" if abs(t) > 1.96 else "not significant"
            print(f"{r['run']:<32} - {base['run']:<32} "
                  f"diff={mean:+.4f}  se={se:.4f}  t={t:+.2f}  ({verdict})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
