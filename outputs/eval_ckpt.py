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
                "action_name_max_len")
_CACHE: dict[tuple, tuple] = {}


def get_corpus(cfg):
    key = tuple(getattr(cfg, k, None) for k in _CORPUS_KEYS)
    if key not in _CACHE:
        train_ds = Live2DDataset(cfg, split="train")
        val_ds = Live2DDataset(cfg, split="val")
        g_lo, g_hi, per_lo, per_hi = compute_range(train_ds, n=400)
        _CACHE[key] = (train_ds, val_ds, g_lo, g_hi, per_lo, per_hi)
    return _CACHE[key]


def evaluate(run: str, device: torch.device, n_samples: int,
             which: str = "best", stratified: int = 0, dump: str = "") -> dict:
    ckpt_path = RUNS / run / f"ckpt_{which}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = PipelineConfig()
    cfg.__dict__.update(sd.get("cfg", {}))          # restore the run's own config

    train_ds, val_ds, g_lo, g_hi, per_lo, per_hi = get_corpus(cfg)
    model = Live2DModel(cfg, train_ds.word2idx, len(train_ds.action2idx),
                        cfg.max_tokens, train_ds.action_char2idx)
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
    collect: dict = {}
    with torch.no_grad():
        m = recon_metrics(model, loader, device, cfg, per_lo, per_hi,
                          g_lo, g_hi, alphabar, n_samples=cap, steps=50,
                          collect=collect)
    return {
        "run": run,
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
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    res = []
    for r in args.runs:
        one = evaluate(r, device, args.n_samples, args.ckpt, args.stratified)
        if args.dump:
            dump_errors(one, args.dump)
        res.append(one)

    n = res[0]["n_samples"]
    print(f"\n=== offline eval | val samples scored: {n} | ckpt_{args.ckpt}.pt ===")
    print(f"{'run':<28}{'ep':>4}{'abs_mae':>10}{'exem_abs':>10}{'vs exem':>10}"
          f"{'rel_f':>9}{'exem_rel_f':>12}{'n_inst':>9}")
    print("-" * 88)
    for r in sorted(res, key=lambda x: x["abs"]):
        exem = r["exem_abs"]
        gain = (exem - r["abs"]) / exem * 100 if exem else float("nan")
        print(f"{r['run']:<28}{r['epoch']:>4}{r['abs']:>10.4f}{exem:>10.4f}"
              f"{gain:>9.1f}%{r['rel_f']:>9.4f}{r['exem_rel_f']:>12.4f}"
              f"{len(r['err']):>9}")

    if len(res) > 1:
        print("\n=== paired comparisons (positive = first row is WORSE) ===")
        base = min(res, key=lambda x: x["abs"])
        for r in res:
            if r is base:
                continue
            mean, se, t = paired(r, base, "err")
            verdict = "significant" if abs(t) > 1.96 else "not significant"
            print(f"{r['run']:<28} - {base['run']:<28} "
                  f"diff={mean:+.4f}  se={se:.4f}  t={t:+.2f}  ({verdict})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
