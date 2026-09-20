"""Does the implementation gap come from BANK COVERAGE?

Hypothesis under test (from the V11e capacity-vs-structure diagnosis):
    The structured head can reach shape corr ~0.8556 at r=8, but only achieves
    0.7124 -> an "implementation gap" of 0.143. The suspected cause is that the
    per-(character, action) style information is only available through the
    motion bank, and the bank is thin/empty for long-tail characters, so the
    model falls back to the cross-character exem prior exactly where it must not.

This script tests it WITHOUT retraining, by stratifying the existing 614-sample
render dump on two independent coverage measures:

  A. bank_size      = how many OTHER actions this character has in the corpus
                      (the K the bank attention can actually pick from;
                       bank_k=8, so >=8 means the bank is saturated)
  B. action_shared  = how many distinct characters have this action at all
                      (1 = character-exclusive long tail)

Falsifiable predictions:
  - If the hypothesis holds, pooled shape corr rises monotonically with both.
  - If corr is flat across bins, coverage is NOT the binding constraint and the
    implementation gap must be sought elsewhere (optimisation / loss weighting).

Also reports the exem-error quartile breakdown (cheap, from the manifest):
if the model only ever "polishes" the prior, its RELATIVE improvement should be
flat while the ABSOLUTE residual concentrates in the worst-prior quartile.

Usage:
    .venv/bin/python outputs/diag_bank_coverage.py \
        outputs/render_data/abl_V11e_B1b_best.pkl \
        --bank outputs/motionbank_580_v9all.pkl \
        --ref   outputs/render_data/abl_R_all_v9_best.pkl
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# metrics -- identical definitions to diag_shape.py
# --------------------------------------------------------------------------- #
def delta(x: np.ndarray) -> np.ndarray:
    return np.diff(x, axis=-1)


def pooled_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict[str, float]:
    """Metrics pooled over every (instance-param, frame) element given."""
    out: dict[str, float] = {}
    ae = np.abs(pred - tgt)
    de = np.abs(delta(pred) - delta(tgt))
    ptp = tgt.max(axis=-1) - tgt.min(axis=-1)
    const = ptp < 1e-6
    mov = ~const
    out["n_slots"] = int(pred.size)
    out["n_moving"] = int(mov.sum())
    out["point"] = float(ae.mean())
    out["point_var"] = float(ae[mov].mean()) if mov.any() else float("nan")
    out["delta"] = float(de.mean())
    out["delta_var"] = float(de[mov].mean()) if mov.any() else float("nan")
    if mov.any():
        a = delta(pred)[mov].ravel()
        b = delta(tgt)[mov].ravel()
        out["shape_corr"] = (
            float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        )
        out["abs_dpred"] = float(np.abs(delta(pred)[mov]).mean())
        out["abs_dtgt"] = float(np.abs(delta(tgt)[mov]).mean())
        out["retain"] = out["abs_dpred"] / max(out["abs_dtgt"], 1e-12)
    else:
        out["shape_corr"] = float("nan")
        out["abs_dpred"] = out["abs_dtgt"] = out["retain"] = float("nan")
    return out


def per_instance_metrics(items: list[dict]) -> list[dict]:
    rows = []
    for it in items:
        pred = np.asarray(it["pred"], dtype=np.float64)
        tgt = np.asarray(it["target"], dtype=np.float64)
        exm = np.asarray(it["exem"], dtype=np.float64)
        ptp = tgt.max(axis=-1) - tgt.min(axis=-1)
        mov = ptp >= 1e-6
        if mov.sum() == 0:
            corr = float("nan")
        else:
            a = delta(pred)[mov].ravel()
            b = delta(tgt)[mov].ravel()
            corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        rows.append(
            dict(
                model=it["model"],
                action=it["action"],
                corr=corr,
                de_mov=float(np.abs(delta(pred) - delta(tgt))[mov].mean()) if mov.any() else float("nan"),
                de_ex_mov=float(np.abs(delta(exm) - delta(tgt))[mov].mean()) if mov.any() else float("nan"),
                err_pred=float(np.abs(pred - tgt).mean()),
                err_exem=float(np.abs(exm - tgt).mean()),
                n_move=int(mov.sum()),
                n_slot=int(pred.shape[0]),
            )
        )
    return rows


def report(title: str, groups: dict[str, list[dict]]) -> None:
    """One row per bucket. `corr` is pooled over all moving elements in the
    bucket (stable, matches diag_shape.py); `corr_inst` is the mean of the
    per-instance correlations (noisier but shows within-bucket spread)."""
    print(f"\n{'=' * 104}\n{title}\n{'=' * 104}")
    hdr = (f"{'bucket':<18}{'n_inst':>7}{'n_mov':>8}{'point':>9}{'delta':>9}"
           f"{'dMAE|mov':>10}{'corr':>9}{'retain':>9}{'corr_inst':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name, rows in groups.items():
        if not rows:
            continue
        preds = [r["_pred"] for r in rows]
        tgts = [r["_tgt"] for r in rows]
        pm = pooled_metrics(np.concatenate(preds, 0), np.concatenate(tgts, 0))
        crs = np.array([r["corr"] for r in rows if np.isfinite(r["corr"])])
        ci = crs.mean() if crs.size else float("nan")
        print(f"{name:<18}{len(rows):>7}{pm['n_moving']:>8}{pm['point']:>9.4f}{pm['delta']:>9.4f}"
              f"{pm['delta_var']:>10.4f}{pm['shape_corr']:>9.4f}{pm['retain']:>9.4f}{ci:>11.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("render", type=Path)
    ap.add_argument("--bank", type=Path, default=Path("outputs/motionbank_580_v9all.pkl"))
    ap.add_argument("--ref", type=Path, default=None)
    args = ap.parse_args()

    # ---- corpus coverage from the motion-bank keys ----
    bank = pickle.load(open(args.bank, "rb"))
    char2act: dict[str, set[str]] = defaultdict(set)
    act2char: dict[str, set[str]] = defaultdict(set)
    for k in bank.keys():
        c, a = eval(k) if isinstance(k, str) else k  # keys may be tuples or their repr
        char2act[c].add(a)
        act2char[a].add(c)
    print(f"[bank] entries={len(bank)} characters={len(char2act)} actions={len(act2char)}")

    # ---- render dump ----
    ren = pickle.load(open(args.render, "rb"))
    items = ren["items"]
    print(f"[render] {args.render.name} run={ren['run']} ckpt={ren['ckpt']} "
          f"epoch={ren['epoch']} n={ren['n']}")

    rows = per_instance_metrics(items)

    # attach pooled arrays for bucketing (cast once)
    for r, it in zip(rows, items):
        r["_pred"] = np.asarray(it["pred"], dtype=np.float64)
        r["_tgt"] = np.asarray(it["target"], dtype=np.float64)

    # ---- coverage diagnostics ----
    models = {r["model"] for r in rows}
    actions = {r["action"] for r in rows}
    miss_m = [m for m in models if m not in char2act]
    miss_a = [a for a in actions if a not in act2char]
    print(f"[coverage] render has {len(models)} characters / {len(actions)} actions")
    print(f"[coverage] characters missing from bank: {len(miss_m)} {miss_m[:5]}")
    print(f"[coverage] actions   missing from bank: {len(miss_a)} {miss_a[:5]}")

    bs = np.array([len(char2act.get(r["model"], set()) - {r["action"]}) for r in rows])
    ash = np.array([len(act2char.get(r["action"], set())) for r in rows])
    print(f"[bank_size]  min={bs.min()} p25={np.percentile(bs,25):.0f} med={np.median(bs):.0f} "
          f"p75={np.percentile(bs,75):.0f} max={bs.max()}  (bank_k=8)")
    print(f"[action_shared] min={ash.min()} p25={np.percentile(ash,25):.0f} med={np.median(ash):.0f} "
          f"p75={np.percentile(ash,75):.0f} max={ash.max()}")

    # ---- stratification A: bank_size ----
    edges = [0, 1, 3, 8, 10**9]
    labels = ["0 (empty)", "1-2", "3-7", "8+ (≥bank_k)"]
    ga: dict[str, list[dict]] = {l: [] for l in labels}
    for r, v in zip(rows, bs):
        for i in range(len(labels)):
            if edges[i] <= v < edges[i + 1]:
                ga[labels[i]].append(r)
                break
    report("A. stratified by BANK SIZE (other actions this character has)", ga)

    # ---- stratification B: action sharedness ----
    edges = [0, 1, 2, 5, 10**9]
    labels = ["1 (exclusive)", "2", "3-4", "5+"]
    gb: dict[str, list[dict]] = {l: [] for l in labels}
    for r, v in zip(rows, ash):
        for i in range(len(labels)):
            if edges[i] <= v < edges[i + 1]:
                gb[labels[i]].append(r)
                break
    report("B. stratified by ACTION SHAREDNESS (characters having this action)", gb)

    # ---- stratification C: how bad the prior is (mechanism probe) ----
    ee = np.array([r["err_exem"] for r in rows])
    print("\n[err_exem distribution] " + " ".join(
        f"{k}={np.percentile(ee, p):.4f}" for k, p in
        [("min", 0), ("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90), ("max", 100)]))
    print(f"[err_exem distribution] frac==0: {(ee <= 1e-9).mean():.3f}   mean={ee.mean():.4f}")
    # BALANCED quartiles by rank (percentile edges collapse when the
    # distribution is atomised at 0, which silently produced 2 empty buckets).
    labels = ["Q1 best prior", "Q2", "Q3", "Q4 worst prior"]
    gc: dict[str, list[dict]] = {l: [] for l in labels}
    order = np.argsort(ee, kind="stable")
    q = max(len(order) // 4, 1)
    lab = np.clip(np.arange(len(order)) // q, 0, 3)
    for pos, r in zip(lab, [rows[i] for i in order]):
        gc[labels[int(pos)]].append(r)
    report("C. stratified by EXEM PRIOR ERROR (is the model only polishing the prior?)", gc)

    print("\n[C] absolute vs relative improvement over exem, by prior-error quartile:")
    print(f"{'bucket':<16}{'err_pred':>10}{'err_exem':>10}{'abs_gain':>10}{'rel_gain%':>11}")
    for l in labels:
        rows_l = gc[l]
        if not rows_l:
            continue
        ep = np.mean([r["err_pred"] for r in rows_l])
        ee_ = np.mean([r["err_exem"] for r in rows_l])
        print(f"{l:<16}{ep:>10.4f}{ee_:>10.4f}{ee_ - ep:>10.4f}{100 * (ee_ - ep) / max(ee_, 1e-12):>11.1f}")

    # ---- D. where does the per-instance correlation actually live? ----
    # diag_shape.py reports shape_corr as the MEAN of the per-instance
    # correlations (load_metrics -> avg). The POOLED correlation over all moving
    # deltas is much higher -- so the headline number is dragged down by a
    # subset of hard/sparse instances, not by a uniform shortfall.
    cr = np.array([r["corr"] for r in rows])
    nm = np.array([r["n_move"] for r in rows])
    ok = np.isfinite(cr)
    print("\n" + "=" * 104)
    print("D. per-instance shape corr: distribution and what it correlates with")
    print("=" * 104)
    print("deciles of per-instance corr: " + " ".join(
        f"p{p}={np.nanpercentile(cr, p):.3f}" for p in [5, 10, 25, 50, 75, 90, 95]))
    print(f"frac corr < 0.25 : {(cr[ok] < 0.25).mean():.3f}"
          f"   frac corr < 0.0 : {(cr[ok] < 0.0).mean():.3f}"
          f"   frac corr > 0.9 : {(cr[ok] > 0.9).mean():.3f}")
    with np.errstate(invalid="ignore"):
        rho = np.corrcoef(nm[ok], cr[ok])[0, 1]
    print(f"corr( n_moving_channels , per-instance corr ) = {rho:+.3f}")
    for lo, hi, lab in [(0, 3, "1-2 mov"), (3, 10, "3-9 mov"), (10, 30, "10-29 mov"),
                        (30, 10**9, "30+ mov")]:
        sel = ok & (nm >= lo) & (nm < hi)
        if sel.sum():
            print(f"   {lab:<10} n={sel.sum():>4}  mean corr={np.nanmean(cr[sel]):.4f}")

    print("\nworst 15 instances by per-instance corr:")
    hdr = f"{'model':<26}{'action':<22}{'corr':>8}{'n_mov':>7}{'bank':>6}{'shared':>8}{'err_pred':>10}{'err_exem':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r, b, s in sorted(zip(rows, bs, ash), key=lambda t: t[0]["corr"])[:15]:
        print(f"{r['model']:<26}{r['action'][:20]:<22}{r['corr']:>8.3f}{r['n_move']:>7}"
              f"{b:>6}{s:>8}{r['err_pred']:>10.3f}{r['err_exem']:>10.3f}")

    # ---- reference comparison (optional) ----
    if args.ref is not None and args.ref.exists():
        ref = pickle.load(open(args.ref, "rb"))
        rrows = per_instance_metrics(ref["items"])
        print(f"\n[ref] {args.ref.name} run={ref['run']} n={ref['n']}: "
              f"mean corr(inst)={np.nanmean([r['corr'] for r in rrows]):.4f} "
              f"mean err_pred={np.mean([r['err_pred'] for r in rrows]):.4f}")
        # same BANK-SIZE stratification for the reference arm (own edges/labels)
        rbs = np.array([len(char2act.get(r["model"], set()) - {r["action"]}) for r in rrows])
        r_edges = [0, 1, 3, 8, 10**9]
        r_labels = ["0 (empty)", "1-2", "3-7", "8+ (≥bank_k)"]
        print(f"{'bucket':<18}{'n_inst':>7}{'corr_inst(REF)':>16}")
        for i, l in enumerate(r_labels):
            sel = [r for r, v in zip(rrows, rbs) if r_edges[i] <= v < r_edges[i + 1]]
            if not sel:
                continue
            print(f"{l:<18}{len(sel):>7}{np.nanmean([r['corr'] for r in sel]):>16.4f}")


if __name__ == "__main__":
    main()
