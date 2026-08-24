"""Behavioural fingerprints for Live2D parameters + cross-model retrieval test.

The question this answers: *can a model tell what a parameter controls, when the
id is meaningless (PARAM_14) and no cdi3 names exist?*

Instead of arguing, measure. For every (model, param) we build a fingerprint
from how the parameter is USED across that model's hand-authored motions
(value distribution, activation sparsity, time-scale, spectral shape, segment
types). Then we run cross-model nearest-neighbour retrieval on parameters whose
identity we DO know (shared standard ids) and report recall@1/@5.

Recall high  -> behaviour alone identifies the parameter; geometry is a bonus.
Recall low   -> behaviour is ambiguous; geometric probing is mandatory there.

Usage:
    uv run --with numpy python tools/param_fingerprint.py --limit 80
    uv run --with numpy python tools/param_fingerprint.py --limit 80 --dump fp.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from motion_curves import load_motion  # noqa: E402

FPS = 30.0
EPS = 1e-6

FEAT_NAMES = [
    "mean", "std", "p05", "p50", "p95", "iqr", "skew", "kurtosis",
    "active_ratio", "mode_dwell", "zero_dwell", "n_levels",
    "d_mean", "d_p95", "d_max", "dd_mean",
    "ac_lag1", "ac_lag3", "ac_lag10", "ac_lag30",
    "spec_centroid", "spec_lowband", "spec_peak_hz", "spec_flatness",
    "event_rate", "event_dur", "bidirectional", "range_use",
    "seg_linear", "seg_bezier", "seg_stepped", "seg_invstep",
]


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 1:
        return 0.0
    a, b = x[:-lag], x[lag:]
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > EPS else 0.0


def fingerprint(vals: list[np.ndarray], segh: np.ndarray) -> np.ndarray:
    """vals: list of per-motion 1-D trajectories (raw units). -> feature vector."""
    cat = np.concatenate(vals)
    lo, hi = float(cat.min()), float(cat.max())
    rng = hi - lo
    if rng < EPS:                       # constant parameter across the corpus
        f = np.zeros(len(FEAT_NAMES), dtype=np.float32)
        f[FEAT_NAMES.index("mean")] = 0.5
        return f
    n = [(v - lo) / rng for v in vals]  # per-param min-max -> [0,1]
    x = np.concatenate(n)

    d_all, dd_all, ac = [], [], defaultdict(list)
    spec_c, spec_l, spec_p, spec_f = [], [], [], []
    ev_cnt, ev_len, tot_t = 0, [], 0.0
    for v in n:
        if len(v) < 4:
            continue
        d = np.abs(np.diff(v))
        d_all.append(d)
        dd_all.append(np.abs(np.diff(v, 2)))
        for lg in (1, 3, 10, 30):
            ac[lg].append(_autocorr(v, lg))
        # spectrum of the mean-removed signal
        w = v - v.mean()
        if np.abs(w).max() > 1e-4:
            sp = np.abs(np.fft.rfft(w)) ** 2
            fr = np.fft.rfftfreq(len(w), d=1.0 / FPS)
            s = sp.sum()
            if s > EPS:
                spec_c.append(float((sp * fr).sum() / s))
                spec_l.append(float(sp[fr <= 1.0].sum() / s))
                spec_p.append(float(fr[int(np.argmax(sp))]))
                pos = sp[sp > EPS]
                if len(pos) > 1:
                    spec_f.append(float(np.exp(np.log(pos).mean()) / pos.mean()))
        # "events": excursions away from the dominant resting level
        hist, edges = np.histogram(v, bins=20, range=(0, 1))
        rest = 0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1])
        off = np.abs(v - rest) > 0.25
        if off.any():
            edge = np.diff(off.astype(np.int8))
            ev_cnt += int((edge == 1).sum()) + (1 if off[0] else 0)
            ev_len.append(float(off.mean() * len(v) / FPS))
        tot_t += len(v) / FPS

    d = np.concatenate(d_all) if d_all else np.zeros(1)
    dd = np.concatenate(dd_all) if dd_all else np.zeros(1)
    hist, edges = np.histogram(x, bins=20, range=(0, 1))
    mode_bin = int(np.argmax(hist))
    mu, sd = float(x.mean()), float(x.std())
    z = (x - mu) / (sd + EPS)
    segsum = float(segh.sum())
    segp = segh / segsum if segsum > 0 else np.zeros(4, dtype=np.float32)

    f = {
        "mean": mu, "std": sd,
        "p05": float(np.percentile(x, 5)), "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "skew": float((z ** 3).mean()), "kurtosis": float((z ** 4).mean()),
        "active_ratio": float((d > 1e-3).mean()),
        "mode_dwell": float(hist[mode_bin] / max(hist.sum(), 1)),
        "zero_dwell": float((x < 0.02).mean()),
        "n_levels": float(min(len(np.unique(np.round(x, 2))), 100) / 100.0),
        "d_mean": float(d.mean()), "d_p95": float(np.percentile(d, 95)),
        "d_max": float(d.max()), "dd_mean": float(dd.mean()),
        "ac_lag1": float(np.mean(ac[1])) if ac[1] else 0.0,
        "ac_lag3": float(np.mean(ac[3])) if ac[3] else 0.0,
        "ac_lag10": float(np.mean(ac[10])) if ac[10] else 0.0,
        "ac_lag30": float(np.mean(ac[30])) if ac[30] else 0.0,
        "spec_centroid": float(np.mean(spec_c)) / 15.0 if spec_c else 0.0,
        "spec_lowband": float(np.mean(spec_l)) if spec_l else 0.0,
        "spec_peak_hz": float(np.mean(spec_p)) / 15.0 if spec_p else 0.0,
        "spec_flatness": float(np.mean(spec_f)) if spec_f else 0.0,
        "event_rate": float(ev_cnt / max(tot_t, EPS)),
        "event_dur": float(np.mean(ev_len)) if ev_len else 0.0,
        "bidirectional": float(min((x < mu - sd).mean(), (x > mu + sd).mean()) * 2),
        "range_use": float(min(rng, 10.0) / 10.0),
        "seg_linear": float(segp[0]), "seg_bezier": float(segp[1]),
        "seg_stepped": float(segp[2]), "seg_invstep": float(segp[3]),
    }
    return np.array([f[k] for k in FEAT_NAMES], dtype=np.float32)


def build(root: Path, limit: int | None):
    models = sorted(d for d in root.iterdir() if d.is_dir())
    if limit:
        # spread the sample across the corpus rather than taking a prefix
        step = max(len(models) // limit, 1)
        models = models[::step][:limit]

    per_model: dict[str, dict[str, np.ndarray]] = {}
    for mi, md in enumerate(models):
        mdir = md / "motions"
        if not mdir.is_dir():
            continue
        traj: dict[str, list[np.ndarray]] = defaultdict(list)
        segs: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.float32))
        files = list({f.name: f for f in mdir.glob("*.json")}.values())
        for mf in files:
            curves, segstats, _ = load_motion(mf, FPS)
            for pid, arr in curves.items():
                traj[pid].append(arr)
                segs[pid] += segstats.get(pid, np.zeros(4, dtype=np.float32))
        if not traj:
            continue
        fps_ = {}
        for pid, vs in traj.items():
            if sum(len(v) for v in vs) < 30:
                continue
            fps_[pid] = fingerprint(vs, segs[pid])
        if fps_:
            per_model[md.name] = fps_
        if (mi + 1) % 20 == 0:
            print(f"  ... {mi+1}/{len(models)} models", flush=True)
    return per_model


def retrieval_test(per_model, min_models=8, topk=(1, 5)):
    """For ids shared by many models: query model A's fingerprint, retrieve
    inside model B's full parameter set, check whether top-1 is the same id."""
    id_models = defaultdict(list)
    for mname, d in per_model.items():
        for pid in d:
            id_models[pid].append(mname)
    shared = [p for p, ms in id_models.items() if len(ms) >= min_models]

    # z-score normalisation over the whole population
    allf = np.stack([v for d in per_model.values() for v in d.values()])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6

    names = sorted(per_model)
    mats = {m: (np.stack([per_model[m][p] for p in sorted(per_model[m])]) - mu) / sd
            for m in names}
    keys = {m: sorted(per_model[m]) for m in names}

    # semantic bucket of every id, for the "is it the RIGHT KIND?" evaluation
    from classify_hetero import classify
    bucket = {}
    for m in names:
        for p in keys[m]:
            bucket[p] = classify(p)[0]

    rng = np.random.default_rng(0)
    results = defaultdict(lambda: {"n": 0, "hit1": 0, "hit5": 0, "rank": [],
                                   "bhit1": 0, "bhit5": 0})
    trials = 0
    for pid in shared:
        ms = id_models[pid]
        for _ in range(min(20, len(ms))):
            a, b = rng.choice(ms, 2, replace=False)
            if pid not in per_model[b]:
                continue
            q = (per_model[a][pid] - mu) / sd
            M = mats[b]
            dist = np.linalg.norm(M - q, axis=1)
            order = np.argsort(dist)
            gt = keys[b].index(pid)
            rank = int(np.where(order == gt)[0][0])
            gb = bucket[pid]
            top5 = [bucket[keys[b][i]] for i in order[:5]]
            r = results[pid]
            r["n"] += 1
            r["hit1"] += int(rank == 0)
            r["hit5"] += int(rank < 5)
            r["bhit1"] += int(top5[0] == gb)
            r["bhit5"] += int(gb in top5)
            r["rank"].append(rank)
            trials += 1
    return results, trials, shared


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    print(f"[1/2] building fingerprints (limit={args.limit}) ...", flush=True)
    per_model = build(Path(args.root), args.limit)
    n_params = sum(len(d) for d in per_model.values())
    print(f"      {len(per_model)} models, {n_params} (model,param) fingerprints\n")

    print("[2/2] cross-model retrieval ...", flush=True)
    res, trials, shared = retrieval_test(per_model)
    tot_n = sum(r["n"] for r in res.values())
    tot1 = sum(r["hit1"] for r in res.values())
    tot5 = sum(r["hit5"] for r in res.values())
    tb1 = sum(r["bhit1"] for r in res.values())
    tb5 = sum(r["bhit5"] for r in res.values())
    print(f"      shared ids tested: {len(shared)}, trials: {trials}")
    print(f"\n=== EXACT-ID   recall@1 = {100*tot1/max(tot_n,1):.1f}%   "
          f"recall@5 = {100*tot5/max(tot_n,1):.1f}%   "
          f"(chance ~ {100/np.mean([len(d) for d in per_model.values()]):.2f}%)")
    print(f"=== SEMANTIC-BUCKET  top1 = {100*tb1/max(tot_n,1):.1f}%   "
          f"in-top5 = {100*tb5/max(tot_n,1):.1f}%    "
          f"<- 'did we land on the right KIND of parameter'\n")

    rows = [(p, r["hit1"] / r["n"], r["hit5"] / r["n"], np.median(r["rank"]), r["n"])
            for p, r in res.items() if r["n"] >= 5]
    rows.sort(key=lambda x: -x[1])
    print(f"{'param':32s} {'R@1':>6s} {'R@5':>6s} {'medRank':>8s} {'n':>4s}")
    print("-" * 62)
    for p, r1, r5, mr, n in rows[:22]:
        print(f"{p:32s} {100*r1:5.0f}% {100*r5:5.0f}% {mr:8.0f} {n:4d}")
    print("   ... worst ...")
    for p, r1, r5, mr, n in rows[-18:]:
        print(f"{p:32s} {100*r1:5.0f}% {100*r5:5.0f}% {mr:8.0f} {n:4d}")

    if args.dump:
        np.savez_compressed(
            args.dump,
            feat_names=np.array(FEAT_NAMES),
            models=np.array(sorted(per_model)),
            **{f"{m}::{p}": v for m, d in per_model.items() for p, v in d.items()},
        )
        print(f"\n[SUCCESS] fingerprints -> {args.dump}")


if __name__ == "__main__":
    main()
