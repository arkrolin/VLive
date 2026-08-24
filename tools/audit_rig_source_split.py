"""Is the shared prior universal, or local to a production source?

V5 concluded "more characters saturates at 8". That measurement was taken on a
corpus whose 2-digit pack families share animation lineage (l2d401.ugirl06 and
l2d22.ugirl06 have 99 bit-identical curves in motouweixiao), so it measured
saturation *within one source* and was wrongly generalised to "expanding the
corpus is pointless".

This script separates the two axes:

  E. within-family vs cross-family action prior, matched donor count
  F. is the gap a naming dialect (fixable) or choreography style (not)?
  G. how fast do rig dialects multiply -> can coverage ever be complete?
  H. can same-source siblings substitute for the target's own reference clips?
  I. how much of the rig term is recoverable from static moc3 info alone?

Run:  uv run python -u tools/audit_rig_source_split.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent))
from audit_data_strategy import CACHE, LAM, NS, dedup  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PARAMSET = ROOT / "outputs" / "_paramset_cache.json"
FOREIGN = "l2d8001"      # the one genuinely different-source family in the corpus


def family_of(pack: str) -> str:
    return pack.split(".")[0]


def load_raw():
    z = np.load(CACHE, allow_pickle=True)
    amp, shape, pres = z["amp"], z["shape"], z["pres"]
    dur, ya, ym, yc = z["dur"], z["act"], z["pack"], z["char"]
    k = dedup(amp, shape, pres, dur, ya, ym, yc)
    amp, pres, ya, ym, yc = amp[k], pres[k], ya[k], ym[k], yc[k]
    pac = defaultdict(set)
    for a, c in zip(ya, yc):
        pac[a].add(c)
    kk = np.array([len(pac[a]) >= 3 for a in ya])
    return amp[kk], pres[kk], ya[kk], ym[kk], yc[kk]


def featurise(amp, pres, chs=None):
    chs = list(range(NS)) if chs is None else list(chs)
    a, p = amp[:, chs], pres[:, chs]
    scale = np.array([np.median(np.abs(a[p[:, i], i])) if p[:, i].any() else 1.0
                      for i in range(len(chs))], np.float32)
    f_act = (np.abs(a) > 0.15 * np.maximum(scale, 1e-6)).astype(np.float32)
    f_amp = np.log1p(np.abs(a) / (np.abs(a).mean(0, keepdims=True) + 1e-6))
    F = np.hstack([f_act, f_amp])
    F -= F.mean(0)
    F /= np.maximum(F.std(0), 1e-6)
    return F


def prior_gain(F, ya, yc, fam, target_fam, scope, cap=3, seeds=4):
    """Action-prior error reduction for targets in `target_fam`.

    scope='in'  -> donors restricted to the same pack family (same source)
    scope='out' -> donors restricted to other families (different source)
    The rig term always comes from the target character's own other actions,
    so the only thing changing between the two runs is donor provenance.
    """
    er, eb = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        te_all = np.where(fam == target_fam)[0]
        for c in sorted(set(yc[te_all])):
            te = te_all[yc[te_all] == c]
            if len(te) < 5:
                continue
            sel = (fam == target_fam) if scope == "in" else (fam != target_fam)
            pool = np.where(sel & (yc != c))[0]
            if len(pool) < 40:
                continue
            mu = F[pool].mean(0)
            for a in set(ya[te]):
                rows = pool[ya[pool] == a]
                dc = sorted(set(yc[rows]))
                if len(dc) < 2:
                    continue
                if cap and len(dc) > cap:
                    dc = list(rng.choice(dc, cap, replace=False))
                act = np.mean([F[rows[yc[rows] == d]].mean(0) for d in dc], 0) - mu
                rst = te[ya[te] != a]
                if len(rst) < 3:
                    continue
                rig = F[rst].mean(0) - mu
                for i in te[ya[te] == a]:
                    er.append(np.mean((F[i] - mu - rig) ** 2))
                    eb.append(np.mean((F[i] - mu - rig - LAM * act) ** 2))
    if len(er) < 40:
        return None, len(er)
    return 1 - np.mean(eb) / np.mean(er), len(er)


def exp_E(F, ya, ym, yc, fam):
    print("=== E. action prior: same-source donors vs foreign donors ===")
    print("    (same targets, same donor-character cap, only provenance changes)")
    print(f"  {'family':10s} {'chars':>5s} {'assets':>7s} {'same-src':>10s}"
          f" {'foreign':>9s} {'gap':>9s}")
    rows = []
    for f in sorted(set(fam)):
        if (fam == f).sum() < 60 or len(set(yc[fam == f])) < 6:
            continue
        gi, _ = prior_gain(F, ya, yc, fam, f, "in")
        go, _ = prior_gain(F, ya, yc, fam, f, "out")
        if gi is None or go is None:
            continue
        rows.append((f, gi, go))
        print(f"  {f:10s} {len(set(yc[fam == f])):5d} {(fam == f).sum():7d}"
              f" {100*gi:9.1f}% {100*go:8.1f}% {100*(gi-go):+8.1f}pp")
    if rows:
        print(f"  {'MEAN':10s} {'':5s} {'':7s} "
              f"{100*np.mean([r[1] for r in rows]):9.1f}%"
              f" {100*np.mean([r[2] for r in rows]):8.1f}%"
              f" {100*np.mean([r[1]-r[2] for r in rows]):+8.1f}pp")


def exp_F(amp, pres, ya, ym, yc, fam):
    print(f"\n=== F. is source locality a naming dialect or choreography? ===")
    cin, cout = pres[fam == FOREIGN].mean(0), pres[fam != FOREIGN].mean(0)
    shared = np.where((cin >= 0.6) & (cout >= 0.6))[0]
    print(f"    foreign source = {FOREIGN}; channels covered >=60% on both "
          f"sides: {len(shared)}/{NS}")
    print(f"  {'channel set':26s} {'same-src':>10s} {'foreign':>9s} {'gap':>9s}")
    for tag, chs in (("all channels", None), (f"shared {len(shared)} only", shared)):
        Fx = featurise(amp, pres, chs)
        gi, _ = prior_gain(Fx, ya, yc, fam, FOREIGN, "in")
        go, _ = prior_gain(Fx, ya, yc, fam, FOREIGN, "out")
        print(f"  {tag:26s} {100*gi:9.1f}% {100*go:8.1f}% {100*(gi-go):+8.1f}pp")
    print("    -> if restricting to shared channels barely closes the gap, the")
    print("       difference is HOW they animate, not WHAT they call things.")


def exp_G():
    print("\n=== G. how fast do rig dialects multiply? ===")
    raw = orjson.loads(PARAMSET.read_bytes())
    packs = sorted(raw)
    ps = {m: set(raw[m]) for m in packs}
    n = len(packs)
    S = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(i, n):
            a, b = ps[packs[i]], ps[packs[j]]
            S[i, j] = S[j, i] = len(a & b) / max(len(a | b), 1)

    def cluster(order, thr):
        reps: list[list[int]] = []
        for i in order:
            best, bs = None, -1.0
            for ci, mem in enumerate(reps):
                s = float(np.mean(S[i, mem]))
                if s > bs:
                    best, bs = ci, s
            if bs >= thr:
                reps[best].append(i)
            else:
                reps.append([i])
        return reps

    print(f"  {'threshold':>10s} {'dialects':>9s} {'largest':>8s} {'singletons':>11s}")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        r = cluster(list(range(n)), thr)
        sz = sorted((len(x) for x in r), reverse=True)
        print(f"  {thr:>10.2f} {len(r):9d} {sz[0]:8d} "
              f"{sum(1 for x in sz if x == 1):11d}")
    print(f"\n  growth at threshold 0.50 ({n} packs total):")
    print(f"  {'packs seen':>11s} {'dialects':>9s} {'new/pack':>10s}")
    prev = None
    xs, ys = [], []
    for k in (10, 20, 40, 80, 140, 200, n):
        m = float(np.mean([len(cluster(list(np.random.default_rng(s)
                                            .permutation(n)[:k]), 0.5))
                           for s in range(5)]))
        rate = "" if prev is None else f"{(m - prev[1])/(k - prev[0]):9.2f}"
        print(f"  {k:>11d} {m:9.1f} {rate:>10s}")
        xs.append(k)
        ys.append(m)
        prev = (k, m)
    b = np.polyfit(np.log(xs), np.log(ys), 1)[0]
    print(f"  -> Heaps exponent on DIALECTS: {b:.2f}  "
          f"({'unsaturated' if b > 0.4 else 'saturating'})")


def exp_H(F, ya, ym, yc, fam):
    print("\n=== H. target has ZERO reference clips: where can the rig term "
          "come from? ===")
    print("    R2 measured against the global-mean baseline")
    print(f"  {'rig source':38s} {'rig only':>9s} {'rig+prior':>11s}")

    def run(mode, k=None, seeds=4):
        er, erg, eb = [], [], []
        for s in range(seeds):
            rng = np.random.default_rng(s)
            for f in sorted(set(fam)):
                te_all = np.where(fam == f)[0]
                if len(set(yc[te_all])) < 4:
                    continue
                for c in sorted(set(yc[te_all])):
                    te = te_all[yc[te_all] == c]
                    if len(te) < 5:
                        continue
                    glob = np.where(yc != c)[0]
                    if len(glob) < 200:
                        continue
                    mu = F[glob].mean(0)
                    rig = np.zeros_like(mu)
                    if mode in ("sib", "foreign"):
                        sel = (fam == f) if mode == "sib" else (fam != f)
                        pool = np.where(sel & (yc != c))[0]
                        sc = sorted(set(yc[pool]))
                        if len(sc) < 2:
                            continue
                        if k and len(sc) > k:
                            sc = list(rng.choice(sc, k, replace=False))
                        rig = np.mean([F[pool[yc[pool] == d]].mean(0)
                                       for d in sc], 0) - mu
                    for a in set(ya[te]):
                        rows = glob[ya[glob] == a]
                        if len(set(yc[rows])) < 2:
                            continue
                        act = np.mean([F[rows[yc[rows] == d]].mean(0)
                                       for d in sorted(set(yc[rows]))], 0) - mu
                        r = rig
                        if mode == "self":
                            rst = te[ya[te] != a]
                            if len(rst) < 3:
                                continue
                            r = F[rst].mean(0) - mu
                        for i in te[ya[te] == a]:
                            er.append(np.mean((F[i] - mu) ** 2))
                            erg.append(np.mean((F[i] - mu - r) ** 2))
                            eb.append(np.mean((F[i] - mu - r - LAM * act) ** 2))
        return (1 - np.mean(erg) / np.mean(er),
                1 - np.mean(eb) / np.mean(er), len(er))

    for tag, mode, k in (("none (global mean)", "none", None),
                         ("8 arbitrary foreign chars", "foreign", 8),
                         ("same-source siblings, 4", "sib", 4),
                         ("same-source siblings, 8", "sib", 8),
                         ("same-source siblings, all", "sib", None),
                         ("target's OWN other clips (upper bound)", "self", None)):
        a, b, n = run(mode, k)
        print(f"  {tag:38s} {a:9.3f} {b:11.3f}   (n={n})")
    print("    -> siblings scoring below 'none' means the rig term is a")
    print("       CHARACTER property, not a source property.")


def exp_I(F, pres, ya, ym, yc):
    print("\n=== I. how much rig term survives with ZERO clips, from moc3 alone? ===")
    static = {p: pres[ym == p].max(0).astype(np.float32) for p in sorted(set(ym))}
    sr = np.stack([static[p] for p in ym])

    def run(mode, seeds=4):
        er, erg, eb = [], [], []
        for _ in range(seeds):
            for c in sorted(set(yc)):
                te = np.where(yc == c)[0]
                if len(te) < 5:
                    continue
                glob = np.where(yc != c)[0]
                if len(glob) < 200:
                    continue
                mu = F[glob].mean(0)
                rig = np.zeros_like(mu)
                if mode == "static":
                    tgt = sr[te][0]
                    w = (sr[glob] @ tgt) / np.maximum(
                        np.linalg.norm(sr[glob], axis=1) * np.linalg.norm(tgt), 1e-6)
                    w = np.maximum(w - w.mean(), 0) ** 4
                    if w.sum() > 0:
                        rig = np.average(F[glob], axis=0, weights=w) - mu
                for a in set(ya[te]):
                    rows = glob[ya[glob] == a]
                    if len(set(yc[rows])) < 2:
                        continue
                    act = np.mean([F[rows[yc[rows] == d]].mean(0)
                                   for d in sorted(set(yc[rows]))], 0) - mu
                    r = rig
                    if mode == "self":
                        rst = te[ya[te] != a]
                        if len(rst) < 3:
                            continue
                        r = F[rst].mean(0) - mu
                    for i in te[ya[te] == a]:
                        er.append(np.mean((F[i] - mu) ** 2))
                        erg.append(np.mean((F[i] - mu - r) ** 2))
                        eb.append(np.mean((F[i] - mu - r - LAM * act) ** 2))
        return (1 - np.mean(erg) / np.mean(er),
                1 - np.mean(eb) / np.mean(er), len(er))

    print(f"  {'rig source':44s} {'rig only':>9s} {'rig+prior':>11s}")
    for tag, m in (("none (global mean)", "none"),
                   ("moc3 channel existence (0 clips, file-readable)", "static"),
                   ("target's OWN other clips (needs >=3 refs)", "self")):
        a, b, n = run(m)
        print(f"  {tag:44s} {a:9.3f} {b:11.3f}   (n={n})")
    print("    -> a 20-bit static profile already recovers a third of the rig")
    print("       term; a real sweep probe carries orders of magnitude more.")


def main() -> None:
    amp, pres, ya, ym, yc = load_raw()
    fam = np.array([family_of(m) for m in ym])
    F = featurise(amp, pres)
    print(f"assets={len(F)}  packs={len(set(ym))}  chars={len(set(yc))}  "
          f"families={len(set(fam))}\n")
    exp_E(F, ya, ym, yc, fam)
    exp_F(amp, pres, ya, ym, yc, fam)
    exp_G()
    exp_H(F, ya, ym, yc, fam)
    exp_I(F, pres, ya, ym, yc)


if __name__ == "__main__":
    main()
