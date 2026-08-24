"""Does adding HETEROGENEOUS sources help, or only more donors?

The earlier scaling audit answered "does donor *identity* matter" by ranking
donors by similarity **to the target**. That test cannot see the mechanism the
user is asking about: expanding to genuinely different sources changes the
*internal spread* of the donor pool, not its similarity to any one target.

Four experiments here:

  A. how much spread does the corpus actually have (3 metrics)
  B. fixed donor count k, vary the donor set's INTERNAL diversity
  C. saturation curve within one pack family vs across families
  D. hold out an entire pack family as a stand-in "new source"

Run:  uv run python -u tools/audit_source_diversity.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent))
from audit_data_strategy import LAM, load_feats  # noqa: E402
from audit_leakage import char_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PARAMSET = ROOT / "outputs" / "_paramset_cache.json"


def family_of(pack: str) -> str:
    return pack.split(".")[0]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


# ----------------------------------------------------------------- A. spread
def measure_spread(F, ya, ym, yc):
    raw = orjson.loads(PARAMSET.read_bytes())
    by_char = defaultdict(set)
    vocab = defaultdict(set)
    for p, ps in raw.items():
        by_char[char_of(p)] |= set(ps)
    for a, c in zip(ya, yc):
        vocab[c].add(a)

    chars = sorted(set(yc) & set(by_char))
    beh = np.stack([F[yc == c].mean(0) for c in chars])
    beh /= np.maximum(np.linalg.norm(beh, axis=1, keepdims=True), 1e-6)

    js, jv, bs = [], [], []
    for i in range(len(chars)):
        for j in range(i + 1, len(chars)):
            js.append(jaccard(by_char[chars[i]], by_char[chars[j]]))
            jv.append(jaccard(vocab[chars[i]], vocab[chars[j]]))
            bs.append(float(beh[i] @ beh[j]))
    js, jv, bs = np.array(js), np.array(jv), np.array(bs)

    print("=== A. how much spread does the corpus actually contain ===")
    print(f"  {'metric':26s} {'p05':>7s} {'median':>7s} {'p95':>7s} {'range':>7s}")
    for name, v in (("rig param Jaccard", js),
                    ("action vocab Jaccard", jv),
                    ("behaviour cosine", bs)):
        lo, mid, hi = np.percentile(v, [5, 50, 95])
        print(f"  {name:26s} {lo:7.3f} {mid:7.3f} {hi:7.3f} {hi - lo:7.3f}")
    return chars, beh, by_char


# ------------------------------------------- B. donor-set internal diversity
def donor_set_diversity(F, ya, ym, yc, chars, beh, ks=(3, 5, 8)):
    """Fixed donor count, vary how spread out the donors are among themselves."""
    pos = {c: i for i, c in enumerate(chars)}
    idx = {c: np.where(yc == c)[0] for c in chars}
    D = 1.0 - beh @ beh.T                       # behavioural distance

    def pick_spread(cand, k, rng):
        """greedy farthest-point: maximise internal spread"""
        sel = [cand[rng.integers(len(cand))]]
        while len(sel) < k:
            best, bd = None, -1
            for x in cand:
                if x in sel:
                    continue
                d = min(D[pos[x], pos[s]] for s in sel)
                if d > bd:
                    best, bd = x, d
            sel.append(best)
        return sel

    def pick_tight(cand, k, rng):
        """seed + its k-1 nearest: minimise internal spread"""
        seed = cand[rng.integers(len(cand))]
        rest = sorted((x for x in cand if x != seed),
                      key=lambda x: D[pos[seed], pos[x]])
        return [seed] + rest[: k - 1]

    print("\n=== B. fixed donor count, vary the donor set's INTERNAL spread ===")
    print("    (this is the mechanism 'add heterogeneous sources' actually changes)")
    print(f"  {'k':>3s} {'random':>10s} {'tight cluster':>15s} {'max spread':>12s}"
          f" {'spread-tight':>14s}")
    for k in ks:
        res = {m: ([], []) for m in ("rand", "tight", "spread")}
        internal = {m: [] for m in ("rand", "tight", "spread")}
        for seed in range(4):
            rng = np.random.default_rng(seed)
            for c in chars:
                te = idx[c]
                if len(te) < 6:
                    continue
                pool = [x for x in chars if x != c]
                tr = np.concatenate([idx[x] for x in pool])
                mu = F[tr].mean(0)
                for a in set(ya[te]):
                    cand = [x for x in pool if (ya[idx[x]] == a).any()]
                    if len(cand) < k + 1:
                        continue
                    rest = te[ya[te] != a]
                    if len(rest) < 3:
                        continue
                    rig = F[rest].mean(0) - mu
                    tgt = te[ya[te] == a]
                    sets = {
                        "rand": list(rng.choice(cand, k, replace=False)),
                        "tight": pick_tight(cand, k, rng),
                        "spread": pick_spread(cand, k, rng),
                    }
                    for m, sel in sets.items():
                        act = np.mean([F[idx[x][ya[idx[x]] == a]].mean(0)
                                       for x in sel], axis=0) - mu
                        dd = [D[pos[u], pos[v]] for i, u in enumerate(sel)
                              for v in sel[i + 1:]]
                        internal[m].append(float(np.mean(dd)))
                        for i in tgt:
                            res[m][0].append(np.mean((F[i] - mu - rig) ** 2))
                            res[m][1].append(
                                np.mean((F[i] - mu - rig - LAM * act) ** 2))
        g = {m: 1 - np.mean(res[m][1]) / np.mean(res[m][0]) for m in res}
        print(f"  {k:>3d} {100*g['rand']:9.1f}% {100*g['tight']:14.1f}%"
              f" {100*g['spread']:11.1f}% {100*(g['spread']-g['tight']):+13.1f}pp")
        print(f"      internal distance: tight={np.mean(internal['tight']):.3f}"
              f"  random={np.mean(internal['rand']):.3f}"
              f"  spread={np.mean(internal['spread']):.3f}")


# ---------------------------------------- C. within-family vs cross-family
def family_transfer(F, ya, ym, yc):
    """Are donors from the SAME pack family better than from a different one?"""
    fam = np.array([family_of(m) for m in ym])
    chars = sorted(set(yc))
    idx = {c: np.where(yc == c)[0] for c in chars}

    print("\n=== C. does the pack family (= source proxy) carry transfer info ===")
    for cap in (2, 4, 8, None):
        out = {}
        for mode in ("same_fam", "diff_fam", "any"):
            er, eb, n = [], [], 0
            for seed in range(3):
                rng = np.random.default_rng(seed)
                for c in chars:
                    te = idx[c]
                    if len(te) < 6:
                        continue
                    tr = np.concatenate([idx[x] for x in chars if x != c])
                    mu = F[tr].mean(0)
                    myfam = set(fam[te])
                    for a in set(ya[te]):
                        rows = tr[ya[tr] == a]
                        if mode == "same_fam":
                            rows = rows[np.isin(fam[rows], list(myfam))]
                        elif mode == "diff_fam":
                            rows = rows[~np.isin(fam[rows], list(myfam))]
                        dchars = sorted(set(yc[rows]))
                        if len(dchars) < 2:
                            continue
                        if cap and len(dchars) > cap:
                            dchars = list(rng.choice(dchars, cap, replace=False))
                        act = np.mean([F[rows[yc[rows] == d]].mean(0)
                                       for d in dchars], axis=0) - mu
                        rest = te[ya[te] != a]
                        if len(rest) < 3:
                            continue
                        rig = F[rest].mean(0) - mu
                        for i in te[ya[te] == a]:
                            er.append(np.mean((F[i] - mu - rig) ** 2))
                            eb.append(np.mean((F[i] - mu - rig - LAM * act) ** 2))
                        n += 1
            out[mode] = (1 - np.mean(eb) / np.mean(er), n) if er else (0.0, 0)
        tag = f"cap={cap}" if cap else "cap=all"
        print(f"  {tag:8s}  same-family {100*out['same_fam'][0]:5.1f}%"
              f" (n={out['same_fam'][1]:5d})   "
              f"diff-family {100*out['diff_fam'][0]:5.1f}%"
              f" (n={out['diff_fam'][1]:5d})   "
              f"any {100*out['any'][0]:5.1f}%")


# --------------------------------------- D. hold out a family as new source
def holdout_family(F, ya, ym, yc):
    fam = np.array([family_of(m) for m in ym])
    fams = [f for f in sorted(set(fam)) if (fam == f).sum() >= 60]
    print("\n=== D. hold out an entire pack family = a stand-in 'new source' ===")
    print(f"  {'family':10s} {'assets':>7s} {'chars':>6s} {'新动作占比':>10s}"
          f" {'它能拿到的先验':>14s} {'它对别人的贡献':>14s}")
    for f in fams:
        te = np.where(fam == f)[0]
        tr = np.where(fam != f)[0]
        seen = set(ya[tr])
        novel = float(np.mean([a not in seen for a in ya[te]]))

        # how much prior can the held-out family draw from the rest?
        mu = F[tr].mean(0)
        er, eb = [], []
        for c in sorted(set(yc[te])):
            tec = te[yc[te] == c]
            trx = tr[yc[tr] != c]          # never use the same character
            if len(tec) < 5 or len(trx) < 100:
                continue
            mux = F[trx].mean(0)
            for a in set(ya[tec]):
                rows = trx[ya[trx] == a]
                if len(set(yc[rows])) < 2:
                    continue
                act = np.mean([F[rows[yc[rows] == d]].mean(0)
                               for d in sorted(set(yc[rows]))], axis=0) - mux
                rest = tec[ya[tec] != a]
                if len(rest) < 3:
                    continue
                rig = F[rest].mean(0) - mux
                for i in tec[ya[tec] == a]:
                    er.append(np.mean((F[i] - mux - rig) ** 2))
                    eb.append(np.mean((F[i] - mux - rig - LAM * act) ** 2))
        draw = 1 - np.mean(eb) / np.mean(er) if er else float("nan")

        # marginal contribution of this family to everyone else
        g_with = _global_gain(F, ya, ym, yc, None)
        g_wo = _global_gain(F, ya, ym, yc, f)
        print(f"  {f:10s} {len(te):7d} {len(set(yc[te])):6d} {100*novel:9.1f}%"
              f" {100*draw:13.1f}% {100*(g_with-g_wo):+13.2f}pp")


_GCACHE: dict = {}


def _global_gain(F, ya, ym, yc, drop_fam):
    key = drop_fam or "__all__"
    if key in _GCACHE:
        return _GCACHE[key]
    fam = np.array([family_of(m) for m in ym])
    ok = np.ones(len(ya), bool) if drop_fam is None else (fam != drop_fam)
    chars = sorted(set(yc))
    idx = {c: np.where(yc == c)[0] for c in chars}
    er, eb = [], []
    for c in chars:
        te = idx[c]
        if len(te) < 6:
            continue
        tr = np.concatenate([idx[x] for x in chars if x != c])
        tr = tr[ok[tr]]
        if len(tr) < 100:
            continue
        mu = F[tr].mean(0)
        for a in set(ya[te]):
            rows = tr[ya[tr] == a]
            if len(set(yc[rows])) < 2:
                continue
            act = np.mean([F[rows[yc[rows] == d]].mean(0)
                           for d in sorted(set(yc[rows]))], axis=0) - mu
            rest = te[ya[te] != a]
            if len(rest) < 3:
                continue
            rig = F[rest].mean(0) - mu
            for i in te[ya[te] == a]:
                er.append(np.mean((F[i] - mu - rig) ** 2))
                eb.append(np.mean((F[i] - mu - rig - LAM * act) ** 2))
    g = 1 - np.mean(eb) / np.mean(er) if er else 0.0
    _GCACHE[key] = g
    return g


def main() -> None:
    F, ya, ym, yc = load_feats()
    pac = defaultdict(set)
    for a, c in zip(ya, yc):
        pac[a].add(c)
    keep = np.array([len(pac[a]) >= 3 for a in ya])
    F, ya, ym, yc = F[keep], ya[keep], ym[keep], yc[keep]
    print(f"assets={len(F)}  packs={len(set(ym))}  chars={len(set(yc))}  "
          f"families={len(set(family_of(m) for m in ym))}\n")

    chars, beh, _ = measure_spread(F, ya, ym, yc)
    donor_set_diversity(F, ya, ym, yc, chars, beh)
    family_transfer(F, ya, ym, yc)
    holdout_family(F, ya, ym, yc)


if __name__ == "__main__":
    main()
