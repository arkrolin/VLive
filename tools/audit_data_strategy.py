"""If more data helps, WHICH data? And is the bottleneck data at all?

audit_scaling_law.py established three facts:

  * WIDEN (more characters) is nearly flat once prior COVERAGE saturates.
    2 -> 28 training characters moves L2 R2 from 0.371 to 0.361, i.e. nothing.
    All apparent gain was coverage: at N=2 only 44.6% of held-out actions had
    any prior at all, at N=28 it is 100%.
  * Heaps exponent 0.82: extra characters bring mostly NEW action names.
    Support-per-action therefore grows only as N^0.18 -- 11x the characters
    buys 1.5x the support depth. That is why WIDEN saturates.
  * THICKEN looks strong: actions demonstrated by >=9 characters get a 25.0%
    prior gain versus 9.9% for actions seen in 2.

But THICKEN is confounded. Actions that many characters happen to share are
also the most stereotyped ones (idle, nod, wave), so the 25% could be a
property of those actions rather than a payoff from deeper support. Getting
this wrong inverts the recommendation, so it is tested directly here by fixing
the action and subsampling its support characters.

The second hypothesis is that the bottleneck is not data volume but LABEL
GRANULARITY. 3668 assets spread over 1189 distinct action names is 3.1 assets
per name. If many of those names are synonyms -- and V4 already showed the
signal lives in gesture morphemes (motouweixiao transfers at +0.96 while
weixiao transfers at +0.01) -- then collapsing the vocabulary thickens support
for free, with no new models collected at all.

The third is the user's own framing: the corpus comes from one kind of source.
If rigging-schema clusters proxy for studio/source, cross-cluster transfer
should be measurably worse than within-cluster, and diversifying sources would
then be counterproductive rather than helpful.

Usage:
    uv run python tools/audit_data_strategy.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent))
from audit_generative_prior import CACHE, NS, dedup  # noqa: E402
from audit_leakage import char_of  # noqa: E402

LAM = 0.5


def load_feats():
    z = np.load(CACHE, allow_pickle=True)
    amp, shape, pres = z["amp"], z["shape"], z["pres"]
    dur, ya, ym, yc = z["dur"], z["act"], z["pack"], z["char"]
    k = dedup(amp, shape, pres, dur, ya, ym, yc)
    amp, shape, pres, dur = amp[k], shape[k], pres[k], dur[k]
    ya, ym, yc = ya[k], ym[k], yc[k]
    scale = np.array([np.median(amp[pres[:, i], i]) if pres[:, i].any() else 1.0
                      for i in range(NS)], np.float32)
    f_act = (amp > 0.15 * np.maximum(scale, 1e-6)).astype(np.float32)
    f_amp = np.log1p(np.abs(amp) / (np.abs(amp).mean(0, keepdims=True) + 1e-6))
    F = np.hstack([f_act, f_amp])
    F -= F.mean(0)
    F /= np.maximum(F.std(0), 1e-6)
    return F, ya, ym, yc


def gain(F, ya, yc, eval_mask=None, sup_cap=None, rng=None):
    """Leave-one-character-out: error reduction of the action prior over rig.

    sup_cap limits how many SUPPORT CHARACTERS may contribute to act(a), which
    is what isolates depth-of-support from which-action-it-is.
    """
    chars = sorted(set(yc))
    idx_of = {c: np.where(yc == c)[0] for c in chars}
    e_rig, e_both, n = [], [], 0
    for c in chars:
        te = idx_of[c]
        if len(te) < 6:
            continue
        tr = np.concatenate([idx_of[x] for x in chars if x != c])
        mu = F[tr].mean(0)
        ya_tr, yc_tr = ya[tr], yc[tr]
        for a in set(ya[te]):
            m = tr[ya_tr == a]
            if len(m) < 2:
                continue
            sup_chars = sorted(set(yc_tr[ya_tr == a]))
            if sup_cap is not None:
                if len(sup_chars) < sup_cap:
                    continue
                pick = set(rng.choice(sup_chars, sup_cap, replace=False))
                m = m[[yc[i] in pick for i in m]]
                if len(m) < 1:
                    continue
            act = F[m].mean(0) - mu
            for i in te[ya[te] == a]:
                if eval_mask is not None and not eval_mask[i]:
                    continue
                rest = te[ya[te] != a]
                if len(rest) < 3:
                    continue
                rig = F[rest].mean(0) - mu
                t = F[i]
                e_rig.append(np.mean((t - mu - rig) ** 2))
                e_both.append(np.mean((t - mu - rig - LAM * act) ** 2))
                n += 1
    if n < 20:
        return None, n
    return 1 - np.mean(e_both) / np.mean(e_rig), n


# --------------------------------------------------------------------------
def disentangle_thicken(F, ya, yc):
    """Fix the action, vary only how many characters demonstrate it."""
    pac = defaultdict(set)
    for a, c in zip(ya, yc):
        pac[a].add(c)
    deep = {a for a, s in pac.items() if len(s) >= 12}
    mask = np.array([a in deep for a in ya])
    print(f"\n=== THICKEN, confound removed ===")
    print(f"  restricted to {len(deep)} actions that have >=12 support "
          f"characters, so every depth is reachable on the SAME actions")
    print(f"  {'support chars used':>20s} {'prior gain':>12s} {'n eval':>8s}")
    rng = np.random.default_rng(0)
    for cap in (1, 2, 4, 8, 11):
        vals = []
        for s in range(4):
            r = np.random.default_rng(31 * s + cap)
            g, n = gain(F, ya, yc, eval_mask=mask, sup_cap=cap, rng=r)
            if g is not None:
                vals.append(g)
        if vals:
            print(f"  {cap:>20d} {100*np.mean(vals):11.1f}% {n:8d}")
    g, n = gain(F, ya, yc, eval_mask=mask)
    print(f"  {'all available':>20s} {100*g:11.1f}% {n:8d}")


# --------------------------------------------------------------------------
MORPH_MIN_LEN = 4


def mine_morphemes(ya, yc, min_actions=3, min_chars=3):
    """Frequent substrings shared across distinct action names AND characters."""
    acts = sorted(set(ya))
    chars_of = defaultdict(set)
    for a, c in zip(ya, yc):
        chars_of[a].add(c)
    cnt_a, cnt_c = Counter(), defaultdict(set)
    for a in acts:
        seen = set()
        for L in range(MORPH_MIN_LEN, min(len(a), 12) + 1):
            for i in range(len(a) - L + 1):
                seen.add(a[i:i + L])
        for s in seen:
            cnt_a[s] += 1
            cnt_c[s] |= chars_of[a]
    cand = [s for s in cnt_a
            if cnt_a[s] >= min_actions and len(cnt_c[s]) >= min_chars]
    # prefer longer morphemes; drop a shorter one if a longer superstring
    # covers the same action set (it is just a fragment of it)
    cand.sort(key=lambda s: (-len(s), -cnt_a[s]))
    kept: list[str] = []
    for s in cand:
        if any(s in k and cnt_a[s] <= cnt_a[k] * 1.15 for k in kept):
            continue
        kept.append(s)
    return kept, cnt_a


def collapse_vocab(ya, morphs):
    """Map each action to its longest contained morpheme (else itself)."""
    order = sorted(morphs, key=len, reverse=True)
    out, hit = [], 0
    cache: dict[str, str] = {}
    for a in ya:
        if a not in cache:
            m = next((s for s in order if s in a), None)
            cache[a] = m if m else f"~{a}"
        lab = cache[a]
        hit += not lab.startswith("~")
        out.append(lab)
    return np.array(out), hit / len(ya)


def vocab_experiment(F, ya, yc):
    print("\n=== is the bottleneck DATA or LABEL GRANULARITY? ===")
    pac = defaultdict(set)
    for a, c in zip(ya, yc):
        pac[a].add(c)
    sup = np.array([len(pac[a]) for a in sorted(set(ya))])
    print(f"  raw vocabulary   : {len(set(ya)):5d} names, "
          f"median support {np.median(sup):.0f} chars, "
          f"{100*np.mean(sup >= 9):4.1f}% of names have >=9")

    morphs, cnt = mine_morphemes(ya, yc)
    yb, cov = collapse_vocab(ya, morphs)
    pbc = defaultdict(set)
    for a, c in zip(yb, yc):
        pbc[a].add(c)
    sup2 = np.array([len(pbc[a]) for a in sorted(set(yb))])
    real = [m for m in sorted(set(yb)) if not m.startswith("~")]
    print(f"  mined morphemes  : {len(morphs):5d}  "
          f"(top: {', '.join(sorted(morphs, key=lambda s:-cnt[s])[:8])})")
    print(f"  collapsed vocab  : {len(real):5d} morpheme classes "
          f"+ {len(set(yb)) - len(real)} singletons, "
          f"{100*cov:.1f}% of assets mapped")
    print(f"                     median support {np.median(sup2):.0f} chars, "
          f"{100*np.mean(sup2 >= 9):4.1f}% of classes have >=9")

    for tag, labels in (("raw action name", ya), ("morpheme class", yb)):
        p = defaultdict(set)
        for a, c in zip(labels, yc):
            p[a].add(c)
        m = np.array([len(p[a]) >= 3 for a in labels])
        g, n = gain(F, labels, yc, eval_mask=m)
        print(f"  prior gain with {tag:16s}: "
              f"{100*g:5.1f}%   (n={n}, {len(set(labels[m]))} classes)")


# --------------------------------------------------------------------------
def source_clusters(F, ya, yc, whitelist="outputs/model_whitelist.json",
                    cache="outputs/_paramset_cache.json"):
    """Does rigging-schema similarity (a proxy for studio) gate transfer?"""
    p = Path(cache)
    if not p.exists():
        print("\n=== source homogeneity: skipped (no paramset cache) ===")
        return
    raw = orjson.loads(p.read_bytes())
    by_char = defaultdict(set)
    for m, params in raw.items():
        by_char[char_of(m)] |= set(params)
    chars = [c for c in sorted(set(yc)) if c in by_char and by_char[c]]
    if len(chars) < 8:
        print("\n=== source homogeneity: skipped (too few chars) ===")
        return
    n = len(chars)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = by_char[chars[i]], by_char[chars[j]]
            J[i, j] = len(a & b) / max(len(a | b), 1)
    off = J[~np.eye(n, dtype=bool)]
    print(f"\n=== source homogeneity ({n} characters with a param schema) ===")
    print(f"  pairwise rig-schema Jaccard: median {np.median(off):.3f}  "
          f"p10 {np.percentile(off, 10):.3f}  p90 {np.percentile(off, 90):.3f}")

    pos = {c: i for i, c in enumerate(chars)}
    idx_of = {c: np.where(yc == c)[0] for c in set(yc)}
    lo, hi = np.percentile(off, 33), np.percentile(off, 67)
    buckets = {"schema-far": [[], []], "schema-mid": [[], []],
               "schema-near": [[], []]}
    for c in chars:
        te = idx_of[c]
        if len(te) < 6:
            continue
        for a in set(ya[te]):
            for other in chars:
                if other == c:
                    continue
                m = idx_of[other][ya[idx_of[other]] == a]
                if len(m) < 1:
                    continue
                tr = np.concatenate([idx_of[x] for x in chars if x != c])
                mu = F[tr].mean(0)
                act = F[m].mean(0) - mu
                rest = te[ya[te] != a]
                if len(rest) < 3:
                    continue
                rig = F[rest].mean(0) - mu
                j = J[pos[c], pos[other]]
                b = ("schema-far" if j < lo else
                     "schema-near" if j > hi else "schema-mid")
                for i in te[ya[te] == a]:
                    t = F[i]
                    buckets[b][0].append(np.mean((t - mu - rig) ** 2))
                    buckets[b][1].append(np.mean((t - mu - rig - LAM*act) ** 2))
    print(f"  {'donor schema similarity':>26s} {'prior gain':>12s} {'n':>8s}")
    for b in ("schema-far", "schema-mid", "schema-near"):
        er, eb = buckets[b]
        if len(er) < 30:
            continue
        print(f"  {b:>26s} {100*(1 - np.mean(eb)/np.mean(er)):11.1f}% "
              f"{len(er):8d}")


# --------------------------------------------------------------------------
def budget_simulation(F, ya, yc):
    """Same asset budget, three collection strategies. Which wins?"""
    chars = sorted(set(yc))
    idx_of = {c: np.where(yc == c)[0] for c in chars}
    big = [c for c in chars if len(idx_of[c]) >= 20]
    if len(big) < 10:
        print("\n=== budget simulation: skipped ===")
        return
    print(f"\n=== equal-budget collection strategies ===")
    print("  budget = 600 training assets, held-out characters fixed")
    print(f"  {'strategy':>34s} {'prior gain':>12s}")
    rng = np.random.default_rng(0)
    top_actions = [a for a, _ in Counter(ya).most_common(40)]

    def run(pick_fn, tag):
        vals = []
        for s in range(4):
            r = np.random.default_rng(7 * s)
            sel = pick_fn(r)
            if sel is None or len(sel) < 100:
                continue
            sub = np.zeros(len(ya), bool)
            sub[sel] = True
            g, n = gain(F[sub], ya[sub], yc[sub])
            if g is not None:
                vals.append(g)
        if vals:
            print(f"  {tag:>34s} {100*np.mean(vals):11.1f}%")

    def wide(r):
        out = []
        for c in chars:
            i = idx_of[c]
            out += list(r.choice(i, min(len(i), max(1, 600 // len(chars))),
                                 replace=False))
        return np.array(out)

    def deep(r):
        sel = list(r.choice(big, min(len(big), 12), replace=False))
        out = []
        per = 600 // len(sel)
        for c in sel:
            i = idx_of[c]
            out += list(r.choice(i, min(len(i), per), replace=False))
        return np.array(out)

    def focused(r):
        keep = np.where(np.isin(ya, top_actions))[0]
        return r.choice(keep, min(600, len(keep)), replace=False)

    run(wide, "WIDEN  thin slice of every character")
    run(deep, "DEEPEN 12 characters, many assets each")
    run(focused, "FOCUS  fixed 40-action list, all chars")


def main() -> None:
    F, ya, ym, yc = load_feats()
    print(f"=== {len(ya)} assets | {len(set(yc))} characters | "
          f"{len(set(ya))} action names ===")
    pac = defaultdict(set)
    for a, c in zip(ya, yc):
        pac[a].add(c)
    keep = np.array([len(pac[a]) >= 3 for a in ya])
    Fe, yae, yce = F[keep], ya[keep], yc[keep]

    vocab_experiment(F, ya, yc)
    disentangle_thicken(Fe, yae, yce)
    source_clusters(Fe, yae, yce)
    budget_simulation(Fe, yae, yce)


if __name__ == "__main__":
    main()
