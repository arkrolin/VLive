"""Would collecting more models actually help? Measure the scaling curve.

The user's observation: the corpus is high-quality but homogeneous (mobile-game
extractions from a narrow set of studios), only 3362 distinct assets across 36
characters, so maybe the weak cross-character prior is just a data-volume
problem. That is a testable claim, and it decides where the next month of
effort goes: crawling more models, or accepting the ceiling and redesigning.

The trap is that "more data" is three different things with three different
payoff curves, and they must not be conflated:

    WIDEN   more characters, same actions-per-character
            -> improves the shared prior act(a), the cross-rig term
    DEEPEN  more actions per character
            -> improves the few-shot rig term rig(c), which is ALREADY the
               dominant term (R2 0.33 vs the prior's +0.11)
    THICKEN more characters *per action*
            -> improves the specific action embeddings that already exist,
               versus spreading thin over a longer tail

Only WIDEN is what "collect more models" naively means, and it is the one most
likely to saturate: if the reason cross-character transfer fails is that two
artists genuinely choreograph "shy" differently, no amount of extra characters
fixes it -- the variance is irreducible, not sampling noise.

Method. Same additive factorisation as audit_generative_prior.py:

    f(char, action) ~= mu + rig(char) + lam * act(action)

held out one CHARACTER at a time. The difference here is that act(a) is
estimated from a *subsample* of N training characters. Sweeping N traces the
learning curve. Crucially the eval set is held FIXED across all N: when an
action has no support inside the subsample, the prediction falls back to
rig-only. That is exactly what happens with a smaller corpus, so the curve
measures prior quality AND prior coverage together, which is the quantity that
actually matters.

An irreducible-noise probe is included: the same-character oracle (predict a
held-out asset from OTHER assets of the SAME character and SAME action) is the
ceiling any amount of cross-character data could ever approach.

Usage:
    uv run python tools/audit_scaling_law.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audit_generative_prior import CACHE, NS, SLOTS, dedup  # noqa: E402

LAM = 0.5          # best shrinkage from the audit_generative_prior sweep
SEEDS = 6


def load():
    if not CACHE.exists():
        sys.exit("run tools/audit_generative_prior.py first to build the cache")
    z = np.load(CACHE, allow_pickle=True)
    amp, shape, pres = z["amp"], z["shape"], z["pres"]
    dur, ya, ym, yc = z["dur"], z["act"], z["pack"], z["char"]
    k = dedup(amp, shape, pres, dur, ya, ym, yc)
    return (amp[k], shape[k], pres[k], dur[k], ya[k], ym[k], yc[k])


def features(amp, shape, pres, dur):
    """L1 = does the channel actually move, L2 = amplitude signature."""
    scale = np.array([np.median(amp[pres[:, i], i]) if pres[:, i].any() else 1.0
                      for i in range(NS)], np.float32)
    f_act = (amp > 0.15 * np.maximum(scale, 1e-6)).astype(np.float32)
    f_amp = np.log1p(np.abs(amp) / (np.abs(amp).mean(0, keepdims=True) + 1e-6))
    return f_act, f_amp


# --------------------------------------------------------------------------
def curve_widen(F, ya, yc, ns, tag):
    """R2 as a function of the number of TRAINING CHARACTERS."""
    chars = sorted(set(yc))
    idx_of = {c: np.where(yc == c)[0] for c in chars}
    test = [c for c in chars if len(idx_of[c]) >= 6]
    rng = np.random.default_rng(0)

    print(f"\n  {tag}   ({len(test)} held-out characters, pool={len(chars) - 1})")
    print(f"    {'N train chars':>14s} {'R2':>8s} {'gain vs rig':>12s} "
          f"{'prior coverage':>15s}")

    base_rig = None
    out = []
    for n in ns:
        r2s, gains, covs = [], [], []
        for seed in range(SEEDS if n < max(ns) else 1):
            rs = np.random.default_rng(1000 * seed + n)
            se_mu, se_rig, se_both, cov_hit, cov_tot = [], [], [], 0, 0
            for c in test:
                pool = [x for x in chars if x != c]
                if n >= len(pool):
                    sub = pool
                elif n < 2:
                    continue
                else:
                    sub = list(rs.choice(pool, n, replace=False))
                tr = np.concatenate([idx_of[x] for x in sub])
                te = idx_of[c]
                if len(tr) < 30:
                    continue
                mu = F[tr].mean(0)
                act_tbl = {}
                ya_tr = ya[tr]
                for a in set(ya[te]):
                    m = tr[ya_tr == a]
                    if len(m) >= 2:
                        act_tbl[a] = F[m].mean(0) - mu
                for i in te:
                    sup = te[ya[te] != ya[i]]
                    if len(sup) < 3:
                        continue
                    rig = F[sup].mean(0) - mu
                    act = act_tbl.get(ya[i])
                    truth = F[i]
                    se_mu.append(np.mean((truth - mu) ** 2))
                    se_rig.append(np.mean((truth - mu - rig) ** 2))
                    p = mu + rig + (LAM * act if act is not None else 0.0)
                    se_both.append(np.mean((truth - p) ** 2))
                    cov_tot += 1
                    cov_hit += act is not None
            if not se_mu:
                continue
            emu, erig, eb = np.mean(se_mu), np.mean(se_rig), np.mean(se_both)
            r2s.append(1 - eb / emu)
            gains.append(1 - eb / erig)
            covs.append(cov_hit / max(cov_tot, 1))
        if not r2s:
            continue
        if base_rig is None:
            base_rig = 1 - erig / emu
        out.append((n, float(np.mean(r2s)), float(np.mean(gains))))
        print(f"    {n:>14d} {np.mean(r2s):8.3f} {100*np.mean(gains):11.1f}% "
              f"{100*np.mean(covs):14.1f}%")
    print(f"    {'(rig only)':>14s} {base_rig:8.3f} {0.0:11.1f}% {'-':>15s}")
    return out, base_rig


def extrapolate(out, base_rig, tag):
    """Fit R2(N) = ceil - a * N^(-b) and report the implied ceiling."""
    if len(out) < 4:
        return
    n = np.array([x[0] for x in out], float)
    r = np.array([x[1] for x in out], float)
    best = None
    for ceil in np.arange(r.max(), r.max() + 0.45, 0.005):
        d = ceil - r
        if (d <= 1e-6).any():
            continue
        A = np.vstack([np.ones_like(n), -np.log(n)]).T
        coef, res, *_ = np.linalg.lstsq(A, np.log(d), rcond=None)
        pred = A @ coef
        sse = float(np.sum((np.log(d) - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, ceil, float(np.exp(coef[0])), float(coef[1]))
    if best is None:
        return
    _, ceil, a, b = best
    cur = out[-1][1]
    print(f"\n    fitted  R2(N) = {ceil:.3f} - {a:.3f} * N^-{b:.2f}   [{tag}]")
    for n_future in (36, 72, 150, 400, 1000):
        v = ceil - a * n_future ** (-b)
        extra = 100 * (v - cur) / max(cur, 1e-6)
        print(f"      N={n_future:5d} chars -> R2 {v:.3f}   "
              f"({extra:+5.1f}% vs today)")
    print(f"      asymptotic ceiling  R2 {ceil:.3f}   "
          f"(today {cur:.3f}, rig-only {base_rig:.3f})")


# --------------------------------------------------------------------------
def curve_thicken(F, ya, yc, tag):
    """Does an action's prior improve when MORE characters demonstrate it?"""
    chars = sorted(set(yc))
    idx_of = {c: np.where(yc == c)[0] for c in chars}
    test = [c for c in chars if len(idx_of[c]) >= 6]
    bucket = defaultdict(lambda: [[], []])
    for c in test:
        tr = np.concatenate([idx_of[x] for x in chars if x != c])
        te = idx_of[c]
        mu = F[tr].mean(0)
        ya_tr = ya[tr]
        for a in set(ya[te]):
            m = tr[ya_tr == a]
            n_sup_chars = len(set(yc[m]))
            if n_sup_chars < 1:
                continue
            act = F[m].mean(0) - mu
            for i in te[ya[te] == a]:
                sup = te[ya[te] != a]
                if len(sup) < 3:
                    continue
                rig = F[sup].mean(0) - mu
                truth = F[i]
                e_rig = np.mean((truth - mu - rig) ** 2)
                e_b = np.mean((truth - mu - rig - LAM * act) ** 2)
                b = (1 if n_sup_chars == 1 else 2 if n_sup_chars == 2 else
                     3 if n_sup_chars <= 4 else 5 if n_sup_chars <= 8 else 9)
                bucket[b][0].append(e_rig)
                bucket[b][1].append(e_b)
    print(f"\n  {tag}: prior quality vs how many characters demonstrate it")
    print(f"    {'#chars w/ action':>18s} {'n eval':>8s} {'gain over rig-only':>20s}")
    for b in sorted(bucket):
        er, eb = np.mean(bucket[b][0]), np.mean(bucket[b][1])
        lbl = {1: "1", 2: "2", 3: "3-4", 5: "5-8", 9: ">=9"}[b]
        print(f"    {lbl:>18s} {len(bucket[b][0]):8d} "
              f"{100*(1 - eb/er):19.1f}%")


def curve_deepen(F, ya, yc, tag):
    """How many few-shot examples from the target rig do we actually need?"""
    chars = sorted(set(yc))
    idx_of = {c: np.where(yc == c)[0] for c in chars}
    test = [c for c in chars if len(idx_of[c]) >= 12]
    print(f"\n  {tag}: few-shot support size on the TARGET rig "
          f"({len(test)} chars with >=12 assets)")
    print(f"    {'k support':>10s} {'R2 rig only':>13s} {'R2 rig+prior':>14s}")
    for k in (1, 2, 4, 8, 16, 9999):
        se_mu, se_rig, se_both = [], [], []
        for seed in range(SEEDS):
            rs = np.random.default_rng(97 * seed + k)
            for c in test:
                te, tr = idx_of[c], np.concatenate(
                    [idx_of[x] for x in chars if x != c])
                mu = F[tr].mean(0)
                ya_tr = ya[tr]
                for i in te:
                    pool = te[ya[te] != ya[i]]
                    if len(pool) < 3:
                        continue
                    take = pool if k >= len(pool) else rs.choice(pool, k, False)
                    rig = F[take].mean(0) - mu
                    m = tr[ya_tr == ya[i]]
                    act = F[m].mean(0) - mu if len(m) >= 2 else 0.0
                    truth = F[i]
                    se_mu.append(np.mean((truth - mu) ** 2))
                    se_rig.append(np.mean((truth - mu - rig) ** 2))
                    se_both.append(np.mean((truth - mu - rig - LAM * act) ** 2))
        emu = np.mean(se_mu)
        lbl = "all" if k == 9999 else str(k)
        print(f"    {lbl:>10s} {1 - np.mean(se_rig)/emu:13.3f} "
              f"{1 - np.mean(se_both)/emu:14.3f}")


def oracle_ceiling(F, ya, yc, tag):
    """Same character + same action, held-out asset. The irreducible floor."""
    groups = defaultdict(list)
    for i, (c, a) in enumerate(zip(yc, ya)):
        groups[(c, a)].append(i)
    mu = F.mean(0)
    se_mu, se_o = [], []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        idx = np.array(idx)
        for i in idx:
            rest = idx[idx != i]
            se_mu.append(np.mean((F[i] - mu) ** 2))
            se_o.append(np.mean((F[i] - F[rest].mean(0)) ** 2))
    if se_mu:
        print(f"\n  {tag}: same-character same-action oracle "
              f"R2={1 - np.mean(se_o)/np.mean(se_mu):.3f}  (n={len(se_mu)}) "
              f"<- ceiling no cross-character data can pass")


def vocabulary_growth(ya, yc):
    """Heaps' law: do new characters bring new ACTIONS or repeat old ones?"""
    chars = sorted(set(yc))
    acts_of = {c: set(ya[yc == c]) for c in chars}
    rng = np.random.default_rng(0)
    curves = []
    for _ in range(40):
        order = list(rng.permutation(chars))
        seen, row = set(), []
        for c in order:
            seen |= acts_of[c]
            row.append(len(seen))
        curves.append(row)
    m = np.mean(curves, axis=0)
    n = np.arange(1, len(m) + 1)
    b, loga = np.polyfit(np.log(n), np.log(m), 1)
    a = np.exp(loga)
    print("\n=== action-vocabulary growth (Heaps' law) ===")
    print(f"  distinct actions after N characters: "
          f"{', '.join(f'N={k}:{m[k-1]:.0f}' for k in (1, 2, 4, 8, 16, len(m)))}")
    print(f"  fit V(N) = {a:.1f} * N^{b:.2f}")
    print(f"  marginal new actions per extra character now: "
          f"{a*b*len(m)**(b-1):.1f}  (at N=1 it was {a*b:.1f})")
    for k in (72, 150, 400):
        print(f"    N={k:4d} chars -> ~{a*k**b:5.0f} distinct actions "
              f"({a*k**b/m[-1]:.1f}x today)")
    print(f"  -> exponent {b:.2f}: "
          + ("new characters mostly bring NEW actions, the vocabulary keeps "
             "spreading thin" if b > 0.7 else
             "new characters mostly REPEAT known actions, so extra data "
             "thickens existing action classes rather than adding tail"))


def action_variance_decomposition(F, ya, yc):
    """How much of the variance is action vs rig vs irreducible within-cell?

    This is the honest ceiling argument. If the action factor explains only a
    few percent of the total variance while the within-(char,action) residual
    is large, the limit is artist-to-artist disagreement, and no volume of
    additional characters removes it.
    """
    mu = F.mean(0)
    tot = np.mean((F - mu) ** 2)
    cell = defaultdict(list)
    for i, (c, a) in enumerate(zip(yc, ya)):
        cell[(c, a)].append(i)
    within = []
    for idx in cell.values():
        if len(idx) < 2:
            continue
        g = F[np.array(idx)]
        within.append(np.mean((g - g.mean(0)) ** 2) * len(idx) / (len(idx) - 1))
    act_m = {a: F[ya == a].mean(0) for a in set(ya)}
    rig_m = {c: F[yc == c].mean(0) for c in set(yc)}
    v_act = np.mean([np.mean((act_m[a] - mu) ** 2) for a in ya])
    v_rig = np.mean([np.mean((rig_m[c] - mu) ** 2) for c in yc])
    print("\n=== variance decomposition (what is even learnable) ===")
    print(f"  total variance                              {tot:.4f}  100%")
    print(f"  explained by RIG identity                   {v_rig:.4f}  "
          f"{100*v_rig/tot:4.1f}%")
    print(f"  explained by ACTION label                   {v_act:.4f}  "
          f"{100*v_act/tot:4.1f}%")
    if within:
        w = float(np.mean(within))
        print(f"  irreducible within (same char, same action) {w:.4f}  "
              f"{100*w/tot:4.1f}%  <- noise floor")
        print(f"  -> even a PERFECT (rig x action) table would leave "
              f"{100*w/tot:.1f}% unexplained")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-chars", type=int, default=3)
    args = ap.parse_args()

    amp, shape, pres, dur, ya, ym, yc = load()
    print(f"=== {len(ya)} distinct assets | {len(set(ym))} packs | "
          f"{len(set(yc))} characters ===")
    per = Counter(yc)
    print(f"  assets/character: median {np.median(list(per.values())):.0f}  "
          f"max {max(per.values())}  min {min(per.values())}")

    vocabulary_growth(ya, yc)

    pac = defaultdict(set)
    for a, c in zip(ya, yc):
        pac[a].add(c)
    keep = np.array([len(pac[a]) >= args.min_chars for a in ya])
    amp, shape, pres, dur = amp[keep], shape[keep], pres[keep], dur[keep]
    ya, ym, yc = ya[keep], ym[keep], yc[keep]
    print(f"\n=== eval subset (action in >={args.min_chars} chars): "
          f"{len(ya)} assets, {len(set(ya))} actions, {len(set(yc))} chars ===")

    f_act, f_amp = features(amp, shape, pres, dur)
    for F in (f_act, f_amp):
        F -= F.mean(0)
        F /= np.maximum(F.std(0), 1e-6)

    action_variance_decomposition(f_amp, ya, yc)

    ns = [2, 4, 6, 8, 12, 16, 20, 24, 28, len(set(yc)) - 1]
    ns = sorted({n for n in ns if n >= 2})

    print("\n=== WIDEN: learning curve over number of training characters ===")
    o1, b1 = curve_widen(f_act, ya, yc, ns, "L1 channel activation")
    extrapolate(o1, b1, "L1")
    o2, b2 = curve_widen(f_amp, ya, yc, ns, "L2 amplitude signature")
    extrapolate(o2, b2, "L2")

    print("\n=== THICKEN: characters per action ===")
    curve_thicken(f_act, ya, yc, "L1")
    curve_thicken(f_amp, ya, yc, "L2")

    print("\n=== DEEPEN: few-shot support on the target rig ===")
    curve_deepen(f_act, ya, yc, "L1")
    curve_deepen(f_amp, ya, yc, "L2")

    oracle_ceiling(f_act, ya, yc, "L1")
    oracle_ceiling(f_amp, ya, yc, "L2")


if __name__ == "__main__":
    main()
