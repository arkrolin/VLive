"""Estimate the gain of a live SWEEP PROBE over static moc3 for the rig term.

A real sweep probe measures, on the target rig at deploy time, the RIG GEOMETRY
with ZERO animation clips: which CANON channels exist, and how far each can move
(its mobility). The static moc3 channel-existence profile (exp I) is only the
existence half; the mobility half is what a probe adds.

We proxy the probe ceiling with the target's OWN clips: per-channel mobility =
std of the channel's value across that character's clips, min-max normalised to
[0,1] per channel. This is the exact information a perfect probe yields live (rig
geometry, not action content), so it is an HONEST CEILING for probe value.

We use the SAME kNN neighbour-weighting as exp I (proven to recover rig), only
switching the signature it operates on:
  exist : per-char channel-existence (20)          == exp I static moc3
  probe : [existence(20), mobility(20)] in [0,1]   == sweep-probe ceiling

Both use identical kNN machinery, so the delta is clean. We report (A) rig-term
recovery and (B) the full B1_k3 setting (action ref = 3 cross-char exemplars) to
see total R2 and how much of the self-upper-bound gap the probe closes.

Run:  uv run python -u tools/audit_sweep_probe_gain.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audit_rig_source_split import load_raw, featurise  # noqa: E402
from audit_data_strategy import LAM  # noqa: E402

SEEDS = 4


def build_sig(pres, amp, ym, yc):
    """Per-asset-row signatures (matches exp I's per-pack existence layout).

    sr_exist[i] : existence of row i's PACK  (exp I static moc3 signature)
    sr_probe[i] : [pack-existence(20), char-mobility(20)]  (sweep-probe ceiling)
    both in [0,1]; mobility is min-max normalised per channel.
    """
    packs = sorted(set(ym))
    static = {p: pres[ym == p].max(0).astype(np.float32) for p in packs}
    sr_exist = np.stack([static[p] for p in ym])
    chars = sorted(set(yc))
    mob_char = {}
    for c in chars:
        m = amp[yc == c].std(0)
        mob_char[c] = m
    M = np.stack([mob_char[c] for c in chars])
    M = np.log1p(M)
    lo, hi = M.min(0), M.max(0)
    M = (M - lo) / np.maximum(hi - lo, 1e-6)
    sr_mob = np.stack([M[chars.index(yc[i])] for i in range(len(yc))])
    sr_probe = np.hstack([sr_exist, sr_mob]).astype(np.float32)
    return sr_exist, sr_probe


def static_rig_sig(c, sr, F, yc):
    """exp I's kNN rig estimate, operating on an arbitrary per-row signature `sr`."""
    te = np.where(yc == c)[0]
    tgt = sr[te][0]
    glob = np.where(yc != c)[0]
    w = (sr[glob] @ tgt) / np.maximum(
        np.linalg.norm(sr[glob], axis=1) * np.linalg.norm(tgt), 1e-6)
    w = np.maximum(w - w.mean(), 0) ** 4
    if w.sum() == 0:
        return np.zeros(F.shape[1], np.float32)
    return np.average(F[glob], axis=0, weights=w) - F[glob].mean(0)


def main() -> None:
    amp, pres, ya, ym, yc = load_raw()
    F = featurise(amp, pres)
    n = len(F)
    chars = sorted(set(yc))
    sr_exist, sr_probe = build_sig(pres, amp, ym, yc)
    mu = F.mean(0)
    keep = [c for c in chars if (yc == c).sum() >= 5]

    print("=== A. rig-term recovery (leave-one-char-out kNN) ===")
    print(f"  {'rig source':46s} {'rig only':>9s} {'rig+prior':>11s} {'n':>6s}")

    def eval_rig(sig_mat):
        er_g, erg_g, eb_g = [], [], []
        for _ in range(SEEDS):
            for c in keep:
                te = np.where(yc == c)[0]
                glob = np.where(yc != c)[0]
                if len(glob) < 200:
                    continue
                rig = (np.zeros(F.shape[1], np.float32) if sig_mat is None
                       else static_rig_sig(c, sig_mat, F, yc))
                for a in set(ya[te]):
                    rows = glob[ya[glob] == a]
                    if len(set(yc[rows])) < 2:
                        continue
                    act = np.mean([F[rows[yc[rows] == d]].mean(0)
                                   for d in sorted(set(yc[rows]))], 0) - mu
                    for i in te[ya[te] == a]:
                        er_g.append(np.mean((F[i] - mu) ** 2))
                        erg_g.append(np.mean((F[i] - mu - rig) ** 2))
                        eb_g.append(np.mean((F[i] - mu - rig - LAM * act) ** 2))
        return (1 - np.mean(erg_g) / np.mean(er_g),
                1 - np.mean(eb_g) / np.mean(er_g), len(er_g))

    r_none = eval_rig(None)
    r_exist = eval_rig(sr_exist)
    r_probe = eval_rig(sr_probe)
    print(f"  {'none (rig=0)':46s} {r_none[0]:9.3f} {r_none[1]:11.3f} {r_none[2]:6d}")
    print(f"  {'static moc3 existence (exp I method)':46s} {r_exist[0]:9.3f} "
          f"{r_exist[1]:11.3f} {r_exist[2]:6d}")
    print(f"  {'sweep-probe sig (existence+mobility)':46s} {r_probe[0]:9.3f} "
          f"{r_probe[1]:11.3f} {r_probe[2]:6d}")
    print(f"\n  rig-term fraction recovered:  existence {100*r_exist[0]:.0f}%"
          f"  ->  probe {100*r_probe[0]:.0f}%"
          f"  (probe adds +{100*(r_probe[0]-r_exist[0]):.0f}pp)")
    print("  (exp I reported existence ~36%; this run's kNN reproduces it.)")

    # ---- (B) full B1_k3: action ref fixed, rig existence vs probe ----
    print("\n=== B. full B1_k3 (action ref = 3 cross-char exemplars) ===")
    ca_idx: dict[tuple, list] = defaultdict(list)
    char_act: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    act_chars: dict[str, set] = defaultdict(set)
    for i in range(n):
        ca_idx[(yc[i], ya[i])].append(i)
        char_act[yc[i]][ya[i]].append(i)
        act_chars[ya[i]].add(yc[i])
    pairs = [(c, a) for (c, a), v in ca_idx.items() if len(v) >= 3]
    print(f"  candidate(target,action)={len(pairs)}")

    er_g2, er_r2, eb2 = (defaultdict(list) for _ in range(3))
    rng = np.random.default_rng(0)
    map_exist = {c: static_rig_sig(c, sr_exist, F, yc) for c in keep}
    map_probe = {c: static_rig_sig(c, sr_probe, F, yc) for c in keep}
    for (c, a) in pairs:
        te = ca_idx[(c, a)]
        others = [cc for cc in act_chars[a] if cc != c]
        if len(others) < 3:
            continue
        eg = float(np.mean([np.mean((F[i] - mu) ** 2) for i in te]))
        sel = list(rng.choice(others, 3, replace=False))
        act = np.mean([F[char_act[cc][a]].mean(0) for cc in sel], 0)
        for mode in ("exist", "probe", "self"):
            if mode == "self":
                rig = F[te].mean(0) - mu
            else:
                rig = (map_exist[c] if mode == "exist" else map_probe[c])
            er = float(np.mean([np.mean((F[i] - mu - rig) ** 2) for i in te]))
            eb = float(np.mean(
                [np.mean((F[i] - mu - rig - LAM * act) ** 2) for i in te]))
            er_g2[mode].append(eg)
            er_r2[mode].append(er)
            eb2[mode].append(eb)
    print(f"  {'rig source':14s} {'total-R2':>9s} {'vs B1_k3':>9s} {'n':>6s}")
    self_total = 1 - np.mean(eb2["self"]) / np.mean(er_g2["self"])
    base_total = 1 - np.mean(eb2["exist"]) / np.mean(er_g2["exist"])
    probe_total = 1 - np.mean(eb2["probe"]) / np.mean(er_g2["probe"])
    print(f"  {'static(exist)':14s} {base_total:9.3f} {'':>9s} {len(eb2['exist']):6d}")
    print(f"  {'sweep-probe':14s} {probe_total:9.3f} "
          f"{probe_total-base_total:+.3f} {len(eb2['probe']):6d}")
    print(f"  {'self (upper)':14s} {self_total:9.3f} {'':>9s} {len(eb2['self']):6d}")
    gap = self_total - base_total
    closed = (probe_total - base_total) / gap * 100 if gap > 0 else 0
    print(f"\n  B1_k3 gap to self upper bound = {gap:.3f}")
    print(f"  sweep probe closes {closed:.0f}% of that gap "
          f"(B1_k3 {base_total:.3f} -> {probe_total:.3f} -> self {self_total:.3f})")


if __name__ == "__main__":
    main()
