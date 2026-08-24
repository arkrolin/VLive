"""Validate option B1 automatic source-matching: can we RETRIEVE 3
cross-character same-action exemplars for a zero-reference target WITHOUT any
manual source label, purely from the static moc3 param-set Jaccard?

If the Jaccard-retrieved set recovers the same-source advantage measured in
V6.1 (B1_same3 +0.141 vs B1_for3 +0.090), then the "pick same-source examples"
step is fully automatable and needs no curated labelling.

We compare four donor-selection strategies for the SAME targets/action:
  rand3    : 3 random other-character exemplars (control)
  same3    : 3 from the target's OWN production family (manual label, V6.1)
  for3     : 3 from foreign families (manual label, V6.1)
  auto3    : 3 highest param-Jaccard to target's own moc3 (AUTOMATIC, no label)
  near     : all donors with Jaccard >= 0.30 (schema-near cluster)

We also report the family-match rate of auto3 (how often the top-Jaccard
donors coincide with the same production family) to show the two signals align.

Run:  uv run python -u tools/audit_b1_auto_retrieval.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent))
from audit_rig_source_split import load_raw, featurise, family_of  # noqa: E402
from audit_leakage import char_of  # noqa: E402
from audit_data_strategy import LAM  # noqa: E402

SEEDS = 4
ROOT = Path(__file__).resolve().parent.parent
PARAMSET = ROOT / "outputs" / "_paramset_cache.json"


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def main() -> None:
    amp, pres, ya, ym, yc = load_raw()
    fam = np.array([family_of(m) for m in ym])
    F = featurise(amp, pres)
    n = len(F)

    # static channel-existence profile per pack (exp I method)
    static = {p: pres[ym == p].max(0).astype(np.float32) for p in sorted(set(ym))}
    sr = np.stack([static[p] for p in ym])

    char_fam = {}
    for i in range(n):
        char_fam.setdefault(yc[i], fam[i])

    # per-char param set from the moc3 cache (static, zero clips needed)
    raw = orjson.loads(PARAMSET.read_bytes())
    char_params: dict[str, set] = defaultdict(set)
    for m, params in raw.items():
        char_params[char_of(m)].update(set(params))
    have = [c for c in set(yc) if c in char_params and char_params[c]]

    # (char, action) groups
    ca_idx: dict[tuple, list] = defaultdict(list)
    char_act: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    act_chars: dict[str, set] = defaultdict(set)
    for i in range(n):
        ca_idx[(yc[i], ya[i])].append(i)
        char_act[yc[i]][ya[i]].append(i)
        act_chars[ya[i]].add(yc[i])

    def static_rig(c: str, mu: np.ndarray) -> np.ndarray:
        tgt = sr[yc == c][0]
        glob = np.where(yc != c)[0]
        w = (sr[glob] @ tgt) / np.maximum(
            np.linalg.norm(sr[glob], axis=1) * np.linalg.norm(tgt), 1e-6)
        w = np.maximum(w - w.mean(), 0) ** 4
        if w.sum() == 0:
            return np.zeros(F.shape[1], np.float32)
        return np.average(F[glob], axis=0, weights=w) - mu

    def act_from_chars(chars, a):
        idxs = [j for cc in chars for j in char_act[cc][a]]
        return None if not idxs else F[idxs].mean(0)

    pairs = [(c, a) for (c, a), v in ca_idx.items() if len(v) >= 3 and c in have]
    print(f"assets={n} chars={len(set(yc))} candidate(target,action)={len(pairs)}\n")

    er_glob, er_rig, eb = (defaultdict(list) for _ in range(3))
    fam_match = []

    for (c, a) in pairs:
        te = ca_idx[(c, a)]
        others = [cc for cc in act_chars[a] if cc != c]
        if len(others) < 2:
            continue
        mu = F[np.where(yc != c)[0]].mean(0)
        rig = static_rig(c, mu)
        eg = float(np.mean([np.mean((F[i] - mu) ** 2) for i in te]))
        er = float(np.mean([np.mean((F[i] - mu - rig) ** 2) for i in te]))

        # automatic retrieval by param-set Jaccard to target's own moc3
        tgt_ps = char_params[c]
        scored = sorted(((jaccard(tgt_ps, char_params[cc]), cc) for cc in others),
                        reverse=True)
        auto3 = [cc for _, cc in scored[:3]]
        near = [cc for s, cc in scored if s >= 0.30][:6]
        if not near:
            near = auto3
        fam_match.append(np.mean([char_fam[cc] == char_fam[c] for cc in auto3]))

        strat = {
            "rand3": None,
            "same3": [cc for cc in others if char_fam[cc] == char_fam[c]],
            "for3":  [cc for cc in others if char_fam[cc] != char_fam[c]],
            "auto3": auto3,
            "near":  near,
        }
        for tag, pool in strat.items():
            if pool is None:  # rand3
                errs = []
                for rng in [np.random.default_rng(s) for s in range(SEEDS)]:
                    sel = list(rng.choice(others, min(3, len(others)), replace=False))
                    act = act_from_chars(sel, a)
                    if act is None:
                        continue
                    errs.append(float(np.mean(
                        [np.mean((F[i] - mu - rig - LAM * act) ** 2) for i in te])))
                if not errs:
                    continue
                ebv = float(np.mean(errs))
            else:
                if len(pool) < 1:
                    continue
                act = act_from_chars(pool, a)
                if act is None:
                    continue
                ebv = float(np.mean(
                    [np.mean((F[i] - mu - rig - LAM * act) ** 2) for i in te]))
            er_glob[tag].append(eg)
            er_rig[tag].append(er)
            eb[tag].append(ebv)

    print(f"  {'strategy':10s} {'action-R2':>10s} {'total-R2':>10s} {'n':>6s}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    order = ["rand3", "for3", "same3", "auto3", "near"]
    for tag in order:
        if tag not in eb:
            continue
        egm, erm, ebm = (np.mean(er_glob[tag]), np.mean(er_rig[tag]),
                         np.mean(eb[tag]))
        aR2 = 1 - ebm / erm
        tR2 = 1 - ebm / egm
        print(f"  {tag:10s} {aR2:+.3f}      {tR2:+.3f}     {len(eb[tag])}")
    print(f"\n  family-match rate of AUTO3 (top-Jaccard == same production family):"
          f" {100*np.mean(fam_match):.1f}%")
    print("  -> if auto3 action-R2 sits near same3 (+0.141) and far above for3")
    print("     (+0.090), source matching is fully automatic with zero labels.")


if __name__ == "__main__":
    main()
