"""Validate option B1: cross-character same-action Live2D exemplar as the
reference input for a target rig that has ZERO own clips.

The target's RIG term always comes from the static moc3 profile (zero clips,
exp I baseline). What we vary here is the ACTION reference:

  none     : static rig only, no action reference
  B1_k{K}  : mean of K other characters' clips of the SAME action (CANON space)
  B1_same3 : 3 exemplars drawn from the SAME production family as target
  B1_for3  : 3 exemplars drawn from FOREIGN families
  self     : target's OWN other clips of the action  (== option A upper bound)

We hold the target's clips of the action OUT of the reference and only use them
as the evaluation target (leave-one-character-out, and leave-one-clip-out for
'self'), so this honestly simulates "target rig has no reference at deploy time".
Action-R2 is measured against the static-rig-only baseline; total-R2 against the
global mean. This directly tests whether a cross-character exemplar beats the
pooled prior (already in the model) and how far it sits from owning the clips.

Run:  uv run python -u tools/audit_b1_reference.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audit_data_strategy import CACHE, LAM, NS, dedup   # noqa: E402
from audit_rig_source_split import load_raw, featurise   # noqa: E402

SEEDS = 4


def family_of(pack: str) -> str:
    return pack.split(".")[0]


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

    # (char, action) -> asset indices ; action -> set of chars
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

    pairs = [(c, a) for (c, a), v in ca_idx.items() if len(v) >= 3]
    print(f"assets={n} chars={len(set(yc))} candidate(target,action)={len(pairs)}\n")

    configs = {
        "none":     ("none", 0),
        "B1_k1":    ("rand", 1),
        "B1_k3":    ("rand", 3),
        "B1_k8":    ("rand", 8),
        "B1_same3": ("same", 3),
        "B1_for3":  ("foreign", 3),
        "self":     ("self", 0),
    }
    er_glob, er_rig, eb = (defaultdict(list) for _ in range(3))
    rngs = [np.random.default_rng(s) for s in range(SEEDS)]

    for (c, a) in pairs:
        te = ca_idx[(c, a)]
        others = [cc for cc in act_chars[a] if cc != c]
        if len(others) < 2:
            continue
        mu = F[np.where(yc != c)[0]].mean(0)
        rig = static_rig(c, mu)
        eg = float(np.mean([np.mean((F[i] - mu) ** 2) for i in te]))
        er = float(np.mean([np.mean((F[i] - mu - rig) ** 2) for i in te]))

        # self: leave-one-clip-out within target's own clips of a
        self_errs = []
        for i in te:
            act = F[[j for j in te if j != i]].mean(0)
            self_errs.append(np.mean((F[i] - mu - rig - LAM * act) ** 2))
        self_err = float(np.mean(self_errs))

        for tag, (mode, K) in configs.items():
            if mode == "none":
                ebv = er
            elif mode == "self":
                ebv = self_err
            else:
                errs = []
                for rng in rngs:
                    if mode == "rand":
                        sel = list(rng.choice(others, min(K, len(others)), replace=False))
                    elif mode == "same":
                        pool = [cc for cc in others if char_fam[cc] == char_fam[c]]
                        if not pool:
                            continue
                        sel = list(rng.choice(pool, min(K, len(pool)), replace=False))
                    else:  # foreign
                        pool = [cc for cc in others if char_fam[cc] != char_fam[c]]
                        if not pool:
                            continue
                        sel = list(rng.choice(pool, min(K, len(pool)), replace=False))
                    act = act_from_chars(sel, a)
                    if act is None:
                        continue
                    errs.append(float(np.mean(
                        [np.mean((F[i] - mu - rig - LAM * act) ** 2) for i in te])))
                if not errs:
                    continue
                ebv = float(np.mean(errs))
            er_glob[tag].append(eg)
            er_rig[tag].append(er)
            eb[tag].append(ebv)

    print(f"  {'config':10s} {'action-R2':>10s} {'total-R2':>10s} {'n':>6s}")
    for tag in configs:
        if tag not in eb:
            continue
        egm, erm, ebm = (np.mean(er_glob[tag]), np.mean(er_rig[tag]),
                         np.mean(eb[tag]))
        aR2 = 1 - ebm / erm
        tR2 = 1 - ebm / egm
        print(f"  {tag:10s} {aR2:+.3f}      {tR2:+.3f}     {len(eb[tag])}")
    print("\n  action-R2 = gain from adding the action reference atop static rig")
    print("  total-R2  = vs global mean (static moc3 alone should ~ +0.208)")


if __name__ == "__main__":
    main()
