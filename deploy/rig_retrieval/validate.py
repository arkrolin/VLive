"""Equivalence regression: the productised retriever must reproduce V6.2.

V6.2 measured the "auto3" donor strategy (top-3 rig-schema-Jaccard matches,
chosen with zero source labels) at:

    action-R2 = +0.144   total-R2 = +0.201   family-match = 75.8%

This script drives the *shipped* ``Retriever`` to make those same picks, then
recomputes the R2 with the canonical feature cache. If the numbers come back in
range, the product is behaviourally identical to the experiment that justified
shipping it. Run via ``cli.py validate`` or directly.

Requires the research feature cache (outputs/_genprior_cache.npz) to exist; it is
produced by tools/audit_generative_prior.py and reused across the V4-V6 series.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from audit_rig_source_split import load_raw, featurise, family_of  # noqa: E402
from audit_data_strategy import LAM                                  # noqa: E402

from .index import CorpusIndex  # noqa: E402
from .retriever import Retriever  # noqa: E402

INDEX_PATH = ROOT / "outputs" / "retrieval_index.jsonl"


def _static_rig(c: str, mu: np.ndarray, sr: np.ndarray, F: np.ndarray, yc) -> np.ndarray:
    tgt = sr[yc == c][0]
    glob = np.where(yc != c)[0]
    w = (sr[glob] @ tgt) / np.maximum(
        np.linalg.norm(sr[glob], axis=1) * np.linalg.norm(tgt), 1e-6)
    w = np.maximum(w - w.mean(), 0) ** 4
    if w.sum() == 0:
        return np.zeros(F.shape[1], np.float32)
    return np.average(F[glob], axis=0, weights=w) - mu


def run(index_path: Path = INDEX_PATH) -> dict:
    amp, pres, ya, ym, yc = load_raw()
    fam = np.array([family_of(m) for m in ym])
    F = featurise(amp, pres)

    static = {p: pres[ym == p].max(0).astype(np.float32) for p in set(ym)}
    sr = np.stack([static[p] for p in ym])

    index = CorpusIndex.load(index_path)
    ret = Retriever(index)
    char_fam = index.char_family

    ca: dict[tuple, list] = defaultdict(list)
    act_chars: dict[str, set] = defaultdict(set)
    for i in range(len(ya)):
        ca[(yc[i], ya[i])].append(i)
        act_chars[ya[i]].add(yc[i])

    have = [c for c in set(yc) if c in index.char_params and index.char_params[c]]
    pairs = [(c, a) for (c, a), v in ca.items() if len(v) >= 3 and c in have]

    er_glob, er_rig, eb, fam_match = [], [], [], []
    for (c, a) in pairs:
        te = ca[(c, a)]
        others = [cc for cc in act_chars[a] if cc != c]
        if len(others) < 2:
            continue
        res = ret.retrieve(
            index.char_params[c], a, k=3,
            exclude_chars=[c], target_family=char_fam.get(c),
        )
        sel = [r.char for r in res.references]
        if not sel:
            continue
        mu = F[np.where(yc != c)[0]].mean(0)
        rig = _static_rig(c, mu, sr, F, yc)
        idxs = [j for cc in sel for j in ca[(cc, a)]]
        if not idxs:
            continue
        act = F[idxs].mean(0) - mu
        eg = float(np.mean([np.mean((F[i] - mu) ** 2) for i in te]))
        er = float(np.mean([np.mean((F[i] - mu - rig) ** 2) for i in te]))
        ebv = float(np.mean([np.mean((F[i] - mu - rig - LAM * act) ** 2)
                            for i in te]))
        er_glob.append(eg); er_rig.append(er); eb.append(ebv)
        if char_fam.get(c) is not None:
            fam_match.append(np.mean([char_fam.get(s) == char_fam.get(c)
                                      for s in sel]))

    aR2 = 1 - np.mean(eb) / np.mean(er_rig)
    tR2 = 1 - np.mean(eb) / np.mean(er_glob)
    fm = float(np.mean(fam_match)) if fam_match else 0.0

    print("=== retrieval equivalence vs V6.2 (auto3) ===")
    print(f"  candidate (target,action) pairs : {len(pairs)}")
    print(f"  {'metric':18s} {'this module':>12s} {'V6.2 target':>14s} {'status':>8s}")
    print(f"  {'action-R2':18s} {aR2:+12.3f} {'+0.144':>14s} "
          f"{_ok(aR2, 0.144, 0.04)}")
    print(f"  {'total-R2':18s} {tR2:+12.3f} {'+0.201':>14s} "
          f"{_ok(tR2, 0.201, 0.05)}")
    print(f"  {'family-match':18s} {100*fm:11.1f}% {'75.8%':>14s} "
          f"{_ok(fm, 0.758, 0.10)}")

    return {"action_R2": aR2, "total_R2": tR2, "family_match": fm,
            "pairs": len(pairs)}


def _ok(value: float, target: float, tol: float) -> str:
    return "OK" if abs(value - target) <= tol else "CHECK"


if __name__ == "__main__":
    run()
