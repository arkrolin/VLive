"""Leakage audit for the 85.8% cross-model action retrieval result.

Two ways that number could be fake:

  (1) CHARACTER LEAKAGE. Model dirs look like l2d22.ugirl06 / l2d08.ugirl06 /
      l2d17.ugirl06 -- the same character shipped in several packs. Retrieving
      'dazhaohu' from another *version of the same character* is not cross-model
      generalisation, it is near-duplicate matching.

  (2) CURVE COPY. Studios reuse a motion asset verbatim across characters, only
      swapping the rig. Then the trajectories are literally identical.

This script re-runs retrieval under progressively harsher splits and reports how
much of the score survives. Anything that survives a cross-CHARACTER split with
exact duplicates removed is real signal.

Usage:
    uv run python tools/audit_leakage.py
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent))
from survey_motion_names import norm_action  # noqa: E402
from verify_action_alignment import CANON, MIN_PER_ACTION, T_NORM, encode_motion  # noqa: E402

# l2d22.ugirl06 -> girl06 ; l2d17.uboy03 -> boy03
CHAR_RE = re.compile(r"[._]u?((?:girl|boy|char|npc)\d+)", re.I)


def char_of(model: str) -> str:
    m = CHAR_RE.search(model)
    return m.group(1).lower() if m else model.lower()


def curve_hash(path: Path) -> str:
    """Hash the raw Segments payload: identical assets hash identically."""
    try:
        j = orjson.loads(path.read_bytes())
    except Exception:
        return ""
    parts = []
    for c in sorted((j.get("Curves") or []), key=lambda c: str(c.get("Id"))):
        parts.append(str(c.get("Id")))
        parts.append(",".join(f"{float(x):.4f}" for x in (c.get("Segments") or [])))
    return hashlib.blake2b("|".join(parts).encode(), digest_size=16).hexdigest()


def retrieval(X, y_act, block, tag: str) -> tuple[float, float, int]:
    """recall@1 / @5 where `block[i]==block[j]` pairs are forbidden."""
    Z = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    S = Z @ Z.T
    same_block = block[:, None] == block[None, :]
    S = np.where(same_block, -np.inf, S)
    np.fill_diagonal(S, -np.inf)
    valid = np.isfinite(S).any(axis=1)
    if valid.sum() == 0:
        return 0.0, 0.0, 0
    order = np.argsort(-S[valid], axis=1)[:, :5]
    tgt = y_act[valid]
    lbl = y_act[order]
    r1 = float((lbl[:, 0] == tgt).mean())
    r5 = float((lbl == tgt[:, None]).any(1).mean())
    print(f"  {tag:38s} r@1={100*r1:5.1f}%  r@5={100*r5:5.1f}%  "
          f"(queries={int(valid.sum())})")
    return r1, r5, int(valid.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--whitelist", default="outputs/model_whitelist.json")
    ap.add_argument("--n-models", type=int, default=140)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    wl = Path(args.whitelist)
    models = (sorted(orjson.loads(wl.read_bytes())["kept"]) if wl.exists()
              else sorted(d.name for d in root.iterdir() if d.is_dir()))
    rng = np.random.default_rng(args.seed)
    if len(models) > args.n_models:
        models = sorted(rng.choice(models, args.n_models, replace=False))

    # ---- how bad is the character overlap in the first place? -------------
    all_models = sorted(d.name for d in root.iterdir() if d.is_dir())
    chars = Counter(char_of(m) for m in all_models)
    multi = {c: n for c, n in chars.items() if n > 1}
    print(f"=== character overlap across the full corpus ===")
    print(f"  {len(all_models)} model dirs -> {len(chars)} distinct characters")
    print(f"  characters appearing in >1 pack: {len(multi)} "
          f"covering {sum(multi.values())} dirs "
          f"({100*sum(multi.values())/len(all_models):.1f}% of corpus)")
    print("  worst offenders: " + ", ".join(
        f"{c}x{n}" for c, n in sorted(multi.items(), key=lambda kv: -kv[1])[:8]))

    # ---- encode ------------------------------------------------------------
    X, y_act, y_model, y_char, y_hash = [], [], [], [], []
    for name in models:
        mdir = root / name / "motions"
        if not mdir.is_dir():
            continue
        for f in sorted({p.name: p for p in mdir.glob("*.json")}.values()):
            act, _ = norm_action(f.stem)
            if act in ("stand", "idle"):
                continue
            v = encode_motion(f)
            if v is None:
                continue
            X.append(v.reshape(-1))
            y_act.append(act)
            y_model.append(name)
            y_char.append(char_of(name))
            y_hash.append(curve_hash(f))

    X = np.stack(X)
    y_act = np.array(y_act)
    y_model = np.array(y_model)
    y_char = np.array(y_char)
    y_hash = np.array(y_hash)

    per_action_char = defaultdict(set)
    for a, c in zip(y_act, y_char):
        per_action_char[a].add(c)
    good = {a for a, cs in per_action_char.items() if len(cs) >= MIN_PER_ACTION}
    keep = np.array([a in good for a in y_act])
    X, y_act, y_model, y_char, y_hash = (X[keep], y_act[keep], y_model[keep],
                                         y_char[keep], y_hash[keep])
    print(f"\n=== {len(X)} motions | {len(set(y_model))} models | "
          f"{len(set(y_char))} characters | {len(good)} actions ===")

    # ---- (2) verbatim duplicate assets ------------------------------------
    hc = Counter(y_hash)
    dup_rows = sum(c for h, c in hc.items() if c > 1 and h)
    cross_char_dup = 0
    by_hash = defaultdict(set)
    for h, c in zip(y_hash, y_char):
        if h:
            by_hash[h].add(c)
    cross_char_dup = sum(1 for h, cs in by_hash.items() if len(cs) > 1)
    print(f"\n=== verbatim curve duplicates ===")
    print(f"  motions sharing an identical curve hash: {dup_rows}/{len(X)} "
          f"({100*dup_rows/len(X):.1f}%)")
    print(f"  hashes spanning >1 character            : {cross_char_dup}")

    # keep one representative per (hash) to kill exact copies
    seen: set[str] = set()
    uniq_mask = []
    for h in y_hash:
        if h and h in seen:
            uniq_mask.append(False)
        else:
            seen.add(h)
            uniq_mask.append(True)
    uniq_mask = np.array(uniq_mask)

    # ---- retrieval under progressively harsher splits ---------------------
    print(f"\n=== retrieval under progressively harsher splits ===")
    retrieval(X, y_act, y_model, "A. block same MODEL (original claim)")
    retrieval(X, y_act, y_char, "B. block same CHARACTER")
    Xu, yu, cu = X[uniq_mask], y_act[uniq_mask], y_char[uniq_mask]
    retrieval(Xu, yu, cu, "C. block CHARACTER + drop exact dupes")

    # near-duplicate removal: cosine > 0.98 between different characters
    Z = Xu / np.maximum(np.linalg.norm(Xu, axis=1, keepdims=True), 1e-6)
    S = Z @ Z.T
    np.fill_diagonal(S, 0.0)
    diff_char = cu[:, None] != cu[None, :]
    near = ((S > 0.98) & diff_char).any(1)
    print(f"\n  near-duplicate (cos>0.98 across characters): "
          f"{int(near.sum())}/{len(Xu)} ({100*near.mean():.1f}%)")
    if (~near).sum() > 50:
        retrieval(Xu[~near], yu[~near], cu[~near],
                  "D. + drop near-dupes (cos>0.98)")

    counts = Counter(yu)
    n = len(yu)
    rand = sum(c * (c - 1) for c in counts.values()) / max(n * (n - 1), 1)
    print(f"\n  random baseline on the hardest split: {100*rand:.2f}%")


if __name__ == "__main__":
    main()
