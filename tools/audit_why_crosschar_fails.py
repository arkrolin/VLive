"""Cross-pack same-character retrieval is 84%. Cross-character is 4.8%. Why?

audit_pack_identity.py established that same-character packs are genuinely
different models: texture Jaccard ~0.00, part Jaccard 0.18-0.71, and removing
verbatim duplicates barely dents the score (85.5% -> 84.1%). So the 84% is not
copy detection. Something real transfers between different rigs of one
character, and does not transfer between characters.

Two competing explanations, with opposite engineering consequences:

  RIG-CONVENTION  Same character keeps one parameter convention (param-id
                  Jaccard 0.73) while different characters share almost nothing
                  (0.15). The choreography might be perfectly comparable; we
                  just fail to line the channels up. => FIXABLE by rig
                  normalisation, and worth heavy investment.

  CHOREOGRAPHY    Each character is animated with her own movement identity.
                  'haixiu' for girl06 is a genuinely different performance from
                  'haixiu' for girl11. => NOT fixable by normalisation; a shared
                  text->motion prior can never be strong, and few-shot from the
                  target character is the only route.

Test: stratify cross-CHARACTER retrieval by how similar the two rigs are. If
recall climbs with rig similarity, it is RIG-CONVENTION. If it stays flat near
chance, it is CHOREOGRAPHY.

Also reports the cosine of correct vs incorrect matches, to confirm the 84% is
graded semantic similarity rather than near-identical copies.

Usage:
    uv run python tools/audit_why_crosschar_fails.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent))
from audit_leakage import char_of  # noqa: E402
from audit_pack_identity import jac, read_model_ids  # noqa: E402
from survey_motion_names import norm_action  # noqa: E402
from verify_action_alignment import CANON, encode_motion  # noqa: E402

CACHE = Path("outputs/_paramset_cache.json")


def load_param_sets(root: Path, models: list[str]) -> dict[str, set[str]]:
    """moc3 scanning costs ~2 min for the whole corpus; cache it."""
    cache: dict[str, list[str]] = {}
    if CACHE.exists():
        cache = orjson.loads(CACHE.read_bytes())
    missing = [m for m in models if m not in cache]
    if missing:
        print(f"  [scanning {len(missing)} moc3 files ...]", flush=True)
        for m in missing:
            p, _ = read_model_ids(root / m)
            cache[m] = sorted(p)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(orjson.dumps(cache))
    return {m: set(cache[m]) for m in models}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--whitelist", default="outputs/model_whitelist.json")
    ap.add_argument("--max-n", type=int, default=4500)
    args = ap.parse_args()

    root = Path(args.root)
    models = sorted(orjson.loads(Path(args.whitelist).read_bytes())["kept"])
    psets = load_param_sets(root, models)

    # ---- encode -----------------------------------------------------------
    X, y_act, y_model, y_char = [], [], [], []
    for name in models:
        mdir = root / name / "motions"
        if not mdir.is_dir():
            continue
        for f in sorted(mdir.glob("*.json")):
            act, _ = norm_action(f.stem)
            if act in ("stand", "idle"):
                continue
            v = encode_motion(f)
            if v is None:
                continue
            X.append(v.reshape(-1)); y_act.append(act)
            y_model.append(name); y_char.append(char_of(name))
    X = np.stack(X).astype(np.float32)
    y_act, y_model, y_char = np.array(y_act), np.array(y_model), np.array(y_char)
    per_act = Counter(y_act)
    keep = np.array([per_act[a] >= 6 for a in y_act])
    X, y_act, y_model, y_char = X[keep], y_act[keep], y_model[keep], y_char[keep]
    rng = np.random.default_rng(0)
    if len(X) > args.max_n:
        sel = np.sort(rng.choice(len(X), args.max_n, replace=False))
        X, y_act, y_model, y_char = X[sel], y_act[sel], y_model[sel], y_char[sel]

    Z = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    S = (Z @ Z.T).astype(np.float32)
    np.fill_diagonal(S, -np.inf)
    same_ch = y_char[:, None] == y_char[None, :]
    same_pack = y_model[:, None] == y_model[None, :]
    print(f"\n=== {len(X)} motions | {len(set(y_model))} packs | "
          f"{len(set(y_char))} chars ===")

    # ---- 1. is the 84% graded similarity, or copy detection? --------------
    Sa = np.where(same_ch & ~same_pack, S, np.float32(-np.inf))
    ok = np.isfinite(Sa).any(1)
    t1 = np.argmax(Sa[ok], axis=1)
    best = Sa[ok][np.arange(ok.sum()), t1]
    corr = y_act[t1] == y_act[ok]
    print("\n=== 1) cross-pack SAME character: copy or semantics? ===")
    print(f"  top-1 cosine when action CORRECT  : "
          f"mean={best[corr].mean():.3f}  median={np.median(best[corr]):.3f}  "
          f">0.95: {100*(best[corr]>0.95).mean():.1f}%")
    print(f"  top-1 cosine when action WRONG    : "
          f"mean={best[~corr].mean():.3f}  median={np.median(best[~corr]):.3f}")
    # NOTE: an earlier version of this script asserted here that the correct
    # matches would "sit well below 1.0" and therefore prove graded semantic
    # transfer. The measurement says the opposite -- median cosine is 1.000 and
    # ~87% exceed 0.95 -- so cross-pack same-character pairs are RETARGETED
    # COPIES of one motion asset, not independent re-animations. The curve hash
    # differs only because the keyframe encoding / parameter names differ.
    if np.median(best[corr]) > 0.9:
        print("  -> median ~1.0: these are the SAME motion asset retargeted to a")
        print("     new rig. Different artwork, reused animation. For a motion")
        print("     model these packs are ONE sample, not many.")

    # ---- 2. stratify cross-character retrieval by rig similarity ----------
    print("\n=== 2) cross-CHARACTER retrieval, stratified by rig similarity ===")
    packs = sorted(set(y_model))
    pidx = {p: i for i, p in enumerate(packs)}
    P = len(packs)
    J = np.zeros((P, P), dtype=np.float32)
    for i in range(P):
        for j in range(i + 1, P):
            v = jac(psets[packs[i]], psets[packs[j]])
            J[i, j] = J[j, i] = v
    mi = np.array([pidx[m] for m in y_model])
    Jm = J[mi[:, None], mi[None, :]]          # pairwise rig similarity

    print(f"  {'rig param-id Jaccard band':30s} {'r@1':>7s} {'r@5':>7s} "
          f"{'chance':>7s} {'n':>7s}")
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]:
        allow = (~same_ch) & (Jm >= lo) & (Jm < hi)
        Sx = np.where(allow, S, np.float32(-np.inf))
        valid = np.isfinite(Sx).any(1)
        if valid.sum() < 50:
            print(f"  [{lo:.1f},{hi:.1f})".ljust(32) + "  (too few)")
            continue
        o = np.argsort(-Sx[valid], axis=1)[:, :5]
        t, l = y_act[valid], y_act[o]
        c = Counter(t); n = len(t)
        ch = sum(k * (k - 1) for k in c.values()) / max(n * (n - 1), 1)
        print(f"  [{lo:.1f},{hi:.1f})".ljust(32) +
              f"{100*(l[:,0]==t).mean():6.1f}% {100*(l==t[:,None]).any(1).mean():6.1f}% "
              f"{100*ch:6.1f}% {n:7d}")

    # same stratification WITHIN a character, as the positive control
    print("\n  positive control, SAME character (different packs):")
    for lo, hi in [(0.0, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        allow = same_ch & (~same_pack) & (Jm >= lo) & (Jm < hi)
        Sx = np.where(allow, S, np.float32(-np.inf))
        valid = np.isfinite(Sx).any(1)
        if valid.sum() < 50:
            print(f"  [{lo:.1f},{hi:.1f})".ljust(32) + "  (too few)")
            continue
        o = np.argsort(-Sx[valid], axis=1)[:, :5]
        t, l = y_act[valid], y_act[o]
        print(f"  [{lo:.1f},{hi:.1f})".ljust(32) +
              f"{100*(l[:,0]==t).mean():6.1f}% "
              f"{100*(l==t[:,None]).any(1).mean():6.1f}% "
              f"{'':7s} {len(t):7d}")

    # ---- 3. channel coverage: is the encoder even seeing the same axes? ---
    print("\n=== 3) do both sides actually populate the canonical channels? ===")
    have = defaultdict(int)
    for m in models:
        ps = {p.upper().replace("_", "") for p in psets[m]}
        for ch_name, rx in CANON.items():
            if any(rx.match(p) or rx.match(p.replace("PARAM", "PARAM_"))
                   for p in psets[m]):
                have[ch_name] += 1
    for k in CANON:
        print(f"  {k:12s} present in {have[k]:3d}/{len(models)} packs "
              f"({100*have[k]/len(models):5.1f}%)")


if __name__ == "__main__":
    main()
