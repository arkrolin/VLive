"""Are same-character packs the SAME rig, or genuinely different models?

audit_leakage.py treated every `*.ugirl06` dir as one character and blocked them
all from retrieving each other. Cross-character recall collapsed 81.6% -> 5.7%,
and I concluded the corpus was 94.8% re-skinned duplicates.

The user says that is wrong: l2d22.ugirl06 / l2d08.ugirl06 / l2d17.ugirl06 are
different outfits, different artwork, different motion sets -- the same
*character* in the story sense, but independently rigged and animated models.

If the user is right, blocking by character throws away real cross-rig
generalisation and the honest number is somewhere between 5.7% and 81.6%.

This script settles it with evidence instead of naming conventions:

  1. RIG IDENTITY   -- Jaccard over parameter ids and part ids between packs of
                       the same character, vs between different characters.
                       Same rig re-skinned => Jaccard ~1.0.
  2. ARTWORK        -- texture byte hashes / sizes. Same art => identical files.
  3. MOTION SETS    -- Jaccard over action names, and how many motions are
                       byte-identical curves across packs.
  4. WHY RETRIEVAL WORKED -- decompose the block-same-MODEL result: how many
                       top-1 hits land on the same character vs a different one,
                       and what recall looks like restricted to each.
  5. THE HONEST SPLIT -- recall for cross-PACK same-character retrieval after
                       removing verbatim and near-duplicate motions. If that
                       stays high, packs are legitimately different rigs.

Usage:
    uv run python tools/audit_pack_identity.py
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
from audit_leakage import char_of, curve_hash  # noqa: E402
from survey_motion_names import norm_action  # noqa: E402
from verify_action_alignment import encode_motion  # noqa: E402


# moc3 stores ids as NUL-padded ASCII; match whole identifier records only
_PARAM_RE = re.compile(rb"(?<![A-Za-z0-9_])([Pp][Aa][Rr][Aa][Mm][A-Za-z0-9_]{0,58})\x00")
_PART_RE = re.compile(
    rb"(?<![A-Za-z0-9_])((?:[Pp][Aa][Rr][Tt]|ART_|D_)[A-Za-z0-9_]{0,58})\x00")


def jac(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def read_model_ids(mdir: Path) -> tuple[set[str], set[str]]:
    """Parameter ids and part ids, straight out of the moc3 binary."""
    moc = next(iter(mdir.glob("*.moc3")), None)
    if moc is None:
        return set(), set()
    blob = moc.read_bytes()
    # A byte-by-byte Python scan over 285 multi-MB moc3 files was both slow and
    # memory-hostile (it died around the 100th model). The ids we want are
    # NUL-delimited ASCII identifiers, so let the regex engine do it in C.
    params = {m.decode("ascii") for m in _PARAM_RE.findall(blob)}
    parts = {m.decode("ascii") for m in _PART_RE.findall(blob)}
    return params, parts


def tex_sig(mdir: Path) -> set[str]:
    """Cheap texture fingerprint: size + head + tail, never the whole file.

    Reading every PNG of 285 models into memory is what OOM'd the first run.
    Size plus 64KB from each end is more than enough to tell 'same file' from
    'different artwork'.
    """
    sigs = set()
    for p in sorted((mdir / "textures").glob("*")):
        if not p.is_file():
            continue
        sz = p.stat().st_size
        h = hashlib.blake2b(str(sz).encode(), digest_size=8)
        with p.open("rb") as fh:
            h.update(fh.read(65536))
            if sz > 131072:
                fh.seek(-65536, 2)
                h.update(fh.read(65536))
        sigs.add(h.hexdigest())
    return sigs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--whitelist", default="outputs/model_whitelist.json")
    ap.add_argument("--focus", default="girl06")
    ap.add_argument("--max-n", type=int, default=4500,
                    help="cap on encoded motions; NxN float32 must fit in RAM")
    args = ap.parse_args()

    root = Path(args.root)
    wl = Path(args.whitelist)
    models = (sorted(orjson.loads(wl.read_bytes())["kept"]) if wl.exists()
              else sorted(d.name for d in root.iterdir() if d.is_dir()))
    by_char = defaultdict(list)
    for m in models:
        by_char[char_of(m)].append(m)

    # ---------------- 1+2+3. pack-vs-pack identity -------------------------
    print("=== 1) RIG IDENTITY: are same-character packs the same rig? ===")
    cache: dict[str, tuple[set, set, set, set]] = {}

    def sig(m: str):
        if m not in cache:
            d = root / m
            p, q = read_model_ids(d)
            acts = {norm_action(f.stem)[0] for f in (d / "motions").glob("*.json")} \
                if (d / "motions").is_dir() else set()
            cache[m] = (p, q, tex_sig(d), acts)
        return cache[m]

    # pick the packs with the most motions, otherwise a 1-motion pack makes the
    # action Jaccard look artificially disjoint
    def n_mot(m: str) -> int:
        d = root / m / "motions"
        return len(list(d.glob("*.json"))) if d.is_dir() else 0

    focus = sorted(by_char.get(args.focus, []), key=lambda m: -n_mot(m))[:8]
    if focus:
        print(f"  pairwise within {args.focus} "
              f"(top {len(focus)} packs by motion count):")
        print(f"  {'pair':28s} {'param':>7s} {'part':>7s} {'tex':>7s} {'action':>7s}")
        for i in range(len(focus)):
            for j in range(i + 1, len(focus)):
                a, b = focus[i], focus[j]
                pa, qa, ta, aa = sig(a)
                pb, qb, tb, ab = sig(b)
                tag = f"{a.split('.')[0]}~{b.split('.')[0]}"
                print(f"  {tag:28s} {jac(pa,pb):7.2f} {jac(qa,qb):7.2f} "
                      f"{jac(ta,tb):7.2f} {jac(aa,ab):7.2f}"
                      f"   |acts| {len(aa):3d}/{len(ab):3d} shared={len(aa&ab):3d}")

    # aggregate: same-character pairs vs different-character pairs
    print("\n  [scanning all packs for aggregate stats ...]", flush=True)
    rng = np.random.default_rng(0)
    same_p, same_t, same_a = [], [], []
    diff_p, diff_t, diff_a = [], [], []
    for c, ms in by_char.items():
        for i in range(len(ms)):
            for j in range(i + 1, min(len(ms), i + 4)):
                pa, qa, ta, aa = sig(ms[i])
                pb, qb, tb, ab = sig(ms[j])
                same_p.append(jac(pa, pb)); same_t.append(jac(ta, tb))
                same_a.append(jac(aa, ab))
    chars = sorted(by_char)
    for _ in range(200):
        c1, c2 = rng.choice(chars, 2, replace=False)
        m1 = by_char[c1][rng.integers(len(by_char[c1]))]
        m2 = by_char[c2][rng.integers(len(by_char[c2]))]
        pa, qa, ta, aa = sig(m1)
        pb, qb, tb, ab = sig(m2)
        diff_p.append(jac(pa, pb)); diff_t.append(jac(ta, tb))
        diff_a.append(jac(aa, ab))

    def stat(v):
        return f"mean={np.mean(v):.2f} median={np.median(v):.2f} max={np.max(v):.2f}"
    print(f"\n  SAME-character pack pairs (n={len(same_p)}):")
    print(f"    param-id Jaccard : {stat(same_p)}")
    print(f"    texture  Jaccard : {stat(same_t)}")
    print(f"    action   Jaccard : {stat(same_a)}")
    print(f"  DIFF-character pairs (n={len(diff_p)}):")
    print(f"    param-id Jaccard : {stat(diff_p)}")
    print(f"    texture  Jaccard : {stat(diff_t)}")
    print(f"    action   Jaccard : {stat(diff_a)}")

    # ---------------- encode motions ---------------------------------------
    X, y_act, y_model, y_char, y_hash = [], [], [], [], []
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
            y_hash.append(curve_hash(f))
    X = np.stack(X).astype(np.float32)
    y_act, y_model = np.array(y_act), np.array(y_model)
    y_char, y_hash = np.array(y_char), np.array(y_hash)
    per_act = Counter(y_act)
    keep = np.array([per_act[a] >= 6 for a in y_act])
    X, y_act, y_model, y_char, y_hash = (X[keep], y_act[keep], y_model[keep],
                                         y_char[keep], y_hash[keep])
    # a full float32 NxN needs 4N^2 bytes; several masked copies of it do not
    # fit, so cap N and reuse ONE similarity matrix everywhere below.
    if len(X) > args.max_n:
        sel = np.sort(rng.choice(len(X), args.max_n, replace=False))
        X, y_act, y_model, y_char, y_hash = (X[sel], y_act[sel], y_model[sel],
                                             y_char[sel], y_hash[sel])
    print(f"\n=== {len(X)} motions | {len(set(y_model))} packs | "
          f"{len(set(y_char))} characters ===")

    # ---------------- 3b. verbatim copies ACROSS packs ---------------------
    by_hash = defaultdict(set)
    for h, m in zip(y_hash, y_model):
        if h:
            by_hash[h].add(m)
    cross_pack = sum(1 for h, ms in by_hash.items() if len(ms) > 1)
    print(f"\n=== 3) verbatim identical curves ===")
    print(f"  distinct curve hashes           : {len(by_hash)}")
    print(f"  hashes appearing in >1 PACK     : {cross_pack} "
          f"({100*cross_pack/max(len(by_hash),1):.1f}%)")

    # ---------------- 4. decompose the block-same-MODEL result -------------
    Z = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    S0 = (Z @ Z.T).astype(np.float32)          # the ONE big matrix
    np.fill_diagonal(S0, -np.inf)
    same_pack = y_model[:, None] == y_model[None, :]
    same_ch = y_char[:, None] == y_char[None, :]

    S = np.where(same_pack, -np.inf, S0)
    top1 = np.argmax(S, axis=1)
    del S
    hit = y_act[top1] == y_act
    same_char_hit = y_char[top1] == y_char
    print(f"\n=== 4) where do the block-same-MODEL top-1 hits come from? ===")
    print(f"  overall r@1                     : {100*hit.mean():5.1f}%")
    print(f"  top-1 neighbour is SAME char    : {100*same_char_hit.mean():5.1f}% "
          f"of queries")
    print(f"    of those, correct action      : {100*hit[same_char_hit].mean():5.1f}%")
    print(f"    of the rest, correct action   : {100*hit[~same_char_hit].mean():5.1f}%")

    # ---------------- 5. cross-PACK same-character, dupes removed ----------
    print(f"\n=== 5) cross-PACK same-character recall, cleaned ===")
    seen: set[str] = set()
    uniq = []
    for h in y_hash:
        uniq.append(not (h and h in seen))
        if h:
            seen.add(h)
    uniq = np.array(uniq)

    def restricted(mask_rows, allow, tag):
        """Score rows in `mask_rows`, only allowed to retrieve where `allow`."""
        Sx = np.where(allow, S0, np.float32(-np.inf))
        valid = np.isfinite(Sx).any(1) & mask_rows
        if valid.sum() < 20:
            print(f"  {tag:46s} (too few queries)")
            del Sx
            return
        o = np.argsort(-Sx[valid], axis=1)[:, :5]
        del Sx
        t, l = y_act[valid], y_act[o]
        cnt = Counter(t); n = len(t)
        rand = sum(c*(c-1) for c in cnt.values()) / max(n*(n-1), 1)
        print(f"  {tag:46s} r@1={100*(l[:,0]==t).mean():5.1f}%  "
              f"r@5={100*(l==t[:,None]).any(1).mean():5.1f}%  "
              f"chance={100*rand:4.1f}%  (n={n})")

    diff_pack = ~same_pack
    all_rows = np.ones(len(X), bool)
    restricted(all_rows, diff_pack & same_ch,
               "A. diff PACK, SAME character (raw)")
    restricted(uniq, diff_pack & same_ch & uniq[None, :],
               "B. diff PACK, SAME character (dupes removed)")
    restricted(all_rows, ~same_ch,
               "C. DIFF character (raw)")
    restricted(uniq, (~same_ch) & uniq[None, :],
               "D. DIFF character (dupes removed)")

    # how similar are those cross-pack pairs, really?
    fin = np.isfinite(S0)
    cross_same = diff_pack & same_ch & fin
    if cross_same.any():
        vals = S0[cross_same]
        vals2 = S0[(~same_ch) & fin]
        print(f"\n  cosine, cross-pack SAME character: "
              f"mean={vals.mean():.3f}  p95={np.percentile(vals,95):.3f}  "
              f">0.98: {100*(vals>0.98).mean():.2f}%")
        print(f"  cosine, DIFFERENT character      : "
              f"mean={vals2.mean():.3f}  p95={np.percentile(vals2,95):.3f}  "
              f">0.98: {100*(vals2>0.98).mean():.2f}%")


if __name__ == "__main__":
    main()
