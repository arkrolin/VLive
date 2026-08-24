"""How many DISTINCT motion assets does the corpus actually contain?

Chain of evidence so far:
  * same-character packs are different models  (texture Jaccard 0.00,
    part Jaccard 0.18-0.71, independent moc3/physics3)   <- the user is right
  * but their motion trajectories match at median cosine 1.000, 86.9% >0.95
    <- the animation data is reused and retargeted, not re-authored
  * cross-character retrieval is flat at ~3-5% no matter how similar the rigs
    <- different characters are genuinely animated differently

So the unit of data is neither the directory (329) nor the character (39). It is
the distinct MOTION ASSET. This script counts those, and prints a concrete
side-by-side so the numbers can be checked by eye.

Usage:
    uv run python tools/audit_effective_dataset.py
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
from motion_curves import curve_extent  # noqa: E402
from survey_motion_names import norm_action  # noqa: E402
from verify_action_alignment import encode_motion  # noqa: E402


def show_pair(root: Path, pack_a: str, pack_b: str, action: str) -> None:
    """Print the raw curve inventory of the same action in two packs."""
    def pick(pack: str):
        d = root / pack / "motions"
        for f in sorted(d.glob("*.json")):
            if norm_action(f.stem)[0] == action:
                return f
        return None

    fa, fb = pick(pack_a), pick(pack_b)
    if not fa or not fb:
        print(f"  ('{action}' not in both packs)")
        return
    ja = orjson.loads(fa.read_bytes())
    jb = orjson.loads(fb.read_bytes())
    ca = {c["Id"]: c for c in ja.get("Curves", []) if c.get("Target") == "Parameter"}
    cb = {c["Id"]: c for c in jb.get("Curves", []) if c.get("Target") == "Parameter"}
    print(f"\n  --- '{action}' :  {fa.name}   vs   {fb.name}")
    print(f"      duration {ja['Meta']['Duration']:.2f}s / "
          f"{jb['Meta']['Duration']:.2f}s   "
          f"curves {len(ca)} / {len(cb)}   shared ids {len(set(ca)&set(cb))}")
    shared = sorted(set(ca) & set(cb))[:6]
    print(f"      {'parameter':26s} {'pack A (lo,hi)':>22s} {'pack B (lo,hi)':>22s}")
    for pid in shared:
        la, ha, _ = curve_extent(ca[pid]["Segments"])
        lb, hb, _ = curve_extent(cb[pid]["Segments"])
        same = "SAME" if abs(la - lb) < 1e-3 and abs(ha - hb) < 1e-3 else "diff"
        print(f"      {pid:26s} {f'({la:.3f}, {ha:.3f})':>22s} "
              f"{f'({lb:.3f}, {hb:.3f})':>22s}  {same}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--whitelist", default="outputs/model_whitelist.json")
    ap.add_argument("--thresh", type=float, default=0.95,
                    help="cosine above which two motions are the same asset")
    args = ap.parse_args()

    root = Path(args.root)
    models = sorted(orjson.loads(Path(args.whitelist).read_bytes())["kept"])

    print("=== concrete check: same character, two different packs ===")
    for act in ("dazhaohu", "haixiu", "shengqi"):
        show_pair(root, "l2d22.ugirl06", "l2d08.ugirl06", act)

    # ---- encode everything -------------------------------------------------
    print("\n\n=== counting distinct motion assets ===")
    X, y_act, y_model, y_char = [], [], [], []
    for name in models:
        mdir = root / name / "motions"
        if not mdir.is_dir():
            continue
        for f in sorted(mdir.glob("*.json")):
            act, _ = norm_action(f.stem)
            v = encode_motion(f)
            if v is None:
                continue
            X.append(v.reshape(-1)); y_act.append(act)
            y_model.append(name); y_char.append(char_of(name))
    X = np.stack(X).astype(np.float32)
    y_act = np.array(y_act); y_model = np.array(y_model); y_char = np.array(y_char)
    Z = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)

    # greedy clustering INSIDE each (character, action) group -- the only place
    # duplicates can plausibly live, and it keeps the problem O(small^2)
    n_files = len(X)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (c, a) in enumerate(zip(y_char, y_act)):
        groups[(c, a)].append(i)

    assets = 0
    dup_rows = 0
    per_char_assets: Counter = Counter()
    for (c, a), idx in groups.items():
        idx = np.array(idx)
        sub = Z[idx]
        unused = np.ones(len(idx), bool)
        while unused.any():
            k = int(np.argmax(unused))
            sims = sub @ sub[k]
            member = unused & (sims >= args.thresh)
            member[k] = True
            assets += 1
            per_char_assets[c] += 1
            dup_rows += int(member.sum()) - 1
            unused &= ~member
    print(f"  motion files (whitelist)          : {n_files}")
    print(f"  distinct assets (cos<{args.thresh} apart): {assets}")
    print(f"  collapsed as retargeted copies    : {dup_rows} "
          f"({100*dup_rows/n_files:.1f}%)")
    print(f"  redundancy factor                 : {n_files/max(assets,1):.2f}x")

    print(f"\n  distinct assets per character:")
    for c, n in per_char_assets.most_common(12):
        print(f"    {c:10s} {n:4d}")
    vals = np.array(sorted(per_char_assets.values(), reverse=True))
    print(f"  median={int(np.median(vals))}  "
          f"chars with >=50 assets: {int((vals>=50).sum())}/{len(vals)}")

    # what the action-label supervision is really worth
    pairs = 0
    for (c, a), idx in groups.items():
        pass
    by_act_char = defaultdict(set)
    for c, a in zip(y_char, y_act):
        by_act_char[a].add(c)
    multi = {a: cs for a, cs in by_act_char.items() if len(cs) >= 2}
    print(f"\n  actions shared by >=2 characters  : {len(multi)}")
    cross_pairs = sum(len(cs) * (len(cs) - 1) // 2 for cs in multi.values())
    print(f"  cross-CHARACTER action pairs      : {cross_pairs} "
          f"(the only non-redundant supervision)")


if __name__ == "__main__":
    main()
