"""Does cross-character transfer improve once ARMS are in the encoder?

Every cross-character number so far (4.8% r@1) was measured with the 14-channel
CANON encoder, which is head / body-angle / eyes / brows / mouth / gaze. It
contains ZERO arm channels -- and arms are exactly what the project must
generate. Two readings of that:

  ARTIFACT   The face barely differs between a shy pose and a greeting; the
             discriminative motion lives in the arms. Add arms and the shared
             prior appears. => a cross-character prior is worth training.

  REAL       Characters genuinely perform the same action differently. Arms
             will not rescue it. => few-shot from the target rig is the only
             route, and the corpus is a rig-conditioning dataset, not a
             text->motion pretraining set.

Arm ids are messy (387 distinct names, only 2 cover >=50% of characters) but I
showed they normalise to ~18 slots via side/role/perspective/index. This script
adds the top arm slots to the encoder and re-runs the decisive comparisons.

Usage:
    uv run python tools/audit_arm_channels.py
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent))
from audit_leakage import char_of  # noqa: E402
from motion_curves import sample_curve  # noqa: E402
from survey_motion_names import norm_action  # noqa: E402
from verify_action_alignment import CANON, T_NORM  # noqa: E402

ARM_HINT = re.compile(r"ARM|HAND|SHOU|ELBOW|WRIST|FINGER", re.I)
# the slots that survived the coverage check, best first
ARM_SLOTS = ["arm_l_1", "arm_r_1", "hand_r_1", "hand_l_1", "arm_l_2", "arm_r_2"]


def canon_arm(pid: str) -> str | None:
    """387 raw arm names -> a handful of side/role/index slots."""
    s = pid.upper().replace("PARAM_", "").replace("PARAM", "")
    if re.search(r"(^|_)(R|RIGHT|YOU)(_|$)", s) or re.search(r"HANDR|ARMR", s):
        side = "r"
    elif re.search(r"(^|_)(L|LEFT|ZUO)(_|$)", s) or re.search(r"HANDL|ARML", s):
        side = "l"
    else:
        return None
    if "TOUSHI" in s or "LEVEL" in s:      # perspective / level helpers: skip
        return None
    role = "hand" if re.search(r"HAND|WRIST|FINGER|SHOU$", s) else "arm"
    m = re.search(r"_(\d\d?)(?!.*\d)", s)
    idx = (m.group(1).lstrip("0") or "1") if m else "1"
    return f"{role}_{side}_{idx}"


def encode(path: Path, use_arms: bool) -> np.ndarray | None:
    try:
        j = orjson.loads(path.read_bytes())
    except Exception:
        return None
    dur = float(j.get("Meta", {}).get("Duration", 0) or 0)
    if dur <= 0:
        return None
    t = np.linspace(0.0, dur, T_NORM)
    slots = list(CANON) + (ARM_SLOTS if use_arms else [])
    out = np.zeros((len(slots), T_NORM), dtype=np.float32)
    pos = {s: i for i, s in enumerate(slots)}
    filled = set()
    for c in j.get("Curves", []):
        if c.get("Target") != "Parameter":
            continue
        pid = str(c.get("Id", ""))
        slot = None
        for name, rx in CANON.items():
            if rx.match(pid):
                slot = name
                break
        if slot is None and use_arms and ARM_HINT.search(pid):
            k = canon_arm(pid)
            if k in pos:
                slot = k
        if slot is None or slot in filled:
            continue
        try:
            v = sample_curve(c.get("Segments") or [], t)
        except Exception:
            continue
        sd = float(np.std(v))
        out[pos[slot]] = (v - float(np.mean(v))) / sd if sd > 1e-6 else 0.0
        filled.add(slot)
    if len(filled) < 3:
        return None
    return out


def run(root: Path, models: list[str], use_arms: bool, max_n: int, seed: int):
    X, ya, ym, yc = [], [], [], []
    for name in models:
        d = root / name / "motions"
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            a, _ = norm_action(f.stem)
            if a in ("stand", "idle"):
                continue
            v = encode(f, use_arms)
            if v is None:
                continue
            X.append(v.reshape(-1)); ya.append(a)
            ym.append(name); yc.append(char_of(name))
    X = np.stack(X).astype(np.float32)
    ya, ym, yc = np.array(ya), np.array(ym), np.array(yc)
    per = Counter(ya)
    keep = np.array([per[a] >= 6 for a in ya])
    X, ya, ym, yc = X[keep], ya[keep], ym[keep], yc[keep]
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        sel = np.sort(rng.choice(len(X), max_n, replace=False))
        X, ya, ym, yc = X[sel], ya[sel], ym[sel], yc[sel]
    Z = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    S = (Z @ Z.T).astype(np.float32)
    np.fill_diagonal(S, -np.inf)
    same_ch = yc[:, None] == yc[None, :]
    same_pk = ym[:, None] == ym[None, :]

    def score(allow, tag):
        Sx = np.where(allow, S, np.float32(-np.inf))
        valid = np.isfinite(Sx).any(1)
        if valid.sum() < 50:
            print(f"    {tag:34s} (too few)")
            return
        o = np.argsort(-Sx[valid], axis=1)[:, :5]
        t, l = ya[valid], ya[o]
        c = Counter(t); n = len(t)
        ch = sum(k * (k - 1) for k in c.values()) / max(n * (n - 1), 1)
        print(f"    {tag:34s} r@1={100*(l[:,0]==t).mean():5.1f}%  "
              f"r@5={100*(l==t[:,None]).any(1).mean():5.1f}%  "
              f"chance={100*ch:4.1f}%  lift={(l[:,0]==t).mean()/max(ch,1e-9):5.1f}x")

    dims = len(CANON) + (len(ARM_SLOTS) if use_arms else 0)
    print(f"  encoder: {dims} channels  |  {len(X)} motions  "
          f"{len(set(ym))} packs  {len(set(yc))} chars")
    score(same_ch & ~same_pk, "same CHAR, diff pack")
    score(~same_ch, "DIFFERENT character")
    return X, ya, ym, yc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--whitelist", default="outputs/model_whitelist.json")
    ap.add_argument("--max-n", type=int, default=4000)
    args = ap.parse_args()
    root = Path(args.root)
    models = sorted(orjson.loads(Path(args.whitelist).read_bytes())["kept"])

    # how many packs can even supply the arm slots?
    print("=== arm slot availability (from motion curves, not moc3) ===")
    cov: Counter = Counter()
    chars_with: dict[str, set] = {s: set() for s in ARM_SLOTS}
    for name in models:
        d = root / name / "motions"
        if not d.is_dir():
            continue
        f = next(iter(sorted(d.glob("*.json"))), None)
        if f is None:
            continue
        try:
            j = orjson.loads(f.read_bytes())
        except Exception:
            continue
        got = set()
        for c in j.get("Curves", []):
            pid = str(c.get("Id", ""))
            if ARM_HINT.search(pid):
                k = canon_arm(pid)
                if k in chars_with:
                    got.add(k)
        for k in got:
            cov[k] += 1
            chars_with[k].add(char_of(name))
    for s in ARM_SLOTS:
        print(f"  {s:10s} {cov[s]:3d}/{len(models)} packs   "
              f"{len(chars_with[s]):2d} characters")

    print("\n=== A) facial-only encoder (the original 14 channels) ===")
    run(root, models, False, args.max_n, 0)
    print("\n=== B) facial + arm encoder ===")
    run(root, models, True, args.max_n, 0)


if __name__ == "__main__":
    main()
