"""Classify hetero-layer params into A(active motion) / B(passive physics) /
C(non-motion), and dump per-model per-column labels.

See 2026-07-17_hetero_layer_restructure.md for the motivation and the full-scan
statistics that justify these buckets. Classification is id-regex based
(first-match-wins), covering both English and pinyin names. Unmatched ids fall
into C_nonmotion.tail — the model-private long tail that hetero+FSQ absorbs.

Usage:
    uv run python tools/classify_hetero.py --stats
    uv run python tools/classify_hetero.py --label --out hetero_labels.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

# --- fine buckets, ordered; first match wins ------------------------------
# Each entry: (bucket, group, regex). group in {A_active, B_passive, C_nonmotion}
_RULES: list[tuple[str, str, re.Pattern]] = [
    # A — active kinematic motion
    # NOTE ordering: body_angle must precede head_angle, otherwise
    # PARAM_BODY_ANGLE_X would be captured by the head rule's ANGLE_[XYZ].
    ("body_angle", "A_active", re.compile(
        r"BODY_?ANGLE|BODY_?[XYZ]$|UPPER_?BODY|LOWER_?BODY|BODY_?TUN|"
        r"BODY_?ROT|SHANGSHEN|XIASHEN", re.I)),
    ("head_angle", "A_active", re.compile(
        r"^PARAM_?ANGLE|HEAD_?ANGLE|(?<![A-Z])ANGLE_?[XYZ]|NECK|HEAD_?[XYZ]$|"
        r"TOUBU|BOZI", re.I)),
    ("breath", "A_active", re.compile(r"BREATH|KOKYU|HUXI", re.I)),
    ("limb_arm", "A_active", re.compile(
        r"ARM|HAND|SHOULDER|ELBOW|WRIST|FINGER|SHIZHI|XIAOZHI|WUMINGZHI|"
        r"ZHONGZHI|MUZHI|SHOUWAN|SHOUBI|GEBO|SHOUZHI|SHOU_", re.I)),
    ("limb_leg", "A_active", re.compile(
        r"LEG|FOOT|FOOD|KNEE|THIGH|ANKLE|TUI|JIAO", re.I)),
    ("global_xform", "A_active", re.compile(
        r"TOTAL|BODYZ|BodyZ|OTHERMOVE|ROOT|POSITION|POS_|PLACE[XY]|"
        r"[LR](EFT|IGHT)?PLACE", re.I)),
    ("face_detail", "A_active", re.compile(
        r"EYE|MOUTH|BROW|CHEEK|TEAR|BLUSH|TERE|TEETH|TONGUE|SHETOU|NOSE|"
        r"HOHO|NAMIDA|HAN|FOCUS|PUPIL|TAIJIAN", re.I)),
    # B — passive / soft-body physics
    ("body_phys", "B_passive", re.compile(
        r"BUST|BREAST|MUNE|CHEST|BELLY|WAIST|HIP|WAVE|FAT|SHISHEN", re.I)),
    ("hair", "B_passive", re.compile(
        r"HAIR|KAMI|MAE|YOKO|USHIRO|AHOGE|ANTENNA|LIU", re.I)),
    ("cloth", "B_passive", re.compile(
        r"SKIRT|CLOTH|DRESS|COAT|SLEEVE|RIBBON|SODE|SCARF|CAPE|FUKU|XIU|QUN",
        re.I)),
    # C — non-motion: appearance / macro / prop
    ("fx", "C_nonmotion", re.compile(
        r"HEIXIAN|LIANHEI|HEIYANQUAN|HEI|DARK|LINE|YINYING|SHINE|STAR|RED|"
        r"FLOWER|WATER|CAT_|EAR|GLASS|HAT|ZHAMAO|COLOR|ALPHA|OPACITY|"
        r"SHADOW|LIGHT", re.I)),
    ("macro", "C_nonmotion", re.compile(
        r"ANGRY|SHY|JINGYA|TANQI|SPECIAL|CHANGE|EMOTION|HEAD_SPECIAL|EXP_",
        re.I)),
    ("prop", "C_nonmotion", re.compile(
        r"DAOJU|GUN|ITEM|PROP|WEAPON|TAIL|WING|HALO|ACC|BAG|PARTS|DAO", re.I)),
]

_TAIL = ("tail", "C_nonmotion")  # fallback for unmatched ids


def classify(pid: str) -> tuple[str, str]:
    """Return (bucket, group) for a parameter id."""
    for bucket, group, pat in _RULES:
        if pat.search(pid):
            return bucket, group
    return _TAIL


def _iter_shards(shard_dir: Path):
    for sp in sorted(glob.glob(str(shard_dir / "*.npz"))):
        d = np.load(sp, allow_pickle=True)
        yield Path(sp).stem, [str(x) for x in d["hetero_ids"]]


def cmd_stats(shard_dir: Path) -> None:
    """Print column-share + model-presence per bucket and per group."""
    col = Counter()
    grp = Counter()
    presence = Counter()
    n = 0
    for _model, hids in _iter_shards(shard_dir):
        n += 1
        seen_bucket = set()
        for h in hids:
            bucket, group = classify(h)
            col[bucket] += 1
            grp[group] += 1
            seen_bucket.add(bucket)
        for b in seen_bucket:
            presence[b] += 1

    total = sum(col.values())
    print(f"shards {n}   total hetero columns {total}\n")
    print(f"{'bucket':16s} {'group':13s} {'cols':>6s} {'%cols':>7s} {'%models':>8s}")
    for bucket, group, _ in _RULES:
        c = col[bucket]
        print(f"{bucket:16s} {group:13s} {c:6d} {100*c/total:6.1f}% "
              f"{100*presence[bucket]/n:7.1f}%")
    tb, tg = _TAIL
    print(f"{tb:16s} {tg:13s} {col[tb]:6d} {100*col[tb]/total:6.1f}% "
          f"{100*presence[tb]/n:7.1f}%")
    print("\n=== group totals ===")
    for g in ("A_active", "B_passive", "C_nonmotion"):
        print(f"{g:13s} {grp[g]:6d}  ({100*grp[g]/total:.1f}%)")


def cmd_label(shard_dir: Path, out_path: Path) -> None:
    """Dump per-model per-column labels to JSON."""
    labels: dict[str, dict] = {}
    for model, hids in _iter_shards(shard_dir):
        cols = []
        for j, h in enumerate(hids):
            bucket, group = classify(h)
            cols.append({"col": j, "id": h, "group": group, "bucket": bucket})
        labels[model] = {
            "n_hetero": len(hids),
            "columns": cols,
        }
    out_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[SUCCESS] wrote {out_path}  ({len(labels)} models)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", action="store_true",
                    help="print bucket/group statistics over shards")
    ap.add_argument("--label", action="store_true",
                    help="dump per-model per-column labels")
    ap.add_argument("--shard-dir", default="out/shards",
                    help="directory of {model}.npz shards")
    ap.add_argument("--out", default="hetero_labels.json",
                    help="output path for --label")
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    if not (args.stats or args.label):
        ap.error("pass --stats and/or --label")

    if args.stats:
        cmd_stats(shard_dir)
    if args.label:
        cmd_label(shard_dir, Path(args.out))


if __name__ == "__main__":
    main()
