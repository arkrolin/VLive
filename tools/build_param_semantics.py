"""Build param_semantics.json: each hetero param id -> English semantic
description, for the semantic-token diffusion design.

See 2026-07-17_token_diffusion_impl.md section 2.1. The description feeds a
frozen text encoder to produce the per-parameter semantic embedding e_p, which
gives each token a cross-model-aligned identity ("left arm" maps to the same
cluster regardless of the raw id spelling).

Strategy (not 5197 hardcoded entries): a body-part lexicon (English + pinyin)
+ structural decomposition of the id (strip PARAM_/PARTS_ prefix, split on _,
translate each token, expand side/axis/level suffixes). First covers the
high-frequency curated core; the long tail falls through to the same algorithm
(good enough for e_p).

Usage:
    uv run python tools/build_param_semantics.py --out param_semantics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

# reuse the classifier's bucket/group logic
import sys
sys.path.insert(0, str(Path(__file__).parent))
from classify_hetero import classify  # noqa: E402

# --- body-part lexicon: token -> English word (English keys + pinyin) -------
_LEX: dict[str, str] = {
    # limbs (english)
    "ARM": "arm", "HAND": "hand", "SHOULDER": "shoulder", "ELBOW": "elbow",
    "WRIST": "wrist", "FINGER": "finger", "FINGERS": "fingers",
    "LEG": "leg", "FOOT": "foot", "FOOD": "foot", "KNEE": "knee",
    "THIGH": "thigh", "ANKLE": "ankle",
    # limbs (pinyin)
    "SHOUWAN": "wrist", "SHOUBI": "forearm", "GEBO": "arm", "SHOUZHI": "finger",
    "SHIZHI": "index finger", "ZHONGZHI": "middle finger",
    "WUMINGZHI": "ring finger", "XIAOZHI": "little finger", "MUZHI": "thumb",
    "TUI": "leg", "JIAO": "foot",
    # face / head
    "EYE": "eye", "BALL": "eyeball", "BROW": "eyebrow", "MOUTH": "mouth",
    "CHEEK": "cheek", "NOSE": "nose", "EAR": "ear", "FACE": "face",
    "NECK": "neck", "TEAR": "tears", "TONGUE": "tongue", "TEETH": "teeth",
    "SHETOU": "tongue", "TERE": "blush", "HAN": "sweat drop",
    "PUPIL": "pupil", "FOCUS": "gaze focus", "TAIJIAN": "eyelid",
    "HEIYANQUAN": "eye bags",
    # body / torso
    "BODY": "body", "HEAD": "head", "BUST": "chest", "BREAST": "chest",
    "MUNE": "chest", "CHEST": "chest", "BELLY": "belly", "WAIST": "waist",
    "HIP": "hip", "FAT": "body fullness", "XIABANSHEN": "lower body",
    "UPPER": "upper body",
    # soft body
    "HAIR": "hair", "KAMI": "hair", "SKIRT": "skirt", "QUN": "skirt",
    "CLOTH": "clothes", "DRESS": "dress", "SLEEVE": "sleeve", "XIU": "sleeve",
    "RIBBON": "ribbon", "SCARF": "scarf", "CAPE": "cape", "COAT": "coat",
    "AHOGE": "hair strand", "ANTENNA": "hair antenna",
    # front/back/side hair
    "FRONT": "front", "BACK": "back", "SIDE": "side",
    # motion descriptors
    "WAVE": "sway", "ROLL": "roll", "MOVE": "motion", "SMALL": "small",
    "BIG": "big", "LEVEL": "level", "DEFORM": "deformation",
    "SPECIAL": "special", "CHANGE": "change", "FORM": "form", "SIZE": "size",
    "WID": "width", "SMILE": "smile", "SHINE": "shine", "OPEN": "open",
    "ANI": "animation", "DONGHUA": "animation", "QIEHUAN": "switch",
    "TOUSHI": "perspective", "BIGROLL": "big roll",
    # appearance / fx
    "HEIXIAN": "dark lines", "HEI": "darkening", "DARK": "dark",
    "LINE": "outline", "YINYING": "shadow", "WATER": "water", "STAR": "star",
    "RED": "red overlay", "FLOWER": "flower", "COLOR": "color",
    "SHADOW": "shadow", "LIGHT": "lighting", "ZHAMAO": "blink macro",
    # emotion macros
    "ANGRY": "angry", "SHY": "shy", "JINGYA": "surprised", "TANQI": "sigh",
    "EMOTION": "emotion",
    # props
    "GUN": "gun prop", "DAOJU": "prop", "DAO": "prop", "ITEM": "item",
    "WEAPON": "weapon", "TAIL": "tail", "WING": "wing", "HALO": "halo",
    "GLASS": "glasses", "HAT": "hat", "BAG": "bag", "CAT": "cat feature",
    "BG": "background", "BACKGROUND": "background",
}

_SIDE = {"L": "left", "R": "right"}
_AXIS = {"X": "horizontal", "Y": "vertical", "Z": "depth"}


def _translate_token(tok: str) -> str | None:
    """Translate one underscore-token. Returns None to drop noise tokens."""
    up = tok.upper()
    if up in _LEX:
        return _LEX[up]
    if up in _SIDE:
        return _SIDE[up]
    if up in _AXIS:
        return _AXIS[up]
    # bare number / numbered part index -> drop (adds no semantics)
    if re.fullmatch(r"\d+", tok):
        return None
    # trailing digits on a word: ARM2 -> arm; keep base if in lexicon
    m = re.fullmatch(r"([A-Za-z]+?)(\d+)", tok)
    if m and m.group(1).upper() in _LEX:
        return _LEX[m.group(1).upper()]
    # unknown pinyin/english token: lowercase as-is (still a usable signal)
    return tok.lower()


def _split_camel(s: str) -> str:
    """Insert _ at camelCase / letter-digit boundaries so ArmR -> Arm_R."""
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)   # armR -> arm_R
    s = re.sub(r"([A-Za-z])(\d)", r"\1_\2", s)   # Arm2 -> Arm_2
    return s


def describe(pid: str) -> str:
    """Build an English semantic description from a raw parameter id.

    Side words (left/right) are moved to the front so that differently-spelled
    ids for the same concept (ARM_L, LEFT_ARM, ArmR) converge on a consistent
    word order ("left arm"), tightening the semantic-embedding clusters.
    """
    s = pid
    s = re.sub(r"^(PARAM_|PARTS_)", "", s, flags=re.I)  # strip prefix
    s = re.sub(r"^\d+_", "", s)                          # PARTS_01_ -> ''
    s = _split_camel(s)                                  # ArmR -> Arm_R
    parts = [p for p in s.split("_") if p]
    words: list[str] = []
    for p in parts:
        w = _translate_token(p)
        if w:
            words.append(w)
    if not words:
        return "misc parameter"
    # pull side words (left/right) to the front for consistent ordering
    sides = [w for w in words if w in ("left", "right")]
    rest = [w for w in words if w not in ("left", "right")]
    ordered = sides + rest
    # dedupe consecutive repeats while keeping order
    out: list[str] = []
    for w in ordered:
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", default="out/shards")
    ap.add_argument("--out", default="param_semantics.json")
    ap.add_argument("--curated-min-models", type=int, default=10,
                    help="ids appearing in >= this many models are tagged curated")
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    freq: Counter[str] = Counter()
    for sp in sorted(glob.glob(str(shard_dir / "*.npz"))):
        d = np.load(sp, allow_pickle=True)
        for h in set(str(x) for x in d["hetero_ids"]):
            freq[h] += 1

    out: dict[str, dict] = {}
    for pid, n_models in freq.items():
        bucket, group = classify(pid)
        out[pid] = {
            "desc": describe(pid),
            "group": group,
            "bucket": bucket,
            "n_models": n_models,
            "src": "curated" if n_models >= args.curated_min_models else "auto",
        }

    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    n_cur = sum(1 for v in out.values() if v["src"] == "curated")
    print(f"[SUCCESS] wrote {args.out}  ({len(out)} ids, {n_cur} curated)")


if __name__ == "__main__":
    main()
