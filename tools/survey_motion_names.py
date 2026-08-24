"""Are motion FILE NAMES a usable text-condition label, and are they shared
across models?

If the same action name (dazhaohu = wave hello, bixin = finger heart) appears in
many models, we get something the Live2D literature does not have: paired data of
"one semantic action, N different riggings". That is direct supervision for the
cross-model generalisation problem, for free.

Usage:
    uv run python tools/survey_motion_names.py --verbose
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import orjson

# Mgirl06_haixiu_a.motion3.json -> prefix 'M', char 'girl06', action 'haixiu',
# variant 'a'
NAME_RE = re.compile(
    r"^(?P<prefix>[A-Za-z]{0,2}?)"
    r"(?P<char>girl\d+|boy\d+|[A-Za-z]*?\d+)?"
    r"_?(?P<action>[a-z][a-z_]*?)"
    r"(?:_(?P<variant>[a-z]))?"
    r"(?P<num>\d*)$",
    re.I,
)
# suffixes authors use for "same action, another take"
VARIANT_TAIL = re.compile(r"(_[a-z]|\d)+$")


def norm_action(stem: str) -> tuple[str, str]:
    """motion file stem -> (action_key, raw_body). Strips char id and variant."""
    s = stem
    for suf in (".motion3", ".exp3"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    # drop leading char token: Mgirl06_ / Egirl06_ / girl08_ / char_
    parts = s.split("_", 1)
    body = parts[1] if len(parts) > 1 and re.search(r"\d", parts[0]) else s
    body = body.lower()
    core = VARIANT_TAIL.sub("", body)
    return (core or body), body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--whitelist", default="outputs/model_whitelist.json")
    ap.add_argument("--out", default="outputs/motion_labels.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    kept = None
    wl = Path(args.whitelist)
    if wl.exists():
        kept = set(orjson.loads(wl.read_bytes())["kept"])

    action_models: dict[str, set[str]] = defaultdict(set)
    action_count: Counter[str] = Counter()
    per_model: dict[str, dict[str, str]] = {}
    n_models = 0
    n_files = 0
    n_pinyin_like = 0

    for md in sorted(d for d in root.iterdir() if d.is_dir()):
        mdir = md / "motions"
        if not mdir.is_dir():
            continue
        files = sorted({f.name: f for f in mdir.glob("*.json")}.values())
        if not files:
            continue
        n_models += 1
        labels = {}
        for f in files:
            n_files += 1
            core, body = norm_action(f.stem)
            labels[f.name] = core
            action_models[core].add(md.name)
            action_count[core] += 1
            if re.fullmatch(r"[a-z]{3,}", core):
                n_pinyin_like += 1
        per_model[md.name] = labels

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(orjson.dumps(
        {"_meta": {"n_models": n_models, "n_files": n_files,
                   "n_distinct_actions": len(action_models)},
         "models": per_model,
         "action_coverage": {a: len(m) for a, m in
                             sorted(action_models.items(),
                                    key=lambda kv: -len(kv[1]))}},
        option=orjson.OPT_INDENT_2))

    shared = {a: len(m) for a, m in action_models.items() if len(m) >= 2}
    cov = np.array(sorted(shared.values(), reverse=True)) if shared else np.zeros(1)
    print(f"=== {n_models} models, {n_files} motion files ===")
    print(f"  distinct action keys      : {len(action_models)}")
    print(f"  shared by >=2 models      : {len(shared)} "
          f"({100*len(shared)/max(len(action_models),1):.1f}%)")
    print(f"  shared by >=10 models     : {int((cov>=10).sum())}")
    print(f"  shared by >=50 models     : {int((cov>=50).sum())}")
    print(f"  shared by >=100 models    : {int((cov>=100).sum())}")
    files_in_shared = sum(action_count[a] for a in shared)
    print(f"  motion files under a shared action: {files_in_shared}/{n_files} "
          f"({100*files_in_shared/max(n_files,1):.1f}%)")
    print(f"  lowercase-latin action keys (pinyin-like): "
          f"{100*n_pinyin_like/max(n_files,1):.1f}% of files")

    if kept:
        pair_tot = 0
        for a, ms in action_models.items():
            k = len(ms & kept)
            if k >= 2:
                pair_tot += k * (k - 1) // 2
        print(f"\n  within whitelist ({len(kept)} models): "
              f"{pair_tot} same-action cross-model PAIRS available")

    print(f"\n-- top shared actions (model coverage) --")
    for a, ms in sorted(action_models.items(), key=lambda kv: -len(kv[1]))[:40]:
        print(f"  {len(ms):4d} models  x{action_count[a]:4d} files  {a}")

    if args.verbose:
        print(f"\n-- singleton actions (1 model only), sample --")
        singles = [a for a, m in action_models.items() if len(m) == 1]
        print(f"  count = {len(singles)}")
        for a in sorted(singles)[:30]:
            print(f"    {a}")


if __name__ == "__main__":
    main()
