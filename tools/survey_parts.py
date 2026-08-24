"""Corpus-wide statistics on PART naming vs PARAMETER naming.

Hypothesis under test: parameter ids are chaotic (7976 distinct, 72.5% private
to one model), but PART ids follow the Cubism official template and are highly
standardised across authors. If true, "which component does parameter p drive?"
becomes a lookup (probe -> drawable -> part -> standard name) rather than a
recognition problem.

Usage:
    uv run --no-project python tools/survey_parts.py --root standrad-live-2d
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PART = re.compile(rb"PARTS?[_A-Za-z0-9]{1,60}")
PARAM = re.compile(rb"PARAM[_A-Za-z0-9]{1,60}")


def norm_part(s: str) -> str:
    """PARTS_01_HAIR_BACK_001 -> HAIR_BACK ; strip template index noise."""
    s = re.sub(r"^PARTS?_", "", s)
    s = re.sub(r"^\d\d_", "", s)          # 01_ / 00_
    s = re.sub(r"_\d{2,3}$", "", s)       # _001 / _00
    s = re.sub(r"_\d$", "", s)
    return s.upper()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    args = ap.parse_args()
    root = Path(args.root)
    models = sorted(d for d in root.iterdir() if d.is_dir())

    part_models: Counter[str] = Counter()      # normalised part -> #models
    part_raw: Counter[str] = Counter()
    param_models: Counter[str] = Counter()
    n_parts_per_model: list[int] = []
    n_ok = 0

    for md in models:
        moc = next(md.glob("*.moc3"), None)
        if not moc:
            continue
        raw = moc.read_bytes()
        parts = {m.group().decode("ascii", "ignore") for m in PART.finditer(raw)}
        parts = {p for p in parts if not p.startswith("PARAM")}
        params = {m.group().decode("ascii", "ignore") for m in PARAM.finditer(raw)}
        if not parts:
            continue
        n_ok += 1
        n_parts_per_model.append(len(parts))
        for p in parts:
            part_raw[p] += 1
            part_models[norm_part(p)] += 1
        for p in params:
            param_models[p] += 1

    def cov(c: Counter, k: int) -> int:
        return sum(1 for v in c.values() if v >= k)

    print(f"=== {n_ok} models parsed ===\n")
    print(f"parts per model: median={sorted(n_parts_per_model)[len(n_parts_per_model)//2]} "
          f"max={max(n_parts_per_model)}")
    print(f"distinct raw part ids        : {len(part_raw)}")
    print(f"distinct normalised part ids : {len(part_models)}")
    print(f"distinct param ids (in moc3) : {len(param_models)}\n")

    print("-- standardisation: how many ids are shared across models --")
    hdr = f"{'threshold':>12s} {'PART(norm)':>12s} {'PARAM':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for k, lbl in ((int(n_ok * 0.9), ">=90% models"), (int(n_ok * 0.5), ">=50%"),
                   (int(n_ok * 0.1), ">=10%"), (2, ">=2 models"), (1, "any")):
        print(f"{lbl:>12s} {cov(part_models,k):12d} {cov(param_models,k):10d}")

    priv_part = sum(1 for v in part_models.values() if v == 1)
    priv_param = sum(1 for v in param_models.values() if v == 1)
    print(f"\nsingle-model-only  PART : {priv_part}/{len(part_models)} "
          f"({100*priv_part/len(part_models):.1f}%)")
    print(f"single-model-only PARAM : {priv_param}/{len(param_models)} "
          f"({100*priv_param/len(param_models):.1f}%)")

    print("\n-- top normalised PART ids (coverage across models) --")
    for p, c in part_models.most_common(34):
        print(f"  {100*c/n_ok:5.1f}%  {p}")

    print("\n-- private / prop-like parts (1-3 models): the 'decoration' worry --")
    rare = [p for p, c in part_models.items() if c <= 3]
    print(f"  count = {len(rare)}   examples:")
    for p in sorted(rare)[:40]:
        print(f"    {p}")


if __name__ == "__main__":
    main()
