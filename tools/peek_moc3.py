"""Extract printable ID strings from .moc3 without the Cubism Core.

moc3 stores Parameter / Part / Drawable ids as fixed-width, NUL-padded ASCII
blocks. We can harvest them with a scan, which is enough to answer: *do the
art-mesh / part names carry human-readable semantics?* -- i.e. is there a
free supervision signal for "which component does this parameter drive?".

Usage:
    uv run --no-project python tools/peek_moc3.py --root standrad-live-2d --n 6
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(rb"[\x20-\x7e]{3,63}")


def harvest(p: Path) -> list[str]:
    raw = p.read_bytes()
    out, seen = [], set()
    for m in TOKEN.finditer(raw):
        s = m.group().decode("ascii", "ignore").strip()
        if len(s) < 3 or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    root = Path(args.root)

    models = sorted(d for d in root.iterdir() if d.is_dir())
    step = max(len(models) // args.n, 1)
    picked = models[::step][:args.n]

    art_like = Counter()
    for md in picked:
        moc = next(md.glob("*.moc3"), None)
        if not moc:
            continue
        ids = harvest(moc)
        # params referenced by the model's motions -> to separate params from meshes
        mparams: set[str] = set()
        for mf in (md / "motions").glob("*.json") if (md / "motions").is_dir() else []:
            try:
                j = json.loads(mf.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            for c in (j.get("Curves") or []):
                if c.get("Target") == "Parameter" and c.get("Id"):
                    mparams.add(c["Id"])
            break
        non_param = [i for i in ids if i not in mparams and not i.startswith("PARAM")]
        print(f"\n===== {md.name}  ({moc.stat().st_size//1024} KB, "
              f"{len(ids)} id-like strings) =====")
        print(f"  animated params in 1st motion : {len(mparams)}")
        print(f"  sample non-PARAM ids (parts / drawables):")
        for s in non_param[:28]:
            print(f"    {s}")
        for s in non_param:
            art_like[s] += 1

    print("\n\n=== ids shared across the sampled models (naming conventions) ===")
    for s, c in art_like.most_common(30):
        if c > 1:
            print(f"  x{c}  {s}")


if __name__ == "__main__":
    main()
