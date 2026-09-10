"""Flatten standrad-live-2d + Live2d-model-master into ONE directory so the
existing pipeline (build_static_pose.py uses root.iterdir(), i.e. ONE level)
works unchanged. Symlinks only - no data copied.

WHY PYTHON AND NOT BASH: 772 / 1604 model dirs under Live2d-model-master have
GBK/Shift-JIS bytes in their names, so Python sees them as lone surrogates and
orjson (used by every downstream script) raises
    TypeError: str is not valid UTF-8: surrogates not allowed
We therefore give every link an ASCII-SAFE name and record the real path in
outputs/all_manifest.json. Names are deterministic (sorted by real path).

Also fixes a bug in the old bash version: when the .moc3 lived one level above
the motions/ dir, `cd "$mdir/.." && pwd` produced an ABSOLUTE path, so 24 links
ended up named like l2dm____root__work__nlp__...

Usage:
    .venv/bin/python outputs/make_unified_root.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STD = ROOT / "standrad-live-2d"
L2DM = ROOT / "Live2d-model-master"
OUT = ROOT / "data" / "all"
MANIFEST = ROOT / "outputs" / "all_manifest.json"


def has_moc3(d: Path) -> bool:
    try:
        return any(f.suffix == ".moc3" for f in d.iterdir() if f.is_file())
    except Exception:
        return False


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() or OUT.is_symlink():
        # remove contents (it only holds symlinks)
        for p in OUT.iterdir():
            p.unlink()
        OUT.rmdir()
    OUT.mkdir(parents=True)

    manifest: dict[str, dict] = {}

    # ---- 1) standrad-live-2d (already one level) ----
    n_std = 0
    for d in sorted(STD.iterdir()):
        if not d.is_dir():
            continue
        name = "std__" + d.name
        (OUT / name).symlink_to(d.resolve())
        manifest[name] = {"corpus": "std", "path": str(d.resolve()), "game": "standrad"}
        n_std += 1
    print(f"std  linked: {n_std}")

    # ---- 2) Live2d-model-master (nested: game/.../model) ----
    motion_dirs = sorted({p.parent for p in L2DM.rglob("*.motion3.json")})
    n_l2 = 0
    skipped = 0
    for mdir in motion_dirs:
        model = mdir
        if not has_moc3(model):
            if has_moc3(mdir.parent):
                model = mdir.parent
            else:
                skipped += 1
                continue
        rel = model.relative_to(L2DM)
        game = rel.parts[0] if rel.parts else "?"
        # ASCII-safe, sortable, unique
        name = "l2dm__%05d" % n_l2
        (OUT / name).symlink_to(model.resolve())
        manifest[name] = {
            "corpus": "l2dm",
            "path": str(model.resolve()),
            "rel": str(rel),
            "game": game.encode("utf-8", "backslashreplace").decode("utf-8"),
        }
        n_l2 += 1
    print(f"l2dm linked: {n_l2}  (skipped, no moc3: {skipped})")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=True, indent=1), encoding="utf-8")

    broken = sum(1 for p in OUT.iterdir() if not p.exists())
    bad = 0
    for p in OUT.iterdir():
        try:
            p.name.encode("utf-8")
        except UnicodeEncodeError:
            bad += 1
    print(f"total      : {len(list(OUT.iterdir()))}")
    print(f"broken     : {broken}")
    print(f"non-utf8   : {bad}")
    print(f"manifest   : {MANIFEST}")


if __name__ == "__main__":
    main()
