"""Offline builder for the retrieval corpus index.

Run once (or whenever the corpus changes) to materialise
``outputs/retrieval_index.jsonl``. The builder is allowed to import the research
tooling (``tools/*``) -- the *output* index is what the deployment runtime
consumes, and the runtime module never imports ``tools``.

Param sets come from the precomputed ``_paramset_cache.json`` (identical to the
V6.2 experiment's source), falling back to live moc3 scanning when a pack is
missing from the cache. Action labels and character ids use the same
``norm_action`` / ``char_of`` as every other analysis, so the index lines up
exactly with the prior results.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent      # D:/VLive
sys.path.insert(0, str(ROOT / "tools"))

from audit_leakage import char_of          # noqa: E402
from survey_motion_names import norm_action  # noqa: E402
import orjson                               # noqa: E402

from .index import CorpusIndex, _Entry       # noqa: E402
from .moc3 import extract_param_set          # noqa: E402


def build_corpus_index(
    root: Path,
    param_cache: Path,
    whitelist: Path | None = None,
) -> CorpusIndex:
    root = Path(root)
    cache: dict = {}
    if Path(param_cache).exists():
        cache = orjson.loads(Path(param_cache).read_bytes())

    if whitelist and Path(whitelist).exists():
        models = sorted(orjson.loads(Path(whitelist).read_bytes())["kept"])
    else:
        models = sorted(d.name for d in root.iterdir() if d.is_dir())

    entries: list[_Entry] = []
    for pack in models:
        pdir = root / pack
        if not pdir.is_dir():
            continue
        char = char_of(pack)
        family = pack.split(".")[0]
        params = set(cache.get(pack, [])) or extract_param_set(pdir)
        mdir = pdir / "motions"
        if not mdir.is_dir():
            continue
        for f in sorted(mdir.glob("*.json")):
            act, _ = norm_action(f.stem)
            if not act or act in ("stand", "idle"):
                continue
            entries.append(_Entry(
                pack=pack, char=char, family=family, action=act,
                motion_path=str(f.resolve()), params=set(params),
            ))
    return CorpusIndex.from_entries(entries)


def run_build(root, out, param_cache, whitelist) -> CorpusIndex:
    """Build the index and write it to *out*. Returns the index too."""
    idx = build_corpus_index(
        Path(root), Path(param_cache),
        Path(whitelist) if whitelist else None,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    idx.save(out)
    print(f"[OK] index -> {out}  ({len(idx.entries)} clips, "
          f"{len(idx.actions())} actions, {len(idx.char_params)} characters)")
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the rig-retrieval corpus index.")
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--out", default="outputs/retrieval_index.jsonl")
    ap.add_argument("--param-cache", default="outputs/_paramset_cache.json")
    ap.add_argument("--whitelist", default="outputs/model_whitelist.json")
    args = ap.parse_args()
    run_build(ROOT / args.root, ROOT / args.out,
              ROOT / args.param_cache, ROOT / args.whitelist)


if __name__ == "__main__":
    main()
