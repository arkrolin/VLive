"""Command-line entry point for the rig-retrieval module.

Subcommands
-----------
  build     Materialise outputs/retrieval_index.jsonl from the corpus.
  retrieve  Print the K closest-source same-action references for a target.
  validate  Reproduce the V6.2 auto3 numbers as an equivalence regression.

Examples
--------
  uv run python -m deploy.rig_retrieval.cli build
  uv run python -m deploy.rig_retrieval.cli retrieve \\
        --target l2d22.ugirl06 --action haixiu --k 3
  uv run python -m deploy.rig_retrieval.cli retrieve \\
        --target /path/to/new_model_dir --action wave --k 3
  uv run python -m deploy.rig_retrieval.cli validate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

from .build_index import build_corpus_index, run_build  # noqa: E402
from .index import CorpusIndex                                       # noqa: E402
from .retriever import Retriever                                     # noqa: E402
from .validate import run as validate_run                            # noqa: E402


def cmd_build(args) -> None:
    run_build(ROOT / "standrad-live-2d", ROOT / "outputs/retrieval_index.jsonl",
              ROOT / "outputs/_paramset_cache.json",
              ROOT / "outputs/model_whitelist.json")


def cmd_retrieve(args) -> None:
    idx = CorpusIndex.load(ROOT / args.index)
    ret = Retriever(idx)

    target = args.target
    packs = {e.pack for e in idx.entries}
    if target in packs:
        res = ret.retrieve_for_pack(target, args.action, k=args.k, fuzzy=args.fuzzy)
    else:
        p = Path(target)
        if not p.exists():
            sys.exit(f"[retrieve] target {target!r} is neither a known pack "
                     f"nor a path on disk")
        res = ret.retrieve_for_model(p, args.action, k=args.k, fuzzy=args.fuzzy)

    print(res.summary())
    if args.json:
        Path(args.json).write_text(
            json.dumps(res.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n[wrote JSON] {args.json}")


def cmd_validate(args) -> None:
    validate_run(ROOT / args.index)


def main() -> None:
    ap = argparse.ArgumentParser(prog="rig_retrieval", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="build the corpus index").set_defaults(
        func=cmd_build)

    p_r = sub.add_parser("retrieve", help="retrieve same-source references")
    p_r.add_argument("--index", default="outputs/retrieval_index.jsonl")
    p_r.add_argument("--target", required=True,
                     help="corpus pack id, or a path to a model dir / .moc3")
    p_r.add_argument("--action", required=True, help="desired action key")
    p_r.add_argument("--k", type=int, default=3)
    p_r.add_argument("--fuzzy", action="store_true",
                     help="allow substring / shared-token action match")
    p_r.add_argument("--json", default="", help="also write result as JSON")
    p_r.set_defaults(func=cmd_retrieve)

    p_v = sub.add_parser("validate", help="reproduce V6.2 numbers")
    p_v.add_argument("--index", default="outputs/retrieval_index.jsonl")
    p_v.set_defaults(func=cmd_validate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
