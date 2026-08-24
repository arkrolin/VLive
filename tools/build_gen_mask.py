"""Decisions #2 and #3, turned into the generation-target mask.

Filter chain (each step is a deterministic file-read rule, no model involved):

    start        parameters that at least one motion animates   (static_pose.vary)
  - physics      appears in this model's physics3.json Output.Destination
                 -> DECISION #2: the user's player runs physics, so these are
                    handed to the physics engine and dropped from the target
  - nonmotion    classify_hetero bucket in {fx, macro, prop}
                 -> NOTE: bucket 'tail' is deliberately KEPT. The pinyin lexicon
                    is incomplete, so tail hides real arm/body params. They are
                    flagged needs_probe and will be re-triaged by the sweep probe.
  = target       the channels flow matching has to produce

Model whitelist (DECISION #3): models whose moc3 Part ids are mostly digits carry
no semantic part names, so the probe cannot ground their parameters. Excluded from
M1/M2, revisited in M3 with geometry retrieval.

Output: outputs/gen_mask.json, outputs/model_whitelist.json

Usage:
    uv run python tools/build_gen_mask.py
    uv run python tools/build_gen_mask.py --verbose
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent))
from classify_hetero import classify  # noqa: E402
from survey_parts import norm_part  # noqa: E402
from exclusion_tables import build_exclusion_set  # noqa: E402

PART_RE = re.compile(rb"PARTS?[_A-Za-z0-9]{1,60}")
DROP_BUCKETS = {"fx", "macro", "prop"}
# a part id is "semantic" if, after stripping the template prefix/suffix noise,
# anything alphabetic is left
SEMANTIC_PART = re.compile(r"[A-Za-z]{2,}")
# what the user explicitly asked to keep: body + arms
ARM_RE = re.compile(r"ARM|HAND|SHOULDER|ELBOW|WRIST|FINGER|SHOU|GEBO", re.I)
BODY_RE = re.compile(r"BODY|TORSO|WAIST|CHEST|HIP|SHEN|YAO", re.I)


def load_json(p: Path):
    try:
        return orjson.loads(p.read_bytes())
    except Exception:
        return None


def physics_outputs(model_dir: Path) -> set[str]:
    p3 = next(model_dir.glob("*.physics3.json"), None)
    if not p3:
        return set()
    j = load_json(p3) or {}
    outs = set()
    for st in (j.get("PhysicsSettings") or []):
        for o in (st.get("Output") or []):
            d = (o.get("Destination") or {}).get("Id")
            if d:
                outs.add(d)
    return outs


def part_semantics(model_dir: Path) -> tuple[int, int, list[str]]:
    """-> (n_semantic_parts, n_total_parts, sample of normalised names)"""
    moc = next(model_dir.glob("*.moc3"), None)
    if not moc:
        return 0, 0, []
    raw = moc.read_bytes()
    parts = {m.group().decode("ascii", "ignore") for m in PART_RE.finditer(raw)}
    parts = {p for p in parts if not p.startswith("PARAM")}
    if not parts:
        return 0, 0, []
    normed = [norm_part(p) for p in parts]
    sem = [p for p in normed if SEMANTIC_PART.search(p)]
    return len(sem), len(normed), sorted(set(normed))[:8]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--static-pose", default="outputs/static_pose.json")
    ap.add_argument("--out", default="outputs/gen_mask.json")
    ap.add_argument("--out-whitelist", default="outputs/model_whitelist.json")
    ap.add_argument("--min-semantic-part-ratio", type=float, default=0.5,
                    help="model kept if >= this share of its parts have letters")
    ap.add_argument("--min-target-dims", type=int, default=8,
                    help="drop models whose target ends up smaller than this")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    sp = load_json(Path(args.static_pose))
    if not sp:
        raise SystemExit(f"missing {args.static_pose}; run build_static_pose.py first")
    sp_models = sp["models"]

    masks: dict[str, dict] = {}
    whitelist: dict[str, dict] = {}
    funnel = Counter()
    dims = defaultdict(list)
    dropped_by_bucket: Counter[str] = Counter()
    kept_tail: Counter[str] = Counter()
    arm_survivors, body_survivors = [], []

    for name, rec in sp_models.items():
        mdir = root / name
        vary = rec["param"]["vary"]
        if not vary:
            continue
        outs = physics_outputs(mdir)
        ex = build_exclusion_set(mdir, vary)
        blink_set = ex["blink"]
        lip_set = ex["lipsync"]
        n_sem, n_part, sample = part_semantics(mdir)
        sem_ratio = n_sem / n_part if n_part else 0.0

        after_phys = [p for p in vary if p not in outs]
        target, dropped = [], []
        for p in after_phys:
            if p in blink_set:
                dropped.append(p)
                dropped_by_bucket["blink"] += 1
                continue
            if p in lip_set:
                dropped.append(p)
                dropped_by_bucket["lipsync"] += 1
                continue
            bucket, group = classify(p)
            if bucket in DROP_BUCKETS:
                dropped.append(p)
                dropped_by_bucket[bucket] += 1
            else:
                target.append(p)
                if bucket == "tail":
                    kept_tail[p] += 1

        funnel["vary"] += len(vary)
        funnel["after_physics"] += len(after_phys)
        funnel["target"] += len(target)
        dims["vary"].append(len(vary))
        dims["after_physics"].append(len(after_phys))
        dims["target"].append(len(target))

        n_arm = sum(1 for p in target if ARM_RE.search(p))
        n_body = sum(1 for p in target if BODY_RE.search(p))
        arm_survivors.append(n_arm)
        body_survivors.append(n_body)

        keep = (sem_ratio >= args.min_semantic_part_ratio
                and len(target) >= args.min_target_dims)
        reason = []
        if sem_ratio < args.min_semantic_part_ratio:
            reason.append(f"part_names_not_semantic({sem_ratio:.0%})")
        if len(target) < args.min_target_dims:
            reason.append(f"target_too_small({len(target)})")
        if n_arm == 0:
            reason.append("no_arm_channel")   # warning only, not a rejection

        masks[name] = {
            "target": sorted(target),
            "n_target": len(target),
            "n_arm": n_arm,
            "n_body": n_body,
            "excluded": {
                "const": sorted(rec["param"]["const"].keys()),
                "switch": sorted(rec["param"]["switch"].keys()),
                "physics_out": sorted(p for p in vary if p in outs),
                "blink": sorted(blink_set),
                "lipsync": sorted(lip_set),
                "nonmotion": sorted(dropped),
            },
            "needs_probe": sorted(p for p in target if classify(p)[0] == "tail"),
            "part_semantic_ratio": round(sem_ratio, 3),
            "n_parts": n_part,
        }
        whitelist[name] = {
            "keep": keep,
            "reasons": reason,
            "n_target": len(target),
            "part_semantic_ratio": round(sem_ratio, 3),
            "sample_parts": sample,
        }

    kept = [k for k, v in whitelist.items() if v["keep"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(orjson.dumps(
        {"_meta": {"drop_buckets": sorted(DROP_BUCKETS),
                   "note": "bucket 'tail' kept on purpose, see needs_probe"},
         "models": masks}, option=orjson.OPT_INDENT_2))
    Path(args.out_whitelist).write_bytes(orjson.dumps(
        {"_meta": {"min_semantic_part_ratio": args.min_semantic_part_ratio,
                   "min_target_dims": args.min_target_dims,
                   "n_kept": len(kept), "n_total": len(whitelist)},
         "kept": sorted(kept),
         "models": whitelist}, option=orjson.OPT_INDENT_2))

    print(f"=== filter funnel over {len(masks)} models ===")
    for k in ("vary", "after_physics", "target"):
        arr = np.array(dims[k])
        print(f"  {k:14s} total={funnel[k]:6d}  per-model median={int(np.median(arr))} "
              f"p10={int(np.percentile(arr,10))} p90={int(np.percentile(arr,90))}")
    print(f"\n  physics removed : {funnel['vary']-funnel['after_physics']} "
          f"({100*(funnel['vary']-funnel['after_physics'])/max(funnel['vary'],1):.1f}%)")
    print(f"  nonmotion removed: {funnel['after_physics']-funnel['target']} "
          f"({100*(funnel['after_physics']-funnel['target'])/max(funnel['after_physics'],1):.1f}%)")
    for b, c in dropped_by_bucket.most_common():
        print(f"      {b:8s} {c}")

    print(f"\n=== surviving target composition by bucket ===")
    comp: Counter[str] = Counter()
    compg: Counter[str] = Counter()
    for m in masks.values():
        for p in m["target"]:
            b, g = classify(p)
            comp[b] += 1
            compg[g] += 1
    tt = sum(comp.values())
    for b, c in comp.most_common():
        print(f"  {b:14s} {c:6d}  {100*c/max(tt,1):5.1f}%")
    print("  " + "  ".join(f"{g}={100*c/max(tt,1):.1f}%" for g, c in compg.most_common()))

    print(f"\n=== user requirement check: arms + body must survive ===")
    a, b = np.array(arm_survivors), np.array(body_survivors)
    print(f"  arm channels per model : median={int(np.median(a))} "
          f"zero-arm models={int((a==0).sum())}/{len(a)}")
    print(f"  body channels per model: median={int(np.median(b))} "
          f"zero-body models={int((b==0).sum())}/{len(b)}")

    print(f"\n=== whitelist (decision #3) ===")
    print(f"  kept {len(kept)}/{len(whitelist)} models")
    rc = Counter()
    for v in whitelist.values():
        if not v["keep"]:
            for r in v["reasons"]:
                if not r.startswith("no_arm"):
                    rc[r.split("(")[0]] += 1
    for r, c in rc.most_common():
        print(f"    rejected by {r}: {c}")
    if kept:
        kt = np.array([whitelist[k]["n_target"] for k in kept])
        print(f"  kept models target dims: median={int(np.median(kt))} "
              f"p10={int(np.percentile(kt,10))} p90={int(np.percentile(kt,90))}")

    if args.verbose:
        print("\n  top 'tail' params kept for probe triage (lexicon misses):")
        for p, c in kept_tail.most_common(25):
            print(f"    x{c:3d}  {p}")


if __name__ == "__main__":
    main()
