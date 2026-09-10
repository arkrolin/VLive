"""Decision #1: constant channels are dropped from the generation target, and at
playback time we re-inject *the value the author actually used* (not the moc3 default).

This script separates every (model, parameter) pair into three classes by looking at
how the value behaves inside each motion and across motions:

    const   within-motion variance == 0 for every motion, and one global value
            -> a true static channel. Inject the single value at playback.

    switch  within-motion variance == 0 for every motion, but different motions
            use different values
            -> a pose switch (costume / prop / expression preset). NOT a motion
               channel, but it carries real information: it should be a *condition*
               on generation, not a regression target. We record the per-motion value
               and the corpus-level mode.

    vary    at least one motion animates it
            -> candidate motion channel, handed to build_gen_mask.py

PartOpacity curves get the same treatment in a separate section, because they are
14.6% of all curves and are pure layer show/hide.

Output: outputs/static_pose.json

Usage:
    uv run python tools/build_static_pose.py
    uv run python tools/build_static_pose.py --limit 20 --verbose
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from motion_curves import curve_extent  # noqa: E402

ROUND = 4          # value quantisation for "is it the same value" tests
FPS = 30.0
EPS = 1e-4         # within-motion peak-to-peak below this counts as flat


def _safe(s: str) -> str:
    """Make a filename UTF-8-serialisable.

    Live2d-model-master ships GBK/Shift-JIS directory names, which Python reads
    as lone surrogates; orjson (used below) raises
    `TypeError: str is not valid UTF-8: surrogates not allowed` on those.
    Names are now ASCII by construction (see outputs/make_unified_root.py); this
    is belt-and-braces so the tool never dies on one bad directory again.
    """
    try:
        s.encode("utf-8")
        return s
    except UnicodeEncodeError:
        return s.encode("utf-8", "backslashreplace").decode("utf-8")


def load_json(p: Path):
    try:
        return orjson.loads(p.read_bytes())
    except Exception:
        try:
            return orjson.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def motion_files(model_dir: Path) -> list[Path]:
    mdir = model_dir / "motions"
    if not mdir.is_dir():
        return []
    # dedupe by filename (some packs ship duplicates in subdirs)
    return list({f.name: f for f in mdir.glob("*.json")}.values())


def scan_motion(path: Path) -> dict[str, dict[str, float]]:
    """-> {target_kind: {id: {flat, val, lo, hi}}}, one entry per curve.

    Flatness is decided from the curve's exact value extent (keyframes + Bezier
    control points), not from a 30fps resample. A Bezier bulge between two equal
    keyframes still shows up, because its control-point values are included.
    """
    j = load_json(path)
    if not j:
        return {}
    dur = float((j.get("Meta") or {}).get("Duration") or 0.0)
    if dur <= 0:
        return {}

    out: dict[str, dict] = {"Parameter": {}, "PartOpacity": {}}
    for c in (j.get("Curves") or []):
        tgt = c.get("Target")
        if tgt not in out:
            continue
        cid = c.get("Id")
        if not cid:
            continue
        lo, hi, v0 = curve_extent(c.get("Segments") or [])
        flat = (hi - lo) <= EPS
        out[tgt][cid] = {
            "flat": flat,
            "val": round(v0 if flat else 0.5 * (lo + hi), ROUND),
            "lo": lo,
            "hi": hi,
        }
    return out


def classify_model(model_dir: Path) -> dict | None:
    files = motion_files(model_dir)
    if not files:
        return None

    # per target kind: pid -> {motion_name: record}
    seen: dict[str, dict[str, dict[str, dict]]] = {
        "Parameter": defaultdict(dict),
        "PartOpacity": defaultdict(dict),
    }
    for mf in files:
        rec = scan_motion(mf)
        for kind, d in rec.items():
            for pid, r in d.items():
                seen[kind][pid][mf.name] = r
    if not seen["Parameter"]:
        return None

    result: dict[str, dict] = {"n_motions": len(files)}
    for kind in ("Parameter", "PartOpacity"):
        const: dict[str, float] = {}
        switch: dict[str, dict] = {}
        vary: list[str] = []
        partial: dict[str, int] = {}     # pid -> motions it is absent from

        for pid, by_motion in seen[kind].items():
            n_absent = len(files) - len(by_motion)
            if n_absent:
                partial[pid] = n_absent
            any_animated = any(r["flat"] is False for r in by_motion.values())
            if any_animated:
                vary.append(pid)
                continue
            vals = [r["val"] for r in by_motion.values()]
            uniq = sorted(set(vals))
            if len(uniq) == 1:
                const[pid] = uniq[0]
            else:
                cnt = Counter(vals)
                switch[pid] = {
                    "mode": cnt.most_common(1)[0][0],
                    "values": uniq[:12],
                    "n_values": len(uniq),
                    "by_motion": {m: r["val"] for m, r in by_motion.items()},
                }
        key = "param" if kind == "Parameter" else "partopacity"
        result[key] = {
            "const": const,
            "switch": switch,
            "vary": sorted(vary),
            "partial": partial,
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--out", default="outputs/static_pose.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    models = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        models = models[: args.limit]

    out: dict[str, dict] = {}
    agg = Counter()
    per_model_counts = defaultdict(list)
    switch_examples: Counter[str] = Counter()

    for md in tqdm(models, desc="models", ncols=80):
        r = classify_model(md)
        if r is None:
            agg["skipped (no usable motions)"] += 1
            continue
        out[_safe(md.name)] = r
        p = r["param"]
        agg["models"] += 1
        agg["const"] += len(p["const"])
        agg["switch"] += len(p["switch"])
        agg["vary"] += len(p["vary"])
        agg["partop_const"] += len(r["partopacity"]["const"])
        agg["partop_switch"] += len(r["partopacity"]["switch"])
        agg["partop_vary"] += len(r["partopacity"]["vary"])
        per_model_counts["const"].append(len(p["const"]))
        per_model_counts["switch"].append(len(p["switch"]))
        per_model_counts["vary"].append(len(p["vary"]))
        per_model_counts["partial"].append(len(p["partial"]))
        for pid in p["switch"]:
            switch_examples[pid] += 1

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "fps": FPS,
            "round": ROUND,
            "flat_eps": EPS,
            "n_models": len(out),
            "doc": "const=inject at playback; switch=condition, not target; "
                   "vary=candidate motion channel",
        },
        "models": out,
    }
    outp.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    tot = agg["const"] + agg["switch"] + agg["vary"]
    print(f"\n=== {agg['models']} models, {tot} (model,param) pairs ===")
    for k in ("const", "switch", "vary"):
        print(f"  {k:8s} {agg[k]:7d}  {100*agg[k]/max(tot,1):5.1f}%")
    print(f"\n  per-model median: "
          f"const={int(np.median(per_model_counts['const']))} "
          f"switch={int(np.median(per_model_counts['switch']))} "
          f"vary={int(np.median(per_model_counts['vary']))} "
          f"partial={int(np.median(per_model_counts['partial']))}")
    ptot = agg["partop_const"] + agg["partop_switch"] + agg["partop_vary"]
    if ptot:
        print(f"\n  PartOpacity ({ptot} pairs): "
              f"const={100*agg['partop_const']/ptot:.1f}% "
              f"switch={100*agg['partop_switch']/ptot:.1f}% "
              f"vary={100*agg['partop_vary']/ptot:.1f}%")
    if args.verbose:
        print("\n  most common SWITCH params (pose/costume/prop presets):")
        for pid, c in switch_examples.most_common(20):
            print(f"    {100*c/max(agg['models'],1):5.1f}%  {pid}")
    print(f"\nwrote {outp}  ({outp.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
