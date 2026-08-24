"""Generation-target exclusion tables (decision #2 / #3, V7.1).

A parameter must be EXCLUDED from the model's generation target when it is
produced by something other than the authored motion:

    * physics   -> physics3.json Output.Destination   (player runs physics)
    * blink     -> Live2D EyeBlinkController auto-generates these
    * lipsync   -> audio-driven mouth-open (MouthController)
    * nonmotion -> fx / macro / prop buckets (classify_hetero)

build_exclusion_set(model_dir, vary) returns the per-category sets plus the
combined union. The training target = (params that vary in motion) - union.

This module is self-contained (re-implements physics_outputs, does NOT mutate
build_gen_mask.py) so it can be unit-tested and imported by #2's trainer.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import orjson

# auto-blink parameters (SDK EyeBlinkController owns these)
BLINK_RE = re.compile(r"(?i)(eye).{0,6}(blink)|(blink).{0,6}(eye)|eyeblink")
# audio lip-sync: the *opening* parameter is audio-driven; form/shape is not
LIPSYNC_RE = re.compile(r"(?i)mouth[_]?open|(mouth).{0,4}(open)")

DROP_BUCKETS = {"fx", "macro", "prop"}


def load_json(p: Path):
    try:
        return orjson.loads(p.read_bytes())
    except Exception:
        return None


def physics_outputs(model_dir: Path) -> set[str]:
    """Param ids driven by the physics engine (-> handed to physics, not generated)."""
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


def blink_params(vary: set[str]) -> set[str]:
    return {p for p in vary if BLINK_RE.search(p)}


def lipsync_params(vary: set[str]) -> set[str]:
    return {p for p in vary if LIPSYNC_RE.search(p)}


def nonmotion_params(vary: set[str]) -> set[str]:
    sys.path.insert(0, str(Path(__file__).parent))
    from classify_hetero import classify

    out = set()
    for p in vary:
        bucket, _ = classify(p)
        if bucket in DROP_BUCKETS:
            out.add(p)
    return out


def build_exclusion_set(model_dir, vary) -> dict[str, set[str]]:
    """Return {category: set} for one model.

    model_dir: path to the model folder (has *.physics3.json / *.model3.json)
    vary:      iterable of param ids that at least one motion animates
    """
    model_dir = Path(model_dir)
    vary = set(vary)
    phys = physics_outputs(model_dir) & vary
    blink = blink_params(vary)
    lip = lipsync_params(vary)
    nonm = nonmotion_params(vary)
    return {
        "physics": phys,
        "blink": blink,
        "lipsync": lip,
        "nonmotion": nonm,
        "all": phys | blink | lip | nonm,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--static-pose", default="outputs/static_pose.json")
    args = ap.parse_args()

    root = Path(args.root)
    mdir = root / args.model_dir
    sp = load_json(Path(args.static_pose)) or {}
    rec = (sp.get("models") or {}).get(args.model_dir)
    if not rec:
        print(f"no static_pose entry for {args.model_dir}")
        return
    vary = set(rec["param"]["vary"])
    ex = build_exclusion_set(mdir, vary)
    print(f"model {args.model_dir}: vary={len(vary)}")
    for cat, s in ex.items():
        if cat == "all":
            continue
        print(f"  {cat:10s} excluded={len(s)}  e.g. {sorted(s)[:6]}")
    print(f"  {'ALL':10s} excluded={len(ex['all'])}  -> target={len(vary) - len(ex['all'])}")


if __name__ == "__main__":
    main()
