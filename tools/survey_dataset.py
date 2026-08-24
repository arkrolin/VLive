"""Fast, dependency-free survey of the Live2D corpus.

Answers the questions that decide the data plan:
  - Do model3.json Groups (EyeBlink / LipSync) actually exist? -> exclusion table
  - Is cdi3.json present?                                      -> human-readable names
  - How many motions per model, how long, how many curves?     -> supervision budget
  - Which params are physics OUTPUTS (must not be generated)?  -> target masking
  - Which params appear in motions vs only in physics?         -> active vs passive

Usage:
    uv run python tools/survey_dataset.py --root standrad-live-2d
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    args = ap.parse_args()
    root = Path(args.root)

    models = sorted(d for d in root.iterdir() if d.is_dir())
    n_models = len(models)

    has_groups = 0
    group_kinds: Counter[str] = Counter()
    has_cdi = 0
    has_physics = 0
    has_pose = 0
    has_expr = 0

    motions_per_model: list[int] = []
    motion_durations: list[float] = []
    motion_fps: Counter[float] = Counter()
    curves_per_motion: list[int] = []
    seg_types: Counter[int] = Counter()

    param_in_motion: Counter[str] = Counter()      # models where id is animated
    param_curve_hits: Counter[str] = Counter()     # total curves across corpus
    phys_out: Counter[str] = Counter()             # models where id is physics output
    phys_in: Counter[str] = Counter()
    target_kinds: Counter[str] = Counter()

    per_model_motion_params: list[int] = []

    for md in models:
        m3 = next(md.glob("*.model3.json"), None)
        if m3:
            j = load_json(m3) or {}
            g = j.get("Groups") or []
            if g:
                has_groups += 1
                for it in g:
                    group_kinds[str(it.get("Name"))] += 1
            fr = j.get("FileReferences") or {}
            if fr.get("DisplayInfo"):
                has_cdi += 1
            if fr.get("Physics"):
                has_physics += 1
            if fr.get("Pose"):
                has_pose += 1
            if fr.get("Expressions"):
                has_expr += 1
        if next(md.glob("*.cdi3.json"), None):
            has_cdi += 1

        # physics
        p3 = next(md.glob("*.physics3.json"), None)
        if p3:
            j = load_json(p3) or {}
            outs, ins = set(), set()
            for st in (j.get("PhysicsSettings") or []):
                for o in (st.get("Output") or []):
                    d = (o.get("Destination") or {}).get("Id")
                    if d:
                        outs.add(d)
                for i in (st.get("Input") or []):
                    s = (i.get("Source") or {}).get("Id")
                    if s:
                        ins.add(s)
            for d in outs:
                phys_out[d] += 1
            for s in ins:
                phys_in[s] += 1

        # motions
        mfiles = sorted((md / "motions").glob("*.json")) if (md / "motions").is_dir() else []
        mfiles = list({f.name: f for f in mfiles}.values())  # dedupe
        motions_per_model.append(len(mfiles))
        seen_ids: set[str] = set()
        for mf in mfiles:
            j = load_json(mf)
            if not j:
                continue
            meta = j.get("Meta") or {}
            if meta.get("Duration"):
                motion_durations.append(float(meta["Duration"]))
            if meta.get("Fps"):
                motion_fps[float(meta["Fps"])] += 1
            curves = j.get("Curves") or []
            curves_per_motion.append(len(curves))
            for c in curves:
                tgt = c.get("Target", "?")
                target_kinds[tgt] += 1
                cid = c.get("Id")
                if cid and tgt == "Parameter":
                    seen_ids.add(cid)
                    param_curve_hits[cid] += 1
                segs = c.get("Segments") or []
                # segments: [t0,v0, type, ...] -> walk
                k = 2
                while k < len(segs):
                    t = int(segs[k])
                    seg_types[t] += 1
                    k += 1 + (6 if t == 1 else 2 if t in (0, 2, 3) else 2)
        per_model_motion_params.append(len(seen_ids))
        for pid in seen_ids:
            param_in_motion[pid] += 1

    def pct(x: int) -> str:
        return f"{x}/{n_models} ({100*x/max(n_models,1):.1f}%)"

    def stats(xs: list) -> str:
        if not xs:
            return "n/a"
        s = sorted(xs)
        n = len(s)
        return (f"n={n} min={s[0]:.4g} p25={s[n//4]:.4g} median={s[n//2]:.4g} "
                f"p75={s[3*n//4]:.4g} p95={s[int(0.95*n)]:.4g} max={s[-1]:.4g}")

    print(f"=== corpus: {n_models} models @ {root} ===\n")
    print("-- asset availability --")
    print(f"model3 Groups non-empty : {pct(has_groups)}   kinds={dict(group_kinds)}")
    print(f"cdi3 / DisplayInfo      : {pct(has_cdi)}")
    print(f"physics3                : {pct(has_physics)}")
    print(f"pose3                   : {pct(has_pose)}")
    print(f"expressions             : {pct(has_expr)}")

    print("\n-- motion supervision budget --")
    print(f"motions per model : {stats(motions_per_model)}")
    print(f"  total motions   : {sum(motions_per_model)}")
    print(f"  models with 0   : {sum(1 for x in motions_per_model if x == 0)}")
    print(f"duration (s)      : {stats(motion_durations)}")
    print(f"  total minutes   : {sum(motion_durations)/60:.1f}")
    print(f"fps histogram     : {dict(motion_fps.most_common(5))}")
    print(f"curves per motion : {stats(curves_per_motion)}")
    print(f"animated params/model : {stats(per_model_motion_params)}")
    print(f"curve Target kinds: {dict(target_kinds.most_common())}")
    seg_names = {0: "Linear", 1: "Bezier", 2: "Stepped", 3: "InverseStepped"}
    print(f"segment types     : { {seg_names.get(k,k): v for k, v in seg_types.most_common()} }")

    print("\n-- physics vs motion overlap (the exclusion question) --")
    po = set(phys_out)
    pm = set(param_in_motion)
    print(f"distinct physics-OUTPUT ids : {len(po)}")
    print(f"distinct physics-INPUT  ids : {len(phys_in)}")
    print(f"distinct animated ids       : {len(pm)}")
    print(f"animated AND physics-output : {len(pm & po)}  <- conflict: authored despite physics")
    print(f"top physics inputs  : {[k for k,_ in phys_in.most_common(12)]}")
    print(f"top physics outputs : {[k for k,_ in phys_out.most_common(12)]}")

    print("\n-- top animated params (models covering) --")
    for pid, c in param_in_motion.most_common(30):
        flag = " [PHYS-OUT]" if pid in po else ""
        print(f"  {100*c/max(n_models,1):5.1f}%  {pid}{flag}")

    # long tail
    tail = [p for p, c in param_in_motion.items() if c == 1]
    print(f"\nsingle-model-only animated ids: {len(tail)} / {len(pm)}")


if __name__ == "__main__":
    main()
