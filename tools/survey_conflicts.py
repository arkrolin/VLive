"""Two correctness checks that decide what goes into the generation target.

1) physics conflict, WITHIN a model: is a parameter that this model's
   physics3.json drives as an OUTPUT also hand-authored in this model's
   motions? Cross-model aggregation would overstate this, so we check per model.

2) discrete channels: how many parameters are effectively switches (few
   distinct values / Stepped-dominated)? Gaussian diffusion on those produces
   illegal in-between values, so they need a classification head instead.

Usage:
    uv run --no-project --with numpy python tools/survey_conflicts.py --limit 120
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from motion_curves import load_motion  # noqa: E402


def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="standrad-live-2d")
    ap.add_argument("--limit", type=int, default=120)
    args = ap.parse_args()
    root = Path(args.root)

    models = sorted(d for d in root.iterdir() if d.is_dir())
    step = max(len(models) // args.limit, 1)
    models = models[::step][:args.limit]

    n_with_phys = 0
    conflict_counts, phys_out_counts, animated_counts = [], [], []
    conflict_ids: Counter[str] = Counter()
    # per-model fraction of authored curves that hit a physics output
    conflict_frac = []

    # discrete channel analysis
    disc_stats = Counter()
    tot_params = 0
    stepped_dominant: Counter[str] = Counter()
    partop_curves = 0
    param_curves = 0

    for md in models:
        p3 = next(md.glob("*.physics3.json"), None)
        outs: set[str] = set()
        if p3:
            j = load_json(p3) or {}
            for st in (j.get("PhysicsSettings") or []):
                for o in (st.get("Output") or []):
                    d = (o.get("Destination") or {}).get("Id")
                    if d:
                        outs.add(d)

        mdir = md / "motions"
        if not mdir.is_dir():
            continue
        traj: dict[str, list[np.ndarray]] = defaultdict(list)
        segs: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.float32))
        files = list({f.name: f for f in mdir.glob("*.json")}.values())
        for mf in files:
            curves, segstats, _ = load_motion(mf, 30.0)
            for pid, arr in curves.items():
                traj[pid].append(arr)
                segs[pid] += segstats.get(pid, np.zeros(4, dtype=np.float32))
            j = load_json(mf) or {}
            for c in (j.get("Curves") or []):
                if c.get("Target") == "PartOpacity":
                    partop_curves += 1
                elif c.get("Target") == "Parameter":
                    param_curves += 1
        if not traj:
            continue

        animated = set(traj)
        if outs:
            n_with_phys += 1
            conf = animated & outs
            conflict_counts.append(len(conf))
            phys_out_counts.append(len(outs))
            animated_counts.append(len(animated))
            conflict_frac.append(len(conf) / max(len(animated), 1))
            for c in conf:
                conflict_ids[c] += 1

        for pid, vs in traj.items():
            tot_params += 1
            cat = np.concatenate(vs)
            uniq = np.unique(np.round(cat, 3))
            h = segs[pid]
            tot = float(h.sum())
            stepped_ratio = float(h[2] / tot) if tot > 0 else 0.0
            if len(uniq) <= 1:
                disc_stats["constant (1 value)"] += 1
            elif len(uniq) == 2:
                disc_stats["binary (2 values)"] += 1
            elif len(uniq) <= 5:
                disc_stats["few-level (3-5)"] += 1
            elif stepped_ratio > 0.8:
                disc_stats["stepped-dominant (>80%)"] += 1
            elif stepped_ratio > 0.5:
                disc_stats["stepped-mixed (50-80%)"] += 1
            else:
                disc_stats["continuous"] += 1
            if stepped_ratio > 0.8:
                stepped_dominant[pid] += 1

    print(f"=== physics conflict, WITHIN model ({n_with_phys} models with physics) ===")
    cf = np.array(conflict_frac) if conflict_frac else np.zeros(1)
    print(f"  physics outputs per model : median={int(np.median(phys_out_counts))}")
    print(f"  animated params per model : median={int(np.median(animated_counts))}")
    print(f"  BOTH (conflict) per model : median={int(np.median(conflict_counts))} "
          f"mean={np.mean(conflict_counts):.1f} max={max(conflict_counts)}")
    print(f"  conflict as share of authored params: "
          f"median={100*np.median(cf):.1f}%  mean={100*cf.mean():.1f}%")
    print(f"  models with ZERO conflict : "
          f"{sum(1 for c in conflict_counts if c == 0)}/{len(conflict_counts)}")
    print(f"\n  top conflicting ids (hand-authored despite physics driving them):")
    for pid, c in conflict_ids.most_common(14):
        print(f"    {100*c/max(n_with_phys,1):5.1f}%  {pid}")

    print(f"\n=== discrete vs continuous channels ({tot_params} model-param pairs) ===")
    for k, v in disc_stats.most_common():
        print(f"  {100*v/max(tot_params,1):5.1f}%  {k}")
    tot_c = param_curves + partop_curves
    print(f"\n  curve targets: Parameter {100*param_curves/max(tot_c,1):.1f}%  "
          f"PartOpacity {100*partop_curves/max(tot_c,1):.1f}%")
    print(f"\n  most frequently stepped-dominant params:")
    for pid, c in stepped_dominant.most_common(12):
        print(f"    x{c:3d}  {pid}")


if __name__ == "__main__":
    main()
