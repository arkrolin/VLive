"""Action coverage: raw names vs V9 semantic-consolidated names.

Q: does a near-universal reference action exist (stand / idle family)?
Reports coverage both on raw action names and after the V9 semantic map
(193 names -> 33 groups), since consolidation merges stand_a/stand_c -> stand.
"""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/root/work/nlp/xjzhao13/lijie_llama/VLive")
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))
import io_motion  # noqa: E402
from io_motion import list_motions  # noqa: E402

DATA = ROOT / "data" / "all"
# strip only the ".motion3" tail (8 chars), not 9
SUF = ".motion3"
def norm(a):
    return a[:-len(SUF)] if a.endswith(SUF) else a

def scan(tag):
    models = sorted([p.name for p in DATA.iterdir() if p.is_dir()])
    act_models = defaultdict(set)
    for m in models:
        try:
            acts = list(list_motions(DATA / m).keys())
        except Exception:
            acts = []
        for a in acts:
            act_models[norm(a)].add(m)
    n = len(models)
    cov = sorted(((len(v), k) for k, v in act_models.items()), reverse=True)
    print(f"\n=========== {tag} ===========")
    print(f"  distinct actions: {len(act_models)}   models: {n}")
    print(f"  {'action':32s} {'models':>7s} {'coverage':>9s}")
    for c, k in cov[:15]:
        print(f"  {k[:32]:32s} {c:7d} {c/n*100:8.1f}%")
    for thr in (0.9, 0.7, 0.5, 0.4, 0.3):
        u = [k for c, k in cov if c / n >= thr]
        print(f"  coverage >= {thr:.0%}: {len(u)} actions" + (f" -> {u[:10]}" if u else ""))
    return cov, n, act_models, models

# --- 1. raw names (no semantic map) ---
cov_raw, n, act_raw, models = scan("RAW action names")

# --- 2. with the V9 semantic map enabled ---
try:
    io_motion._ACTION_MAP_ON = True
    import orjson
    amap = orjson.loads((ROOT / "outputs" / "action_semantic_map.json").read_bytes())
    io_motion._ACTION_MAP = amap
    print("\n[V9 semantic map loaded]")
except Exception as e:
    print(f"\n[could not load semantic map: {e}]")

cov_map, n2, act_map, _ = scan("V9 SEMANTIC-CONSOLIDATED action names")

# --- 3. does each model have a 'stand-like' action? ---
STANDISH = ("stand", "idle", "stan", "idl", "wait", "breath", "default", "normal")
for tag, act_models in (("RAW", act_raw), ("V9-MAP", act_map)):
    have = sum(1 for m in models
               if any(any(s in k for s in STANDISH) for k in
                      [k for k, v in act_models.items() if m in v]))
    print(f"\n[{tag}] models having at least one stand/idle-like action: "
          f"{have}/{len(models)} ({have/len(models)*100:.1f}%)")
