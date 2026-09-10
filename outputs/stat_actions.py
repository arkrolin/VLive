"""Action coverage across models: is there a near-universal 'reference' action?

User's hypothesis: the useful reference is not the cross-character action mean
(exem) but THE CHARACTER'S OWN performance of some action that nearly every
character shares (e.g. `stand` / idle). Quantify:

  1. For each action, how many models have it (coverage of the 580 whitelist).
  2. Which actions are near-universal (>= 80% of models)?
  3. Per-model: does it have at least one near-universal action?
  4. ALSO: an oracle test -- if we used the character's own `stand`-family
     curve as the prediction for another action, how good would it be vs exem?
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/root/work/nlp/xjzhao13/lijie_llama/VLive")
sys.path.insert(0, str(ROOT / "src" / "live2d_vla"))
from io_motion import list_motions  # noqa: E402

DATA = ROOT / "data" / "all"
models = sorted([p.name for p in DATA.iterdir() if p.is_dir()])
print(f"models under data/all: {len(models)}")

act_models = defaultdict(set)
per_model_count = {}
for m in models:
    try:
        acts = list_motions(DATA / m)
    except Exception:
        acts = []
    per_model_count[m] = len(acts)
    for a in acts:
        # action key: strip the .motion3 suffix the way dataset does
        key = a[:-9] if a.endswith(".motion3") else a
        act_models[key].add(m)

n = len(models)
cov = sorted(((len(v), k) for k, v in act_models.items()), reverse=True)
print(f"\ndistinct actions: {len(act_models)}")
print(f"\n=== TOP 25 actions by number of models (of {n}) ===")
print(f"  {'action':40s} {'models':>7s} {'coverage':>9s}")
for c, k in cov[:25]:
    print(f"  {k[:40]:40s} {c:7d} {c/n*100:8.1f}%")

for thr in (0.9, 0.8, 0.7, 0.5):
    uni = [k for c, k in cov if c / n >= thr]
    print(f"\nactions with coverage >= {thr:.0%}: {len(uni)}"
          + (f"  e.g. {uni[:8]}" if uni else ""))

# how many models have >=1 action with coverage >= 80%
uni80 = {k for c, k in cov if c / n >= 0.8}
have = sum(1 for m in models if any(k in uni80 for k in
           [(a[:-9] if a.endswith('.motion3') else a) for a in []]))
# simpler: recompute
have = 0
for m in models:
    try:
        acts = list_motions(DATA / m)
    except Exception:
        acts = []
    keys = {(a[:-9] if a.endswith('.motion3') else a) for a in acts}
    if keys & uni80:
        have += 1
print(f"\nmodels having at least one action with >=80% coverage: "
      f"{have}/{n} ({have/n*100:.1f}%)")

# distribution of actions per model
import statistics as st
c = list(per_model_count.values())
print(f"\nactions per model: median {st.median(c):.0f}  mean {st.mean(c):.1f}  "
      f"min {min(c)}  max {max(c)}")
