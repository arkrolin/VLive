"""One-off: dump per-model Parameter-id presence to model_params.json.

Lets all alias/core analysis run offline instantly instead of re-scanning
8000+ files each time. Not part of the package; a throwaway analysis helper.
"""
import json
from live2d_vla.config import PipelineConfig
from live2d_vla import io
from live2d_vla.steps.step2_semantic import normalize_id

cfg = PipelineConfig()
out = {}
for md in io.iter_model_dirs(cfg.data_root):
    seen = set()
    for m in io.iter_motions(md):
        for c in m.curves:
            if c.target == "Parameter":
                seen.add(normalize_id(c.id))
    if seen:
        out[md.name] = sorted(seen)

json.dump(out, open("model_params.json", "w", encoding="utf-8"),
          ensure_ascii=False)
print("models:", len(out))
