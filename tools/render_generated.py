"""Render generated *_phase6.1.npz motion samples to mp4 for inspection.

Generated shards carry a synthetic model_id (e.g. l2d01.ugirl02_gen_phase6.1);
strip the suffix back to the real model dir under data_root to locate the moc3.
Each window -> one mp4, filename includes a slugged text prompt for reference.
Shard schema matches training shards (core normalized, hetero raw).
"""
import re
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.live2d_vla.config import PipelineConfig
from src.live2d_vla.schema import load_schema
from src.live2d_vla.render import OffscreenRenderer


def real_model_id(gen_id: str) -> str:
    # l2d01.ugirl02_gen_phase6.1 -> l2d01.ugirl02
    return re.sub(r"_gen.*$", "", gen_id)


def slug(text: str) -> str:
    t = re.sub(r"^A Live2D character\s*", "", text).strip().rstrip(".")
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-") or "clip"


def render_file(npz_path: Path, out_dir: Path, cfg: PipelineConfig,
                schema: dict, size: int, fps: float) -> None:
    d = np.load(npz_path, allow_pickle=True)
    gen_id = str(d["model_id"])
    mid = real_model_id(gen_id)
    core_ids = schema["core_params"]
    core = d["explicit_core"]           # (N, G, 21) normalized
    hetero = d["hetero_latent_raw"]     # (N, G, M) raw
    hetero_ids = list(d["hetero_ids"])
    hetero_targets = list(d["hetero_targets"])
    prompts = list(d["text_prompts"])
    N, G, _ = core.shape

    model3 = list((cfg.data_root / mid).glob("*.model3.json"))
    if not model3:
        print(f"  SKIP: no model3.json for {mid}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    r = OffscreenRenderer(size, size)
    try:
        r.load_model(model3[0])
        for w in range(N):
            frames = np.empty((G, size, size, 3), dtype=np.uint8)
            for fi in range(G):
                r.inject_params(core[w, fi], core_ids, hetero[w, fi],
                                hetero_ids, hetero_targets, schema)
                frames[fi] = r.render_frame(bg_color=(0.9, 0.9, 0.95))
            fp = out_dir / f"{mid}_w{w:03d}_{slug(prompts[w])}.mp4"
            iio.imwrite(fp, frames, fps=fps, codec="libx264",
                        pixelformat="yuv420p", output_params=["-crf", "18"])
            print(f"  {fp.name}  ({prompts[w]})")
    finally:
        r.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="specific npz files (default: all *phase6*.npz)")
    ap.add_argument("--out", default="D:/live-2d/videos_generated")
    args = ap.parse_args()

    cfg = PipelineConfig()
    schema = load_schema(cfg.schema_path)
    out_dir = Path(args.out)
    size = 512
    fps = cfg.target_fps

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = sorted(Path("D:/live-2d/generate").glob("*phase6*.npz"))

    if not files:
        print("no files to render")
        return 1
    for f in files:
        print(f"=== {f.name} ===")
        render_file(f, out_dir, cfg, schema, size, fps)
    print(f"done -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
