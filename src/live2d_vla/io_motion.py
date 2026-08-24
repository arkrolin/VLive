"""I/O bridges to the analysis tooling: load/save motion3.json as dense curves.

Thin wrappers over tools/motion_curves (encoder) and tools/motion_io (decoder)
so the trainer never touches Cubism JSON directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from motion_curves import load_motion as _load_motion  # noqa: E402
from motion_io import save_motion as _save_motion      # noqa: E402


def list_motions(model_dir) -> dict[str, Path]:
    """action-name -> motion3.json path for one model folder."""
    model_dir = Path(model_dir)
    out = {}
    for f in sorted(model_dir.glob("motions/*.motion3.json")):
        # Mgirl02_weixiao -> weixiao (strip the MgirlNN_ prefix)
        stem = f.stem
        if "_" in stem:
            action = stem.split("_", 1)[1]
        else:
            action = stem
        out[action] = f
    return out


def _resample(arr: np.ndarray, T: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).ravel()
    n = len(arr)
    if n == 0:
        return np.zeros(T, np.float32)
    if n == T:
        return arr
    old = np.linspace(0.0, 1.0, n)
    new = np.linspace(0.0, 1.0, T)
    return np.interp(new, old, arr).astype(np.float32)


def load_target_curves(model_dir, action: str, target_params: list[str],
                       fps: float = 30.0, T: int = 48) -> dict[str, np.ndarray]:
    """Load one motion's curves for the requested target params, resampled to T frames."""
    mdir = Path(model_dir)
    motions = list_motions(mdir)
    if action not in motions:
        return {}
    curves, _, _ = _load_motion(motions[action], fps=fps)
    if not curves:
        return {}
    out = {}
    for p in target_params:
        if p in curves:
            out[p] = _resample(curves[p], T)
    return out


def save_curves(model_dir, action: str, curves: dict[str, np.ndarray],
                fps: float = 30.0, T: int = 48, out_dir=None) -> Path:
    """Write dense curves to a motion3.json (keyframe-compressed, SDK-valid)."""
    if out_dir is None:
        out_dir = Path(model_dir) / "motions"
    out_dir = Path(out_dir)
    path = out_dir / f"Mgen_{action}.motion3.json"
    _save_motion(curves, path, fps=fps, duration=T / fps, kind="keyframe", epsilon=0.01)
    return path
