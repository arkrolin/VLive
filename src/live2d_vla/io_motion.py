"""I/O bridges to the analysis tooling: load/save motion3.json as dense curves.

Thin wrappers over tools/motion_curves (encoder) and tools/motion_io (decoder)
so the trainer never touches Cubism JSON directly.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from motion_curves import load_motion as _load_motion  # noqa: E402
from motion_io import save_motion as _save_motion      # noqa: E402


# ---- V9: canonical action naming ------------------------------------------ #
# Enabled by set_action_map(); disabled (legacy behaviour) when the map is {}.
_VARIANT_TAIL = re.compile(r"(_[a-z]|\d)+$")
_ACTION_MAP: dict[str, str] = {}
_ACTION_MAP_ON = False


def set_action_map(rev_map):
    """rev_map: normalised core name -> canonical group name. {} disables."""
    global _ACTION_MAP, _ACTION_MAP_ON
    _ACTION_MAP = dict(rev_map or {})
    _ACTION_MAP_ON = bool(rev_map)


def _canonical(stem: str) -> str:
    s = stem
    for suf in (".motion3", ".exp3"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    parts = s.split("_", 1)
    # strip the character prefix only when it actually looks like one (Mgirl02_)
    body = parts[1] if len(parts) > 1 and re.search(r"\d", parts[0]) else s
    body = body.lower().strip()
    core = _VARIANT_TAIL.sub("", body).rstrip("_.-")
    if not core:
        core = body.rstrip("_.-") or s.lower()
    return _ACTION_MAP.get(core, core)


def list_motions(model_dir) -> dict[str, Path]:
    """action-name -> motion3.json path for one model folder."""
    model_dir = Path(model_dir)
    out: dict[str, Path] = {}
    for f in sorted(model_dir.glob("motions/*.motion3.json")):
        stem = f.stem
        if _ACTION_MAP_ON:
            action = _canonical(stem)
        else:
            # Mgirl02_weixiao -> weixiao (strip the MgirlNN_ prefix)
            action = stem.split("_", 1)[1] if "_" in stem else stem
        if action in out:
            # Same canonical name, different clip (stand_a vs stand_c). Keep
            # BOTH: collapsing them dropped 26% of the corpus (7420 -> 5505).
            k = 2
            while f"{action}~{k}" in out:
                k += 1
            action = f"{action}~{k}"
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
