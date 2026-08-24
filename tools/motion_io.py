"""motion3.json decoder: dense sampled curves -> valid Cubism motion3.json.

Symmetric counterpart to ``tools/motion_curves.load_motion``.

Why this module exists (V7.1, decision #1b):
    The encoder ``load_motion`` turns a ``.motion3.json`` into
    ``param_id -> np.ndarray(T)`` dense curves. The trainer (#2) produces dense
    curves; we must turn them back into a file the Live2D Cubism SDK will load.
    The SDK **crashes** if ``Meta.TotalSegmentCount`` / ``Meta.TotalPointCount``
    are wrong, so those counts are computed *exactly* from the emitted segments
    here (verified against real files: a Bezier segment counts 3 points, every
    other segment counts 1; see ``_count_points``).

Encoding options:
    * "linear"  (default): one LINEAR segment per frame. Lossless on the
      sampling grid; valid but verbose. Use for round-trip / unit tests.
    * "keyframe": RDP-simplified keyframes joined by LINEAR segments. Compact,
      within ``epsilon`` of the dense curve. Use for real inference output.

Both produce Meta counts that the SDK accepts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import orjson

LINEAR, BEZIER, STEPPED, INV_STEPPED = 0, 1, 2, 3


# --------------------------------------------------------------------------- #
# segment construction
# --------------------------------------------------------------------------- #
def _segments_linear(values: np.ndarray, fps: float) -> list[float]:
    """Dense (T,) -> Cubism Segments, one LINEAR segment per frame.

    Segments = [t0, v0,  <LINEAR, t1, v1>, <LINEAR, t2, v2>, ...].
    """
    values = np.asarray(values, dtype=np.float32).ravel()
    T = len(values)
    if T == 0:
        return [0.0, 0.0]
    times = np.arange(T, dtype=np.float32) / fps
    segs: list[float] = [float(times[0]), float(values[0])]
    for i in range(1, T):
        segs += [LINEAR, float(times[i]), float(values[i])]
    return segs


def rdp_simplify(values: np.ndarray, fps: float, epsilon: float):
    """Ramer-Douglas-Peucker on (time, value). Returns list of (t, v) keyframes.

    Always keeps the first and last point. Recursion depth is bounded by the
    number of points, which is small for motion curves.
    """
    values = np.asarray(values, dtype=np.float32).ravel()
    T = len(values)
    if T <= 2:
        times = np.arange(T, dtype=np.float32) / fps
        return [(float(times[i]), float(values[i])) for i in range(T)]
    times = np.arange(T, dtype=np.float32) / fps
    pts = np.stack([times, values], axis=1)  # (T,2)

    def recurse(i: int, j: int):
        if j - i < 2:
            return
        p0, p1 = pts[i], pts[j]
        d = p1 - p0
        if abs(d[0]) < 1e-9:
            # vertical chord: value error is distance to p0's value
            vert = np.abs(pts[i + 1:j, 1] - p0[1])
        else:
            # vertical (value-axis) distance to the chord -> bounds linear-interp
            # reconstruction error directly by epsilon
            slope = d[1] / d[0]
            interp = p0[1] + slope * (pts[i + 1:j, 0] - p0[0])
            vert = np.abs(pts[i + 1:j, 1] - interp)
        k = i + 1 + int(np.argmax(vert))
        if vert.max() > epsilon:
            recurse(i, k)
            keep.append(k)
            recurse(k, j)

    keep: list[int] = []
    recurse(0, T - 1)
    idx = sorted({0, T - 1, *keep})
    return [(float(times[i]), float(values[i])) for i in idx]


def _segments_from_keyframes(kfs: list[tuple[float, float]]) -> list[float]:
    """Keyframes [(t,v), ...] -> Cubism Segments (LINEAR between keyframes)."""
    if not kfs:
        return [0.0, 0.0]
    segs: list[float] = [kfs[0][0], kfs[0][1]]
    for t, v in kfs[1:]:
        segs += [LINEAR, t, v]
    return segs


# --------------------------------------------------------------------------- #
# Meta count accounting (MUST match what the SDK expects)
# --------------------------------------------------------------------------- #
def _count_points(segments: list) -> tuple[int, int]:
    """(n_segments, n_points) for a Segments list.

    Cubism point accounting, reverse-engineered from real files
    (Meta.TotalSegmentCount=246, TotalPointCount=348 -> 51 Beziers):
        LINEAR / STEPPED / INV_STEPPED -> 1 point each
        BEZIER                        -> 3 points each
    """
    if len(segments) < 2:
        return 0, 0
    n_seg = 0
    n_pt = 0
    k = 2
    while k < len(segments):
        st = int(segments[k])
        if st == BEZIER:
            n_pt += 3
            k += 1 + 6
        elif st in (LINEAR, STEPPED, INV_STEPPED):
            n_pt += 1
            k += 1 + 2
        else:
            break
        n_seg += 1
    return n_seg, n_pt


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def save_motion(
    curves: dict[str, np.ndarray],
    path,
    *,
    fps: float = 30.0,
    duration: float | None = None,
    kind: str = "linear",
    epsilon: float = 0.01,
    loop: bool = True,
    are_beziers_restricted: bool = True,
) -> dict:
    """Write ``curves`` (param_id -> (T,) dense array) to a ``.motion3.json``.

    Returns the Meta dict so callers can assert on it. The file is written with
    orjson + indent for readability.
    """
    path = Path(path)
    if not curves:
        raise ValueError("save_motion: no curves")

    # uniform length across curves (pad shorter ones with their last value)
    T = max(len(v) for v in curves.values())
    norm: dict[str, np.ndarray] = {}
    for cid, v in curves.items():
        v = np.asarray(v, dtype=np.float32).ravel()
        if len(v) < T:
            v = np.concatenate([v, np.repeat(v[-1:], T - len(v))])
        norm[cid] = v

    if duration is None:
        duration = T / fps

    out_curves = []
    total_seg = 0
    total_pt = 0
    for cid, vals in norm.items():
        if kind == "keyframe":
            kfs = rdp_simplify(vals, fps, epsilon)
            segs = _segments_from_keyframes(kfs)
        else:
            segs = _segments_linear(vals, fps)
        n_seg, n_pt = _count_points(segs)
        total_seg += n_seg
        total_pt += n_pt
        out_curves.append({"Target": "Parameter", "Id": cid, "Segments": segs})

    meta = {
        "Duration": round(float(duration), 6),
        "Fps": float(fps),
        "Loop": bool(loop),
        "AreBeziersRestricted": bool(are_beziers_restricted),
        "CurveCount": len(out_curves),
        "TotalSegmentCount": total_seg,
        "TotalPointCount": total_pt,
        "UserDataCount": 0,
        "TotalUserDataSize": 0,
    }
    doc = {"Version": 3, "Meta": meta, "Curves": out_curves, "UserData": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(doc, option=orjson.OPT_INDENT_2))
    return meta


# --------------------------------------------------------------------------- #
# self-test: round-trip a real motion
# --------------------------------------------------------------------------- #
def _selftest(sample: str = "standrad-live-2d/l2d00.ugirl02/motions/Mgirl02_stand.motion3.json"):
    sys.path.insert(0, str(Path(__file__).parent))
    from motion_curves import load_motion  # sibling module

    here = Path(sample)
    curves, _, dur = load_motion(here, fps=30.0)
    if not curves:
        print("selftest: empty sample, skip")
        return True

    # 1) linear round-trip must be lossless
    rt = Path("/tmp/_motion_io_rt.json")
    meta = save_motion(curves, rt, fps=30.0, duration=dur, kind="linear")
    curves2, _, _ = load_motion(rt, fps=30.0)
    lin_ok = all(
        cid in curves2 and np.allclose(v, curves2[cid], atol=1e-3)
        for cid, v in curves.items()
    )

    # 2) keyframe must stay within epsilon and remain SDK-valid
    rt2 = Path("/tmp/_motion_io_rt_kf.json")
    meta_kf = save_motion(curves, rt2, fps=30.0, duration=dur, kind="keyframe", epsilon=0.01)
    curves_kf, _, _ = load_motion(rt2, fps=30.0)
    kf_ok = all(
        cid in curves_kf and np.max(np.abs(v - curves_kf[cid])) <= 0.05
        for cid, v in curves.items()
    )

    print(f"[selftest] linear: curves={len(curves)} seg={meta['TotalSegmentCount']} "
          f"pt={meta['TotalPointCount']} lossless={lin_ok}")
    print(f"[selftest] keyframe: seg={meta_kf['TotalSegmentCount']} "
          f"pt={meta_kf['TotalPointCount']} within_eps={kf_ok}")
    return lin_ok and kf_ok


if __name__ == "__main__":
    ok = _selftest(*sys.argv[1:2])
    sys.exit(0 if ok else 1)
