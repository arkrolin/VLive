"""motion3.json curve parsing + dense sampling.

Live2D segment encoding (Cubism 3/4):
    Segments = [t0, v0,  <type>, ...payload...,  <type>, ...payload..., ]
      type 0 Linear         : t1, v1                      (2 numbers)
      type 1 Bezier         : c1t, c1v, c2t, c2v, t1, v1  (6 numbers)
      type 2 Stepped        : t1, v1   (hold v0 until t1, then jump to v1)
      type 3 InverseStepped : t1, v1   (jump to v1 immediately, hold to t1)

Every segment ends at (t1, v1) which becomes the next segment's start.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LINEAR, BEZIER, STEPPED, INV_STEPPED = 0, 1, 2, 3
_PAYLOAD = {LINEAR: 2, BEZIER: 6, STEPPED: 2, INV_STEPPED: 2}


def _bezier(t: float, p0, p1, p2, p3) -> float:
    """Cubic bezier value at normalized t (approximates by t-parameterization).

    Cubism evaluates by binary-searching the x-coordinate; for fingerprinting
    the t-parameterization is close enough and far cheaper.
    """
    u = 1.0 - t
    return (u * u * u * p0 + 3 * u * u * t * p1
            + 3 * u * t * t * p2 + t * t * t * p3)


def parse_segments(segments: list[float]):
    """Segments -> (starts, ends, types, v_start, v_end, c1v, c2v) as arrays.

    One row per span. Bezier control-point *values* are kept; control-point
    *times* are dropped because we evaluate by t-parameterization.
    """
    if len(segments) < 2:
        return None
    t0, v0 = float(segments[0]), float(segments[1])
    st, en, ty, va, vb, c1, c2 = [], [], [], [], [], [], []
    k = 2
    while k < len(segments):
        stype = int(segments[k])
        n = _PAYLOAD.get(stype, 2)
        pay = segments[k + 1: k + 1 + n]
        if len(pay) < n:
            break
        if stype == BEZIER:
            _c1t, c1v, _c2t, c2v, t1, v1 = (float(x) for x in pay)
            c1.append(c1v)
            c2.append(c2v)
        else:
            t1, v1 = float(pay[0]), float(pay[1])
            c1.append(v0)
            c2.append(v1)
        st.append(t0)
        en.append(t1)
        ty.append(stype)
        va.append(v0)
        vb.append(v1)
        t0, v0 = t1, v1
        k += 1 + n
    if not st:
        return None
    return (np.asarray(st, np.float32), np.asarray(en, np.float32),
            np.asarray(ty, np.int8), np.asarray(va, np.float32),
            np.asarray(vb, np.float32), np.asarray(c1, np.float32),
            np.asarray(c2, np.float32))


def sample_curve(segments: list[float], times: np.ndarray) -> np.ndarray:
    """Evaluate one curve's Segments at the given time stamps (seconds).

    Vectorised over frames: a curve has O(10) spans but O(300) frames, so all
    four segment types are evaluated with whole-array numpy ops and selected by
    boolean mask instead of a per-frame Python loop.
    """
    p = parse_segments(segments)
    if p is None:
        out = np.empty(len(times), dtype=np.float32)
        out.fill(float(segments[1]) if len(segments) >= 2 else 0.0)
        return out
    st, en, ty, va, vb, c1, c2 = p

    idx = np.clip(np.searchsorted(st, times, side="right") - 1, 0, len(st) - 1)
    a, b = st[idx], en[idx]
    v_a, v_b = va[idx], vb[idx]
    k1, k2 = c1[idx], c2[idx]
    tt = ty[idx]

    u = np.clip((times - a) / np.maximum(b - a, 1e-9), 0.0, 1.0)

    out = v_a + (v_b - v_a) * u                      # LINEAR (default)
    m = tt == BEZIER
    if m.any():
        uu = u[m]
        w = 1.0 - uu
        out[m] = (w ** 3 * v_a[m] + 3 * w * w * uu * k1[m]
                  + 3 * w * uu * uu * k2[m] + uu ** 3 * v_b[m])
    m = tt == STEPPED
    if m.any():
        out[m] = np.where(u[m] >= 1.0, v_b[m], v_a[m])
    m = tt == INV_STEPPED
    if m.any():
        out[m] = np.where(u[m] <= 0.0, v_a[m], v_b[m])

    # frames before the first span / after the last one clamp to the endpoints
    out = np.where(times <= st[0], va[0], out)
    out = np.where(times >= en[-1], vb[-1], out)
    return out.astype(np.float32)


def curve_extent(segments: list[float]) -> tuple[float, float, float]:
    """Exact (lo, hi, first_value) of a curve without sampling it.

    Every number a curve can take is bounded by its keyframe values and, for
    Bezier segments, its control-point values (a cubic Bezier lies inside the
    convex hull of its four control points). So walking the value slots gives an
    exact flatness test in O(n_segments) instead of O(n_frames * n_curves).

    Used by the flatness triage in build_static_pose.py; anything that needs the
    actual waveform still goes through sample_curve().
    """
    if len(segments) < 2:
        return 0.0, 0.0, 0.0
    v0 = float(segments[1])
    lo = hi = v0
    k = 2
    while k < len(segments):
        stype = int(segments[k])
        n = _PAYLOAD.get(stype, 2)
        pay = segments[k + 1: k + 1 + n]
        if len(pay) < n:
            break
        if stype == BEZIER:
            vals = (float(pay[1]), float(pay[3]), float(pay[5]))
        else:
            vals = (float(pay[1]),)
        for v in vals:
            if v < lo:
                lo = v
            elif v > hi:
                hi = v
        k += 1 + n
    return lo, hi, v0


def load_motion(path: Path, fps: float = 30.0):
    """-> (dict[param_id -> np.ndarray(T)], dict[param_id -> seg-type counts], duration)"""
    try:
        j = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}, {}, 0.0
    meta = j.get("Meta") or {}
    dur = float(meta.get("Duration") or 0.0)
    if dur <= 0:
        return {}, {}, 0.0
    T = max(int(round(dur * fps)), 2)
    times = np.arange(T, dtype=np.float32) / fps

    curves: dict[str, np.ndarray] = {}
    segstats: dict[str, np.ndarray] = {}
    for c in (j.get("Curves") or []):
        if c.get("Target") != "Parameter":
            continue
        cid = c.get("Id")
        if not cid:
            continue
        segs = c.get("Segments") or []
        curves[cid] = sample_curve(segs, times)
        # segment type histogram [linear, bezier, stepped, invstepped]
        h = np.zeros(4, dtype=np.float32)
        k = 2
        while k < len(segs):
            st = int(segs[k])
            if 0 <= st <= 3:
                h[st] += 1
            k += 1 + _PAYLOAD.get(st, 2)
        segstats[cid] = h
    return curves, segstats, dur
